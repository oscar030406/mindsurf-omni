"""Put spoken noise back into clean text, so a polisher has something to remove.

Production runs speech -> SenseVoice -> polish. Training a polisher needs pairs
of (what the recogniser heard, what the person meant), and the corpus we have
is written text: punctuated, tidy, filler-free. So the pairs are made by going
round the other way -- inject the noise, have it read aloud, hear it back. The
half that is synthetic is the injection; edge-tts and SenseVoice are the same
two components production uses, so the error distribution in the result comes
from the real pipeline rather than from someone's idea of what ASR gets wrong.

What this cannot make is a real disfluency. A person restarts a clause,
stretches a vowel and repeats a word half-said. This inserts whole filler words
at clause boundaries and a synthesiser reads them as words, cleanly. So any
retention number built on it is about lexical fillers only, and is an upper
bound on how much of a person's disfluency survives recognition: a clearly
articulated 那个 is the easiest possible case for the recogniser to keep.

    python scripts/inject_fillers.py --texts configs/talker_texts_zh_v1.jsonl \
        --output artifacts/polish/texts_filler.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

# The vocabulary lives with the stage that removes it. Imported rather than
# copied: the decoder is allowed to step over exactly these words, and two
# lists would drift.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mindsurf_omni.service.polish import (  # noqa: E402
    BRIDGING_FILLERS,
    LEADING_FILLERS,
)

# Clause boundaries in written Chinese. The insertion points are here because
# that is where a speaker hesitates: mid-clause fillers happen but they are the
# rarer half, and putting one there also breaks a word more often than not.
_CLAUSE = re.compile(r"[，。！？；：、]")


def split_clauses(text: str) -> list[str]:
    """Clauses with their punctuation kept, in order."""
    pieces, start = [], 0
    for match in _CLAUSE.finditer(text):
        pieces.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return [piece for piece in pieces if piece]


def word_boundaries(clause: str) -> list[int]:
    """Positions inside a clause where a speaker could pause, by word.

    Segmented rather than counted: a filler dropped at a random character index
    splits a word (微波然后炉), which is a different noise from the one being
    modelled and one the recogniser would hear as a different word.

    Empty for a clause with no interior boundary -- short clauses simply get no
    interior filler.
    """
    import jieba

    positions: list[int] = []
    offset = 0
    for word in jieba.cut(clause):
        offset += len(word)
        if 2 <= offset <= len(clause) - 2:
            positions.append(offset)
    return positions


# What a repeated span looks like when a person does it, measured against
# CS2W's human annotation over 3807 real repetitions:
#
#            单字   двух字   三字以上   中间隔开
#   真人      58%     31%      11%       49%
#   我们      10%     88%       2%       33%
#
# The injector produced the second row because it copied the clause's first two
# characters and pasted them back adjacent -- one shape, always. And the model
# is worst at exactly the shape it saw least: repetition recall by shape is
# 0.755 adjacent against 0.426 gapped, and single-character repeats are the low
# point of both (0.662 and 0.343). Three measurements, one cause.
REPEAT_LENGTHS = ((1, 0.58), (2, 0.31), (3, 0.07), (4, 0.03), (5, 0.01))
# Half of what people repeat has something in between -- a breath, a comma, a
# filler -- and 但是最后，但是最后 is as ordinary as 我我.
REPEAT_GAPPED = 0.49
GAP_FILLERS = ("嗯", "呃", "那个", "就是", "，")


def repeat_of(clause: str, rng: random.Random) -> str:
    """A repetition shaped the way a speaker makes one, or nothing.

    Returned as the text to insert *before* the clause, gap included, so the
    caller stays a one-liner and the shape lives here.
    """
    body = clause.rstrip("，。！？；：、")
    if not body:
        return ""
    wanted = rng.choices(
        [size for size, _ in REPEAT_LENGTHS], [share for _, share in REPEAT_LENGTHS]
    )[0]
    size = min(wanted, len(body))
    if size < 1:
        return ""
    head = body[:size]
    if not head.strip("，。！？；：、"):
        return ""
    if rng.random() < REPEAT_GAPPED:
        return head + rng.choice(GAP_FILLERS)
    return head


def inject(
    text: str,
    rng: random.Random,
    rate: float,
    cap: int,
    repetition_share: float = 1 / 3,
) -> tuple[str, list[dict[str, Any]]]:
    """The same sentence as someone would say it out loud, and what was added.

    Repetition is included because it is the second thing every transcript of
    real speech has and the polisher has to remove it too -- 我我觉得 is not a
    filler word, and a model trained only on 嗯 and 那个 would leave it in.

    ``repetition_share`` is a knob because the two halves are not equally
    learned. Measured on 986 held-out sentences, the best arm clears 0.995 of
    the vocabulary filler and 0.749 of the repetition; the tagger reaches 0.983
    on filler and 0.437 on repetition, because a per-token head can recognise 嗯
    from the token alone but cannot compare two spans. Repetition is what the
    filler-clearance line is now failing on, and it is a third as frequent as
    filler in the training data.
    """
    injections: list[dict[str, Any]] = []
    spoken = []
    for index, clause in enumerate(split_clauses(text)):
        if len(injections) < cap and rng.random() < rate:
            pool = LEADING_FILLERS if index == 0 else LEADING_FILLERS + BRIDGING_FILLERS
            token = rng.choice(pool)
            # A comma after the filler, because that is how it is punctuated
            # when someone writes speech down, and because without one the
            # synthesiser runs it into the next word.
            spoken.append(f"{token}，")
            injections.append({"kind": "filler", "token": token, "clause": index})
        if len(injections) < cap and rng.random() < rate * repetition_share:
            said_twice = repeat_of(clause, rng)
            if said_twice:
                spoken.append(said_twice)
                injections.append(
                    {"kind": "repetition", "token": said_twice, "clause": index}
                )
        # Inside the clause as well as in front of it. The first round put every
        # filler at a boundary, and the polisher learned the boundary rather
        # than the word: measured on the held-out set, a filler at the front of
        # a sentence survived 5.4% of the time and one further in 21.4% -- four
        # times as often, with no length effect to explain it. A speaker
        # hesitates mid-clause too, and if the training data never does, the
        # model has no reason to.
        inside = clause.rstrip("，。！？；：、")
        cuts = word_boundaries(inside)
        if len(injections) < cap and cuts and rng.random() < rate / 2:
            # At a word boundary, not any character: 微波然后炉 is not a
            # hesitation, it is a broken word, and the recogniser would hear it
            # as one too.
            cut = rng.choice(cuts)
            token = rng.choice(LEADING_FILLERS)
            # No comma this time: nobody writes one there, and the recogniser
            # would not put one there either.
            spoken.append(inside[:cut] + token + inside[cut:] + clause[len(inside) :])
            injections.append({"kind": "filler", "token": token, "clause": index, "inside": True})
            continue
        spoken.append(clause)
    return "".join(spoken), injections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, default=Path("configs/talker_texts_zh_v1.jsonl"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rate",
        type=float,
        default=0.35,
        help="chance of a filler at each clause boundary. 0.35 puts roughly one "
        "in every two clauses, which is the order spontaneous Mandarin sits at",
    )
    parser.add_argument("--cap", type=int, default=3, help="most injections in one sentence")
    parser.add_argument(
        "--repetition-share",
        type=float,
        default=1 / 3,
        help="repetition chance as a share of --rate. The default third is what "
        "the first three rounds used; repetition is the half the filler line now "
        "fails on, so this is the knob for testing whether more of it is learnable",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.texts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no texts in {args.texts}")

    written = []
    for row in rows:
        # Seeded per row rather than per file: adding a sentence tomorrow must
        # not change what was injected into the ones already synthesised.
        rng = random.Random(f"{args.seed}:{row['id']}")
        spoken, injections = inject(row["text"], rng, args.rate, args.cap, args.repetition_share)
        written.append(
            {
                **row,
                # The field the synthesis script reads, so this file drops into
                # the existing harness without a flag.
                "text": spoken,
                "clean_text": row["text"],
                "injections": injections,
                "filler_count": sum(1 for item in injections if item["kind"] == "filler"),
                "repetition_count": sum(1 for item in injections if item["kind"] == "repetition"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in written) + "\n",
        encoding="utf-8",
    )

    total = sum(len(row["injections"]) for row in written)
    untouched = sum(1 for row in written if not row["injections"])
    print(f"{len(written)} 句写到 {args.output}")
    print(
        f"  注入 {total} 处（口语词 {sum(row['filler_count'] for row in written)}、"
        f"重复 {sum(row['repetition_count'] for row in written)}），"
        f"每句 {total / len(written):.2f}，一处没注入的 {untouched} 句"
    )


if __name__ == "__main__":
    main()
