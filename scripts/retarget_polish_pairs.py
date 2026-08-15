"""Rebuild a pair's target from the transcript instead of the corpus sentence.

The target used to be the corpus sentence the clip was made from, which quietly
asked the polisher for three jobs at once: delete the spoken filler, fix what
the recogniser misheard, and re-punctuate to match the corpus. Only the first
is in the product. The floor round measured the second at 0.6% and gave it up
(DECISIONS §16), and the third does not exist at all -- SenseVoice writes
punctuation, in the same places the original had it, and ``normalise_for_cer``
strips punctuation from both sides so the criteria never look at it.

The third job is not free, though. On the 156-sentence held-out set it supplies
191 of 659 delete labels, and every one of them is ambiguous: a comma is
deleted 98 times and kept 253 times, a full stop deleted 30 and kept 195. A
tagger reading those labels is being taught that punctuation is usually
deletable, which for a dictation product is the wrong lesson and not one
anybody asked for.

So the target here is the transcript with the injected spans taken out, and
nothing else changed. What that buys, with no re-synthesis, measured by a
lookup table fitted on train and read on val:

    删除精确率 @ 召回 0.50   0.930 -> 0.967
    删除精确率 @ 召回 0.70   0.664 -> 0.863
    删除精确率 @ 召回 0.80   0.393 -> 0.441

The 0.80 column is where the trained tagger sits (0.394), and it barely moves --
that end of the curve is the genuinely ambiguous part and this does not touch
it. The gain is in the high-precision half, which is the half the acceptance
criteria live in: content retention has to clear 0.98.

That is a claim about the labels, made on 3015 training sentences. It is not
yet a claim about a trained model: the arm this produced (``sft_polish6``) has
every point estimate at least as good as the round before it and clears two
criteria that had never been cleared together, but a paired bootstrap over the
156 held-out sentences puts a zero inside all three of those intervals. The one
difference that does clear zero is invention, and it went the wrong way
(+0.0033, [+0.0007, +0.0064], still under the 0.02 line). Read this file as
removing labels nothing scores, not as a measured win.

The pairs keep their original target under ``target_corpus`` so a run can be
scored against the same sentence the earlier rounds were scored against.

    python scripts/retarget_polish_pairs.py --pairs artifacts/polish_train/pairs_v2.jsonl \
        --pool artifacts/polish_train/pool.jsonl --output artifacts/polish_train/pairs_v3.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

PUNCTUATION = set("，。！？；：、")


def injected_spans(clean: str, spoken: str) -> list[bool]:
    """Which characters of ``spoken`` the injector added.

    Read off the diff rather than from the injection record: the record says
    what was inserted and into which clause, not at which character, and the
    diff is exact because the injector only ever inserts.
    """
    added = [False] * len(spoken)
    for tag, _i1, _i2, start, end in difflib.SequenceMatcher(
        None, clean, spoken, autojunk=False
    ).get_opcodes():
        if tag in {"insert", "replace"}:
            for index in range(start, end):
                added[index] = True
    return added


def strip_injected(clean: str, spoken: str, heard: str) -> str:
    """The transcript with the injected filler removed and nothing else touched.

    Two alignments, because the injection is known against the spoken text and
    has to be deleted from the transcript, and those two differ wherever the
    recogniser misheard. A span the recogniser rewrote is only dropped when the
    whole of its spoken side was injected -- a partial overlap means the
    recogniser merged the filler into a real word, and deleting that would take
    the word with it.
    """
    added = injected_spans(clean, spoken)
    drop = [False] * len(heard)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, spoken, heard, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                if added[i1 + offset]:
                    drop[j1 + offset] = True
        elif tag == "replace" and i2 > i1 and all(added[index] for index in range(i1, i2)):
            for index in range(j1, j2):
                drop[index] = True

    # The filler carried a comma of its own, and taking the words out leaves it
    # against whatever punctuation followed. Doubled and leading marks go; the
    # rest of the recogniser's punctuation stays exactly where it put it.
    out: list[str] = []
    for index, char in enumerate(heard):
        if drop[index]:
            continue
        if char in PUNCTUATION and (not out or out[-1] in PUNCTUATION):
            continue
        out.append(char)
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path, help="the clean sentences")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pool = {}
    for line in args.pool.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pool[row["id"]] = row["text"]

    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    written, fell_back, unchanged = [], 0, 0
    for row in rows:
        clean = pool.get(row["id"])
        target = strip_injected(clean, row["spoken"], row["source"]) if clean is not None else ""
        # An empty result means the two alignments disagreed badly enough to eat
        # the sentence. Keep the old target rather than train on nothing.
        if not target:
            target, fell_back = row["target"], fell_back + 1
        unchanged += target == row["target"]
        written.append({**row, "target": target, "target_corpus": row["target"]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in written) + "\n",
        encoding="utf-8",
    )
    print(f"{len(written)} 对写到 {args.output}")
    print(f"  目标没变的 {unchanged} 对，对齐失败退回原目标的 {fell_back} 对")


if __name__ == "__main__":
    main()
