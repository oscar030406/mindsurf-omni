"""Targets for the graded constraint: content still copied, punctuation learnt.

§18 pointed the target at the transcript with the injected spans removed, and
gave the reason: the corpus original's punctuation could only ever be reached
by *deleting* the recogniser's, because the decode cannot insert. 185 of 659
delete labels were punctuation, and teaching a dictation model that punctuation
is usually deletable was the wrong lesson to be teaching by accident.

Once punctuation may be inserted, that constraint is gone and the question §18
could not ask becomes askable: point the target at the human's punctuation and
let the model learn where marks belong. Content stays projected onto the
transcript, because a target the decode cannot reach is what makes it jump
forward and take content with it -- 45.3% of the clean originals are not
subsequences of their transcript, and that measurement has not changed.

    python scripts/build_graded_targets.py \
        --pairs artifacts/polish_train/pairs_union.jsonl \
        --output artifacts/polish_train/pairs_graded.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKS = set("，。！？；：、,.!?;:")


def strip_marks(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Bare characters, and each mark with the bare index it follows."""
    bare: list[str] = []
    marks: list[tuple[int, str]] = []
    for ch in text:
        if ch in MARKS:
            marks.append((len(bare), ch))
        else:
            bare.append(ch)
    return "".join(bare), marks


def project(source: str, wanted: str) -> str:
    """The longest part of ``wanted`` that ``source`` actually says, in order."""
    import difflib

    matcher = difflib.SequenceMatcher(None, source, wanted, autojunk=False)
    return "".join(source[b.a : b.a + b.size] for b in matcher.get_matching_blocks())


def forward_tn(text: str) -> str:
    """Chinese numerals to Arabic, so a human's 二零一九 can reach an ITN 2019.

    Without this every digit in every clip is destroyed:
    爱数智慧语音采集2019年10月25日 against a human's 二零一九年十月二十五日
    shares only 年月日, so the projection reads the whole date as content to
    delete and the target teaches exactly that. It is the same mismatch that
    made our AISHELL reading 0.0638 where the protocol-matched number is 0.0295.
    """
    try:
        import cn2an
    except ImportError:  # the builder still works, it just loses the digits
        return text
    try:
        return str(cn2an.transform(text, "cn2an"))
    except Exception:  # noqa: BLE001 - a sentence cn2an cannot parse stays as it is
        return text


def graded_target(source: str, human: str) -> str:
    """Content the transcript can reach, punctuated the way the human did."""
    human = forward_tn(human)
    source_bare, _ = strip_marks(source)
    human_bare, human_marks = strip_marks(human)

    kept = project(source_bare, human_bare)
    if not kept:
        return ""

    # Every human mark sits after some human character; carry it to the same
    # character in the projection. A mark whose character did not survive the
    # projection is dropped rather than guessed at a neighbour.
    import difflib

    where: dict[int, int] = {}
    for block in difflib.SequenceMatcher(
        None, human_bare, kept, autojunk=False
    ).get_matching_blocks():
        for k in range(block.size):
            where[block.a + k] = block.b + k

    at: dict[int, list[str]] = {}
    for index, mark in human_marks:
        if index == 0:
            continue  # a leading mark has nothing to attach to
        landed = where.get(index - 1)
        if landed is not None:
            at.setdefault(landed + 1, []).append(mark)

    out: list[str] = []
    for i, ch in enumerate(kept):
        out.extend(at.get(i, ()))
        out.append(ch)
    out.extend(at.get(len(kept), ()))
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    written, unchanged, dropped = [], 0, 0
    for row in rows:
        # Ours carries the corpus original; CS2W's target is already a human's
        # own writing, punctuation included.
        human = row.get("target_corpus") or row["target"]
        target = graded_target(row["source"], human)
        if not target:
            dropped += 1
            continue
        unchanged += target == row["target"]
        written.append({**row, "target": target, "target_before_grading": row["target"]})

    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in written), encoding="utf-8"
    )
    print(f"{len(written)} 对写出，{dropped} 对投影后为空丢弃")
    print(f"其中 {unchanged} 对（{unchanged / max(1, len(written)):.1%}）和原目标一字不差")

    marks_before = sum(sum(ch in MARKS for ch in r["target_before_grading"]) for r in written)
    marks_after = sum(sum(ch in MARKS for ch in r["target"]) for r in written)
    print(f"目标里的标点：{marks_before} → {marks_after}")


if __name__ == "__main__":
    main()
