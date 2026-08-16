"""CS2W's per-character labels as training pairs for the tagger.

Every reading of this line so far was taken on disfluency we injected
ourselves, where the delete label is equivalent to "did we put this here".
CS2W is 7237 sentences of transcribed real conversation with the 1 (filler) /
2 (repetition or restart) / 0 (keep) marks written by people, and its
repetition half is the kind our injector cannot make: a speaker changing
course mid-clause rather than repeating a span cleanly.

The tagger trains from (source, target) and derives its labels by diffing, so
the conversion is to delete what a person marked and keep the rest. Nothing is
inferred: the target is the source minus the marked characters, in order.

Two things this conversion does not fix, and a reader of the numbers needs
both:

* **The input is a human transcript, not SenseVoice output.** Punctuation and
  sentence breaks are written by a person. The difference in punctuation
  convention alone accounts for about 28% of the delete labels, measured, so
  absolute values are not comparable with the self-made holdout.
* **The register is conversation, not dictation.** A tagger trained here is
  trained on people talking to each other, and the product is somebody
  dictating a note.

    python scripts/cs2w_to_pairs.py --train <cs2w>/train.jsonl \\
        --val <cs2w>/val.jsonl --output artifacts/polish_train/pairs_cs2w.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(path: Path, split: str) -> tuple[list[dict[str, str]], int]:
    """Rows as (source, target) pairs, and how many were dropped as unusable."""
    rows: list[dict[str, str]] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source, labels = record["content"], record["disfluency_label"]
        # A label string shorter than its text would misalign every position
        # after the gap, which is a mislabelling no loss curve can show.
        if len(source) != len(labels):
            dropped += 1
            continue
        target = "".join(
            character for character, mark in zip(source, labels, strict=True) if mark not in "12"
        )
        rows.append({"source": source, "target": target, "split": split, "id": record.get("id")})
    return rows, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    rows, dropped = convert(arguments.train, "train")
    print(f"train {len(rows)} 句（对不齐丢掉 {dropped}）")
    if arguments.val:
        held, dropped_val = convert(arguments.val, "val")
        print(f"val   {len(held)} 句（对不齐丢掉 {dropped_val}）")
        rows += held

    unchanged = sum(1 for row in rows if row["source"] == row["target"])
    deleted = sum(len(row["source"]) - len(row["target"]) for row in rows)
    total = sum(len(row["source"]) for row in rows)
    print(f"一字未删 {unchanged}/{len(rows)}，删掉 {deleted}/{total} 个字 ({deleted / total:.1%})")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"写入 {arguments.output}")


if __name__ == "__main__":
    main()
