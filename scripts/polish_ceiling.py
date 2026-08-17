"""What a perfect polisher would score, so an arm can be read against the ceiling.

Every arm so far has been read against the acceptance lines, and the lines have
been treated as reachable. They are not always. The polish stage may only
delete (DECISIONS §16), so the best output it can produce is the transcript
with exactly the injected filler taken out -- and that output still differs
from the corpus sentence wherever the recogniser misheard, and wherever
SenseVoice wrote 6 for 六. Neither is a polishing error and neither is fixable
by a better polisher.

Measured, the ceiling moves with the recogniser rather than with the model:

    集合                    CER      口语词清除   内容保留   编造
    旧集 156              0.0138    1.0000    0.9872   0.0128   四条全过
    大集 986              0.0218    0.9966    0.9807   0.0210   CER 和编造过不了
    线                    ≤ 0.02    ≥ 0.90    ≥ 0.98   ≤ 0.02

So the lines are reachable on the 156-sentence set the criteria were written
against, and not on input where SenseVoice does worse. An arm scoring 0.0496
CER on the big set is 0.028 from the ceiling, not 0.030 from the line, and the
difference between those two readings is the whole question of whether to keep
training.

This writes the perfect output in the same per-row shape the arms use, so it
drops into ``polish_frontier.py`` as one more ``--arm``.

    python scripts/polish_ceiling.py --pairs artifacts/polish_train/pairs_holdout.jsonl \
        --pool artifacts/polish_train/pool_holdout.jsonl \
        --output artifacts/polish_train/val_ceiling.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.measure_polish import (  # noqa: E402
    content_kept,
    filler_removed,
    invented,
    polished_cer,
)
from scripts.retarget_polish_pairs import strip_injected  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--mode",
        choices=("perfect", "nothing"),
        default="perfect",
        help="'nothing' returns the transcript untouched. It belongs in the same "
        "table because content retention is *maximised* by doing nothing "
        "(0.9809 against the perfect deletion's 0.9807), so a row that clears "
        "the retention line has not necessarily done any work",
    )
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
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]

    written = []
    for row in rows:
        clean = pool.get(row["id"])
        if args.mode == "nothing":
            text = row["source"]
        else:
            # Falling back to the transcript rather than skipping: a sentence
            # the ceiling cannot be computed for still has to appear, or the
            # ceiling is measured on a different set than the arms are.
            text = (strip_injected(clean, row["spoken"], row["source"]) if clean else "") or row[
                "source"
            ]
        arrived, removed = filler_removed(row, text)
        written.append(
            {
                **row,
                "polished": text,
                "cer_before": polished_cer(row["target"], row["source"]),
                "cer_after": polished_cer(row["target"], text),
                "content_kept": content_kept(row["target"], text),
                "invented": invented(row["target"], text),
                "filler_arrived": arrived,
                "filler_removed": removed,
                "elapsed_ms": 0.0,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in written) + "\n",
        encoding="utf-8",
    )
    print(f"{len(written)} 句的完美输出写到 {args.output}")


if __name__ == "__main__":
    main()
