"""Keep a fraction of the training pairs and all of the held-out ones.

For the data-volume curve. The held-out side must not move between points or
the curve measures two things at once, so only rows marked ``train`` are
thinned.

Sampled rather than truncated: ``train_polish.py --limit`` takes the first N
rows, and the pairs file is written in pool order, which is source order --
the first N would be all of one source and none of another.

Seeded by the row id rather than by position, so a point on the curve at 50%
contains every row the 25% point contained. Nesting matters here: without it a
non-monotonic curve could be sampling noise between disjoint subsets rather
than an effect of size.

    python scripts/subsample_polish_pairs.py --pairs artifacts/polish_train/pairs_v3.jsonl \
        --fraction 0.5 --output artifacts/polish_train/pairs_v3_50.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def keeps(identifier: str, fraction: float, seed: int) -> bool:
    """Whether this row survives at this fraction.

    A hash of the id put on [0, 1): every fraction keeps a prefix of the same
    ordering, so the subsets nest.
    """
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--fraction", required=True, type=float)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction 要在 (0, 1] 之间")

    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [
        row
        for row in rows
        if row.get("split") != "train" or keeps(row["id"], args.fraction, args.seed)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in kept) + "\n",
        encoding="utf-8",
    )
    trained = sum(1 for row in kept if row.get("split") == "train")
    held = sum(1 for row in kept if row.get("split") == "val")
    before = sum(1 for row in rows if row.get("split") == "train")
    print(f"训练 {trained} 对（原 {before}）、留出 {held} 对 -> {args.output}")


if __name__ == "__main__":
    main()
