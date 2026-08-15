"""Two arms on the same held-out sentences: is the difference bigger than the set.

The arms are measured on the same 156 sentences, so the comparison is paired
and the resampling has to be paired too -- resampling each arm on its own would
report a spread that the shared sentences do not have. What is resampled is
sentences, not rows of a metric: a sentence that both arms find easy carries no
information about which is better, and only a sentence-level bootstrap gives it
the weight it deserves.

Filler clearance is a ratio of counts rather than a mean of per-sentence rates,
because a sentence with three fillers should not weigh the same as one with a
single 嗯. Both arms' numerators and denominators are resampled together.

    python scripts/compare_polish_arms.py --before artifacts/polish_train/val_a.jsonl \
        --after artifacts/polish_train/val_b.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def means(rows: list[tuple[dict, dict]], index: int) -> dict[str, float]:
    side = [pair[index] for pair in rows]
    arrived = sum(row["filler_arrived"] for row in side)
    return {
        "CER": statistics.fmean(row["cer_after"] for row in side),
        "口语词清除": (sum(row["filler_removed"] for row in side) / arrived) if arrived else 0.0,
        "内容保留": statistics.fmean(row["content_kept"] for row in side),
        "编造": statistics.fmean(row["invented"] for row in side),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    left, right = load(args.before), load(args.after)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise SystemExit("两份逐条输出没有共同的句子，比不了")
    rows = [(left[key], right[key]) for key in shared]
    print(f"配对 {len(rows)} 句（{args.before.name} -> {args.after.name}）")

    before, after = means(rows, 0), means(rows, 1)
    rng = random.Random(args.seed)
    spread: dict[str, list[float]] = {name: [] for name in before}
    for _ in range(args.draws):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        drawn_before, drawn_after = means(sample, 0), means(sample, 1)
        for name in spread:
            spread[name].append(drawn_after[name] - drawn_before[name])

    print(f"{'':<12}{'前':>10}{'后':>10}{'差':>10}{'95% 区间':>22}{'':>6}")
    for name in before:
        draws = sorted(spread[name])
        low = draws[int(0.025 * len(draws))]
        high = draws[int(0.975 * len(draws))]
        crosses = low <= 0 <= high
        print(
            f"{name:<12}{before[name]:>10.4f}{after[name]:>10.4f}"
            f"{after[name] - before[name]:>+10.4f}"
            f"{f'[{low:+.4f}, {high:+.4f}]':>22}"
            f"{'  跨 0' if crosses else '  不跨 0':>6}"
        )
    print("\n跨 0 的那几行，这批句子分不开两条臂。")


if __name__ == "__main__":
    main()
