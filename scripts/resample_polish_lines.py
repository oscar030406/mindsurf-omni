"""How often each polish criterion stays inside its line when the rows are resampled.

The release table reports these percentages and nothing in the repository could
recompute them: they were worked out once by hand and typed into
``headline_numbers.json``, where they then outlived the run that produced them.
This is that arithmetic, scripted, so a new reading can carry its own.

Why the percentage and not the point estimate. A criterion sitting a hair inside
its line is not the same result as one sitting well inside it, and a single
number cannot tell them apart. Draw the 986 rows with replacement a few thousand
times, recompute, and count how often the answer is still inside: a stable
result is inside nearly every time, and a borderline one lands around half.
"Point estimate passed, half the resamples did not" is the honest reading of
borderline, and it is the reading the release table is built on.

Rows come from ``measure_polish_service.py --output``. The two halves are
reported apart, because on the 609 rows that got a double-duty injection the
target asks for a content word to be deleted -- work this stage deliberately no
longer does -- and averaging those in measures obedience to the injector.

    python scripts/resample_polish_lines.py --rows artifacts/polish-rows.jsonl \\
        --draws 4000 --report artifacts/polish-resamples.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The lines the algorithm side set for itself, and which side of each one is
# inside. Not the team's acceptance thresholds; see headline_numbers.json.
LINES: dict[str, tuple[float, str]] = {
    "cer_after": (0.02, "at_most"),
    "content_kept": (0.98, "at_least"),
    "invented": (0.02, "at_most"),
    "filler_removed_rate_gated": (0.90, "at_least"),
}


def inside(name: str, value: float | None) -> bool:
    if value is None:
        return False
    line, side = LINES[name]
    return value <= line if side == "at_most" else value >= line


def measure(rows: list[dict[str, Any]], name: str) -> float | None:
    """One criterion over one sample of rows.

    The filler rate is a ratio of totals rather than a mean of per-row rates:
    a row that received one filler and a row that received nine are not equal
    evidence, and averaging their rates would say they were.
    """
    if not rows:
        return None
    if name == "filler_removed_rate_gated":
        seen = sum(row["gated_arrived"] for row in rows)
        return sum(row["gated_removed"] for row in rows) / seen if seen else None
    return statistics.fmean(row[name] for row in rows)


def resample(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    point = {name: measure(rows, name) for name in LINES}
    counts = dict.fromkeys(LINES, 0)
    counted = dict.fromkeys(LINES, 0)
    size = len(rows)
    for _ in range(draws):
        drawn = [rows[rng.randrange(size)] for _ in range(size)]
        for name in LINES:
            value = measure(drawn, name)
            if value is None:
                continue
            counted[name] += 1
            counts[name] += inside(name, value)
    return {
        name: {
            "value": point[name],
            "line": LINES[name][0],
            "side": LINES[name][1],
            "point_inside": inside(name, point[name]),
            "resamples_inside": counts[name] / counted[name] if counted[name] else None,
            "draws": counted[name],
        }
        for name in LINES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="from --output")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trustworthy = [row for row in rows if not row.get("got_double_duty")]

    halves: dict[str, Callable[[], list[dict[str, Any]]]] = {
        "all_986": lambda: rows,
        "rows_with_a_trustworthy_target": lambda: trustworthy,
    }
    report = {
        "rows": str(args.rows),
        "draws": args.draws,
        "seed": args.seed,
        "_": [
            "两半分开报。986 行里有 609 行的 target 要求删掉一个内容词——",
            "这一级现在故意不删，所以那一半量的是对注入器的服从，不是正确性。",
        ],
        "halves": {},
    }
    for name, take in halves.items():
        here = take()
        got = resample(here, args.draws, args.seed)
        report["halves"][name] = {"n": len(here), **got}
        print(f"\n{name}（{len(here)} 行）")
        for criterion, value in got.items():
            mark = "内" if value["point_inside"] else "外"
            share = value["resamples_inside"]
            print(
                f"  {criterion:32s} {value['value']:.4f}  线 {value['line']}  点估计在{mark}"
                f"  重抽在内 {share:.0%}"
                if share is not None
                else ""
            )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n写到 {args.report}")


if __name__ == "__main__":
    main()
