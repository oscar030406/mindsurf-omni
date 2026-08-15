"""The frontier table, with the interval each number is actually known to.

The nine-configuration table this replaces was read on 156 sentences and ranked
its arms by differences of 0.003 to 0.006, while content retention on that set
carries a 95% half-width of ±0.009. Ranking inside the interval is not ranking.
So every cell here comes with one, drawn by resampling sentences.

Sorted by filler clearance because that is the axis the arms are usually swept
along, and the trade against retention is what the table is for: read a row's
clearance, then read what it cost.

    python scripts/polish_frontier.py --arm 生成=artifacts/polish_train/val_a.jsonl \
        --arm 并集=artifacts/polish_train/val_b.jsonl --report artifacts/polish-frontier.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

# The five acceptance lines, restated rather than imported so this file can be
# read on its own. They are pinned in measure_polish.py and do not move.
LINES = {"CER": 0.02, "口语词清除": 0.90, "内容保留": 0.98, "编造": 0.02}
LOWER_IS_BETTER = {"CER", "编造"}


def numbers(rows: list[dict[str, Any]]) -> dict[str, float]:
    arrived = sum(row["filler_arrived"] for row in rows)
    return {
        "CER": statistics.fmean(row["cer_after"] for row in rows),
        # A ratio of counts, not a mean of per-sentence rates: a sentence
        # carrying three fillers is three chances, not one.
        "口语词清除": (sum(row["filler_removed"] for row in rows) / arrived) if arrived else 0.0,
        "内容保留": statistics.fmean(row["content_kept"] for row in rows),
        "编造": statistics.fmean(row["invented"] for row in rows),
    }


def interval(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    spread: dict[str, list[float]] = {name: [] for name in LINES}
    for _ in range(draws):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        for name, value in numbers(sample).items():
            spread[name].append(value)
    out = {}
    for name, values in spread.items():
        values.sort()
        out[name] = (values[int(0.025 * len(values))], values[int(0.975 * len(values))])
    return out


def verdict(name: str, point: float, low: float, high: float) -> str:
    line = LINES[name]
    if name in LOWER_IS_BETTER:
        if high <= line:
            return "过"
        return "跨线" if point <= line else "不过"
    if low >= line:
        return "过"
    return "跨线" if point >= line else "不过"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, action="append", help="名字=路径")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    table = []
    for entry in args.arm:
        if "=" not in entry:
            raise SystemExit(f"--arm 要写成 名字=路径，收到的是 {entry!r}")
        # Split on the last '=', not the first: the arm names carry one
        # themselves ("标注 t=0.9") and a path here never does.
        name, _, path = entry.rpartition("=")
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        point = numbers(rows)
        bounds = interval(rows, args.draws, args.seed)
        table.append(
            {
                "臂": name,
                "n": len(rows),
                "空输出": sum(1 for row in rows if not row["polished"].strip()),
                **{
                    key: {
                        "值": round(point[key], 4),
                        "区间": [round(bounds[key][0], 4), round(bounds[key][1], 4)],
                        "判定": verdict(key, point[key], *bounds[key]),
                    }
                    for key in LINES
                },
            }
        )

    table.sort(key=lambda row: -row["口语词清除"]["值"])
    header = f"{'臂':<22}{'CER':>20}{'口语词清除':>20}{'内容保留':>20}{'编造':>20}{'空':>4}"
    print(header)
    marks = {"过": "✅", "跨线": "~"}
    for row in table:
        cells = "".join(
            f"{row[key]['值']:.4f} {marks.get(row[key]['判定'], ' ')}".rjust(20) for key in LINES
        )
        print(f"{row['臂']:<22}{cells}{row['空输出']:>4}")
    print(f"{'线':<22}{'≤ 0.02':>20}{'≥ 0.90':>20}{'≥ 0.98':>20}{'≤ 0.02':>20}{'0':>4}")
    print("✅ = 95% 区间整体在线的正确一侧；~ = 点估计过了但区间跨线；空白 = 不过")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n表写到 {args.report}")


if __name__ == "__main__":
    main()
