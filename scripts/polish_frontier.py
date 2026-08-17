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
import difflib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.measure_polish import polished_cer  # noqa: E402

# The five acceptance lines, restated rather than imported so this file can be
# read on its own. They are pinned in measure_polish.py and do not move.
# The CER *function* is imported rather than restated for the opposite reason:
# a second copy of the folding rule is how the arms and the ceiling ended up on
# two different rulers in the first place.
LINES = {"CER": 0.02, "口语词清除": 0.90, "内容保留": 0.98, "编造": 0.02}
LOWER_IS_BETTER = {"CER", "编造"}

PUNCTUATION = set("，。！？；：、")


def surviving_punctuation(source: str, output: str) -> set[int]:
    """Indices of the transcript's punctuation that reached the output."""
    kept: set[int] = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
        None, source, output, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            kept.update(range(i1, i2))
    return {index for index in kept if source[index] in PUNCTUATION}


def punctuation_kept(
    rows: list[dict[str, Any]], ceiling: dict[str, dict[str, Any]]
) -> float | None:
    """How much of the punctuation a perfect polisher keeps this arm also kept.

    None of the four acceptance numbers can see this: ``normalise_for_cer``
    strips punctuation from both sides, so an arm that eats every comma scores
    exactly the same as one that keeps them all. For a dictation product that
    is the difference between usable and not, which is why it is measured here
    rather than left to the criteria.

    The reference is the ceiling arm rather than the corpus sentence, because
    the punctuation a polisher should keep is the recogniser's -- minus
    whatever came in attached to an injected filler.
    """
    should = kept = 0
    for row in rows:
        reference = ceiling.get(row["id"])
        if reference is None:
            continue
        wanted = surviving_punctuation(reference["source"], reference["polished"])
        mine = surviving_punctuation(row["source"], row["polished"])
        should += len(wanted)
        kept += len(wanted & mine)
    return kept / should if should else None


def load_arm(path: Path) -> list[dict[str, Any]]:
    """Read an arm, and re-derive its CER rather than trusting the stored one.

    The stored ``cer_after`` says which build wrote the file. Every script that
    writes these called ``character_error_rate`` without ``fold_numbers`` until
    it was fixed, so the files on disk now carry a mix: what was rescored after
    the fix is folded and the rest is not. A table that puts both in one column
    ranks arms by when they happened to be run.

    Once per row here rather than inside ``numbers``, which the bootstrap calls
    two thousand times. ``content_kept``, ``invented`` and ``filler_removed``
    always folded, so this is the only column that needs re-deriving.
    """
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    for row in rows:
        row["cer_after"] = polished_cer(row["target"], row["polished"])
    return rows


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


def reached(name: str, point: float, floor: float, ceiling: float) -> float | None:
    """Share of the reachable distance this arm covered, floor to ceiling.

    None unless the ceiling is meaningfully better than the floor. Content
    retention is exactly the case this guard exists for: doing nothing scores
    0.9809 and deleting perfectly 0.9807, so the span is *negative* -- there is
    no improvement available over doing nothing, and dividing by that span
    produces percentages in the thousands. A tenth of the smallest difference
    the criteria distinguish (0.001) is the floor for calling it a distance.
    """
    span = (floor - ceiling) if name in LOWER_IS_BETTER else (ceiling - floor)
    if span < 1e-3:
        return None
    covered = (floor - point) if name in LOWER_IS_BETTER else (point - floor)
    return covered / span


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
    parser.add_argument(
        "--ceiling",
        type=Path,
        help="polish_ceiling.py --mode perfect 的产物. With --floor it adds a "
        "达成率 column: how much of the reachable distance an arm actually "
        "covered. That is the number that answers whether to keep training, "
        "and the raw gap to the line is not -- the line may sit past the ceiling",
    )
    parser.add_argument("--floor", type=Path, help="polish_ceiling.py --mode nothing 的产物")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    load = load_arm
    ends = None
    if args.ceiling and args.floor:
        ends = (numbers(load(args.floor)), numbers(load(args.ceiling)))
    reference = {row["id"]: row for row in load(args.ceiling)} if args.ceiling else {}

    table = []
    for entry in args.arm:
        if "=" not in entry:
            raise SystemExit(f"--arm 要写成 名字=路径，收到的是 {entry!r}")
        # Split on the last '=', not the first: the arm names carry one
        # themselves ("标注 t=0.9") and a path here never does.
        name, _, path = entry.rpartition("=")
        rows = load(Path(path))
        point = numbers(rows)
        bounds = interval(rows, args.draws, args.seed)
        table.append(
            {
                "臂": name,
                "n": len(rows),
                "空输出": sum(1 for row in rows if not row["polished"].strip()),
                "标点保住率": punctuation_kept(rows, reference) if reference else None,
                **{
                    key: {
                        "值": round(point[key], 4),
                        "区间": [round(bounds[key][0], 4), round(bounds[key][1], 4)],
                        "判定": verdict(key, point[key], *bounds[key]),
                        **(
                            {"达成率": reached(key, point[key], ends[0][key], ends[1][key])}
                            if ends
                            else {}
                        ),
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

    if reference:
        print("\n标点保住率：判据看不见的一维（normalise_for_cer 两边都剥标点）")
        print(f"{'臂':<22}{'标点保住率':>14}")
        for row in table:
            rate = row["标点保住率"]
            if rate is not None:
                print(f"{row['臂']:<22}{rate:>14.4f}")
        print("参照是完美删除那条臂保住的标点：识别器写的、不属于任何注入口语词的那些")

    if ends:
        print("\n达成率：从「什么都不做」到「完美删除」这段距离，这条臂走了多少")
        print(f"{'臂':<22}{'CER':>12}{'口语词清除':>12}{'内容保留':>12}{'编造':>12}")
        for row in table:
            cells = "".join(
                ("——" if row[key]["达成率"] is None else f"{row[key]['达成率']:.0%}").rjust(12)
                for key in LINES
            )
            print(f"{row['臂']:<22}{cells}")
        print("「——」= 地板和天花板一样高，这条判据上没有距离可走")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n表写到 {args.report}")


if __name__ == "__main__":
    main()
