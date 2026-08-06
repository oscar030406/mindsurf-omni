"""Ask the judge the same question twice, with the two answers swapped.

Every "better" in this project is one judge's word. How often that word survives
a change that carries no information is a property of the instrument alone: the
two replies are identical text, only the seats moved, so a verdict that flips
carries no content explanation. It is noise plus a seat preference and nothing
else.

That number is worth having because a judge that coin-flips a share ``f`` of the
pairs it claims to decide pulls every win rate toward 0.5::

    observed = (1 - f) * true + f * 0.5

so a real 0.63 reads as 0.59 at f=0.3, and the noise floor is computed as if
every pair carried information when a share of them carried none. Both errors
run the same way: real effects look smaller and more certain than they are.

What this cannot say is whether the judge is right. Repeatable and correct are
different properties, and a ruler can be stable about the wrong number.

    python scripts/judge_reliability.py swap --judged <pairs> --output <swapped>
    python scripts/blind_preference.py --judge-pairs <swapped> \
        --output <judged> --pairs <full>
    python scripts/judge_reliability.py compare --first <pairs> --second <full> \
        --output <report>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SEATED = ("left", "right")
OPPOSITE = {"left": "right", "right": "left"}


def load_pairs(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    return blob["pairs"] if isinstance(blob, dict) else blob


def swap(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same pairs with the two replies in each other's seat.

    The key gets a suffix rather than being reused. Judging writes results keyed
    by pair, and two rows under one key is the failure ``load_prebuilt`` refuses
    on the way in -- a resume would hand the first round's verdict to the second
    round's question.

    ``winner`` is dropped. Carrying it would leave the previous verdict sitting
    in the file the judge's fresh answer is written into, and the first thing to
    read that file could not tell the two apart.
    """
    swapped = []
    for pair in pairs:
        row = {k: v for k, v in pair.items() if k != "winner"}
        row["key"] = f"{pair['key']}~swap"
        row["of_key"] = pair["key"]
        row["left"], row["right"] = pair["right"], pair["left"]
        if pair.get("longer_side") in OPPOSITE:
            row["longer_side"] = OPPOSITE[pair["longer_side"]]
        swapped.append(row)
    return swapped


def _seat_to_text(pair: dict[str, Any], winner: str | None) -> str | None:
    """Which of the two texts a seated verdict picked.

    Comparing seats across the two rounds would call every consistent judgement
    a flip, since the seats are what changed. Comparing the text is the whole
    point of the exercise.
    """
    if winner not in SEATED:
        return None
    return str(pair[winner])


def compare(
    first: list[dict[str, Any]], second: list[dict[str, Any]], sigma: float = 3.0
) -> dict[str, Any]:
    """Flip rate per bin, plus where the ties went."""
    by_key = {p["key"]: p for p in first}
    report: dict[str, Any] = {"sigma": sigma, "bins": {}, "unmatched": 0}
    rows: list[tuple[str, str, str | None, str | None]] = []
    for pair in second:
        origin = by_key.get(pair.get("of_key", ""))
        if origin is None:
            report["unmatched"] += 1
            continue
        rows.append(
            (
                origin.get("bin", "?"),
                origin["key"],
                _seat_to_text(origin, origin.get("winner")),
                _seat_to_text(pair, pair.get("winner")),
            )
        )

    for name in sorted({bin_name for bin_name, *_ in rows}):
        mine = [r for r in rows if r[0] == name]
        both = [r for r in mine if r[2] is not None and r[3] is not None]
        flips = sum(1 for r in both if r[2] != r[3])
        entry: dict[str, Any] = {
            "n": len(mine),
            "decided_twice": len(both),
            "flips": flips,
            "transitions": {
                "decided_decided": len(both),
                "decided_tie": sum(1 for r in mine if r[2] is not None and r[3] is None),
                "tie_decided": sum(1 for r in mine if r[2] is None and r[3] is not None),
                "tie_tie": sum(1 for r in mine if r[2] is None and r[3] is None),
            },
        }
        if both:
            rate = flips / len(both)
            entry["flip_rate"] = rate
            entry["noise"] = sigma * math.sqrt(0.25 / len(both))
            # A flip means one of the two rounds was a coin toss on a pair the
            # judge claimed to decide. Two coin tosses agree half the time, so
            # the flip rate sees only half of them: f = 2 * flip_rate. The
            # attenuation that implies is what makes this number worth having.
            entry["coin_toss_share"] = min(1.0, 2 * rate)
        report["bins"][name] = entry
    report["flip_shape"] = flip_shape(rows_by_key(first, second), sigma)
    return report


def rows_by_key(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key = {p["key"]: p for p in first}
    matched = []
    for pair in second:
        origin = by_key.get(pair.get("of_key", ""))
        if origin is not None:
            matched.append((origin, pair))
    return matched


def flip_shape(
    matched: list[tuple[dict[str, Any], dict[str, Any]]], sigma: float = 3.0
) -> dict[str, Any]:
    """Whether the flips are noise or a seat the judge likes.

    A flip rate on its own cannot tell those apart, and they mean opposite
    things. Pure noise attenuates every win rate toward 0.5, so the numbers this
    project reports would all understate their effects. A stable seat preference
    does not: the sides are assigned by a seeded coin, and ``tally`` already
    measures and reports what survives that randomisation.

    The split is visible in the flips themselves. A judge that always takes the
    left seat picks left in both rounds every time it flips; noise splits those
    evenly. So the share of flips that went left twice is the test, and 0.5 is
    the null.
    """
    left = right = 0
    for before, after in matched:
        w1, w2 = before.get("winner"), after.get("winner")
        if w1 not in SEATED or w2 not in SEATED:
            continue
        if str(before[w1]) == str(after[w2]):
            continue
        if w1 == w2 == "left":
            left += 1
        elif w1 == w2 == "right":
            right += 1
    total = left + right
    if not total:
        return {"flips": 0, "note": "没有翻转，形状无从判断"}
    share = left / total
    noise = sigma * math.sqrt(0.25 / total)
    return {
        "flips": total,
        "left_twice": left,
        "right_twice": right,
        "left_share": share,
        "noise": noise,
        # Under-powered on purpose-built samples this size; say so rather than
        # letting the point estimate read as a verdict.
        "reads_as": "座位偏好" if abs(share - 0.5) > noise else "对称，分不开噪声与座位偏好",
        "at_1_96_sigma": "座位偏好"
        if abs(share - 0.5) > 1.96 * math.sqrt(0.25 / total)
        else "对称",
    }


def attenuation(coin_toss_share: float, observed: float) -> float | None:
    """What an observed win rate would be without the coin tosses."""
    if coin_toss_share >= 1.0:
        return None
    return (observed - 0.5 * coin_toss_share) / (1 - coin_toss_share)


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    """R1 / R2 / R3, registered before the run."""
    rates = {n: e["flip_rate"] for n, e in report["bins"].items() if "flip_rate" in e}
    if not rates:
        return {"line": "R3", "why": "没有一个箱两次都判出胜负，翻转率算不出来"}
    worst = max(rates, key=lambda n: rates[n])
    if all(r <= 0.10 for r in rates.values()):
        line, why = "R1", "三个箱翻转率都 ≤ 0.10，这把尺子重复得上"
    elif rates[worst] >= 0.30:
        line, why = "R2", f"「{worst}」箱翻转率 {rates[worst]:.4f} ≥ 0.30"
    else:
        line, why = "R3", f"最高的「{worst}」箱 {rates[worst]:.4f}，落在 0.10 与 0.30 之间"

    ordered = ["near", "mid", "far"]
    present = [n for n in ordered if n in rates]
    values = [rates[n] for n in present]
    predicted = values == sorted(values, reverse=True) and len(values) > 1
    return {
        "line": line,
        "why": why,
        "flip_rates": rates,
        "prediction_near_gt_mid_gt_far": {
            "held": predicted,
            "observed_order": {n: rates[n] for n in present},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    build = sub.add_parser("swap", help="write the same pairs with the seats exchanged")
    build.add_argument("--judged", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    read = sub.add_parser("compare", help="flip rate between two judged rounds")
    read.add_argument("--first", type=Path, required=True)
    read.add_argument("--second", type=Path, required=True, help="the swapped round")
    read.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "swap":
        pairs = swap(load_pairs(args.judged))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"pairs": pairs}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{len(pairs)} 对已对调座位，写入 {args.output}")
        return 0

    report = compare(load_pairs(args.first), load_pairs(args.second))
    report["verdict"] = verdict(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, entry in report["bins"].items():
        if "flip_rate" in entry:
            print(
                f"  {name:5s} 两次都判 {entry['decided_twice']:3d}  翻转 "
                f"{entry['flip_rate']:.4f} ± {entry['noise']:.4f}"
                f"  掷硬币比例 {entry['coin_toss_share']:.4f}"
                f"  平局转移 判→平 {entry['transitions']['decided_tie']:3d}"
                f" 平→判 {entry['transitions']['tie_decided']:3d}"
            )
    if report["unmatched"]:
        print(f"  对不上第一轮的 {report['unmatched']} 对，没进统计")
    print(f"  {report['verdict']['line']}: {report['verdict']['why']}")
    print(f"写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
