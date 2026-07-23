"""Read a training log and say what the curve does -- and does not -- show.

Written after reading the curve wrong by eye. Comparing step 7900 of one epoch
against step 27500 of another looked like the audio loss had started climbing;
aligned to the same step range it was flat, and the apparent rise was inside
the noise. The row-group shuffle means each position sees different data, so
any cross-position comparison is confounded before it starts.

So this aligns windows by step, computes a bootstrap noise floor for each, and
refuses to name a direction for a difference that sits inside it. The same rule
the evaluation harness applies to the model, applied to the training curve.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import bootstrap_noise_floor  # noqa: E402

LINE = re.compile(
    r"Epoch:\[(?P<epoch>\d+)/(?P<epochs>\d+)\]\((?P<step>\d+)/(?P<steps>\d+)\).*?"
    r"loss: (?P<loss>[\d.]+), text: (?P<text>[\d.]+), audio: (?P<audio>[\d.]+), "
    r"lr: (?P<lr>[\d.eE+-]+)"
)


@dataclass(frozen=True, slots=True)
class Point:
    epoch: int
    step: int
    steps: int
    loss: float
    text: float
    audio: float
    lr: float


def parse(path: Path) -> list[Point]:
    points = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LINE.search(line)
        if match:
            points.append(
                Point(
                    epoch=int(match["epoch"]),
                    step=int(match["step"]),
                    steps=int(match["steps"]),
                    loss=float(match["loss"]),
                    text=float(match["text"]),
                    audio=float(match["audio"]),
                    lr=float(match["lr"]),
                )
            )
    return points


def window(points: list[Point], epoch: int, low: int, high: int) -> list[Point]:
    return [p for p in points if p.epoch == epoch and low <= p.step <= high]


def compare_epochs(
    points: list[Point], metric: str, low: int, high: int, tolerance: float = 3.0
) -> list[dict[str, object]]:
    """Aligned comparison across epochs, with a verdict per step.

    Only epochs that actually reached this window are compared. An epoch still
    short of it would otherwise contribute a truncated sample and a noise floor
    computed from too few points.
    """
    results: list[dict[str, object]] = []
    previous: tuple[float, float] | None = None

    for epoch in sorted({p.epoch for p in points}):
        selected = window(points, epoch, low, high)
        if len(selected) < 10:
            continue
        values = [getattr(p, metric) for p in selected]
        mean = statistics.fmean(values)
        noise = bootstrap_noise_floor(values)

        row: dict[str, object] = {
            "epoch": epoch,
            "n": len(values),
            "mean": mean,
            "noise_floor": noise,
        }
        if previous is not None:
            previous_mean, previous_noise = previous
            difference = mean - previous_mean
            threshold = tolerance * (noise**2 + previous_noise**2) ** 0.5
            row["difference"] = difference
            row["threshold"] = threshold
            row["verdict"] = (
                "indistinguishable"
                if abs(difference) <= threshold
                else ("regressed" if difference > 0 else "improved")
            )
        results.append(row)
        previous = (mean, noise)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--metric", default="audio", choices=["loss", "text", "audio"])
    parser.add_argument(
        "--window",
        default="5000:8000",
        help="step range to compare across epochs; positions must match, "
        "because shuffling makes different positions see different data",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    points = parse(args.log)
    if not points:
        raise SystemExit(f"no training lines found in {args.log}")

    low, high = (int(part) for part in args.window.split(":"))
    latest = points[-1]
    rows = compare_epochs(points, args.metric, low, high)

    if args.json:
        print(json.dumps({"latest": vars(latest), "comparison": rows}, indent=2))
        return

    print(
        f"最新: epoch {latest.epoch}  step {latest.step}/{latest.steps}  "
        f"loss {latest.loss:.4f}  text {latest.text:.4f}  "
        f"audio {latest.audio:.4f}  lr {latest.lr:.6f}"
    )
    print(f"\n{args.metric} 在 step {low}-{high} 的逐 epoch 对比（对齐位置，含噪声底）:")
    for row in rows:
        line = (
            f"  epoch {row['epoch']}  n={row['n']:3d}  {row['mean']:.4f} ± {row['noise_floor']:.4f}"
        )
        if "verdict" in row:
            line += (
                f"   对上一 epoch {row['difference']:+.4f}, "
                f"阈值 ±{row['threshold']:.4f} -> {row['verdict']}"
            )
        print(line)

    if len(rows) < 2:
        print("\n只有一个完整窗口，还不能比较。")


if __name__ == "__main__":
    main()
