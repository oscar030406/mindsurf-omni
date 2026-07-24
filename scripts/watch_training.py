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
# run_a2a.sh writes one of these before each stage, into a single log. Both
# stages then number their epochs from one, so epoch is not a key on its own.
RUN = re.compile(r"^=====\s+(?P<label>.+?)\s+=====\s*$")


@dataclass(frozen=True, slots=True)
class Point:
    run: str
    epoch: int
    step: int
    steps: int
    loss: float
    text: float
    audio: float
    lr: float


def parse(path: Path) -> list[Point]:
    points = []
    run = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        marker = RUN.match(line.strip())
        if marker:
            run = marker["label"].split()[0]
            continue
        match = LINE.search(line)
        if match:
            points.append(
                Point(
                    run=run,
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


def runs(points: list[Point]) -> list[str]:
    """Run labels in the order they appear, for logs holding more than one."""
    seen: list[str] = []
    for point in points:
        if point.run not in seen:
            seen.append(point.run)
    return seen


def window(points: list[Point], epoch: int, low: int, high: int) -> list[Point]:
    return [p for p in points if p.epoch == epoch and low <= p.step <= high]


def compare_epochs(
    points: list[Point],
    metric: str,
    low: int,
    high: int,
    tolerance: float = 3.0,
    minimum_coverage: float = 0.8,
) -> list[dict[str, object]]:
    """Aligned comparison across epochs of one run, with a verdict per step.

    An epoch is compared only once it has largely crossed the window. A
    partially filled one is not merely a smaller sample -- it is the *first*
    part of the window, which during annealing is systematically different
    from the rest, so its mean is biased rather than noisy.

    Coverage is measured against the fullest epoch rather than an absolute
    count, because the number of logged points per window depends on the log
    interval, which is a run-time choice.

    Callers must pass points from a single run. A2A writes two stages into one
    log and both number their epochs from one, so keying on the epoch alone
    merged the projector stage into the full stage's epoch 1: 122 points where
    every other epoch had 61, which pushed the coverage bar above them and
    dropped the two epochs that could actually be compared. Had the counts
    matched it would have been worse -- a comparison between 0.99M trainable
    parameters at lr 5e-4 and 152M at 2e-5, reported as an epoch effect.
    """
    results: list[dict[str, object]] = []
    previous: tuple[float, float] | None = None

    windows = {
        epoch: window(points, epoch, low, high) for epoch in sorted({p.epoch for p in points})
    }
    fullest = max((len(selected) for selected in windows.values()), default=0)
    if fullest == 0:
        return results

    for epoch, selected in windows.items():
        if len(selected) < max(10, int(fullest * minimum_coverage)):
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
    parser.add_argument(
        "--run",
        help="which stage to compare, when one log holds several (a2a_proj, "
        "a2a_full); defaults to the one the last line belongs to",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    points = parse(args.log)
    if not points:
        raise SystemExit(f"no training lines found in {args.log}")

    low, high = (int(part) for part in args.window.split(":"))
    latest = points[-1]
    # The stage still running is the one being asked about; the earlier ones
    # trained different parameters at a different rate and are not comparable
    # to it, so they are named rather than mixed in.
    selected = args.run or latest.run
    staged = [point for point in points if point.run == selected]
    rows = compare_epochs(staged, args.metric, low, high)

    if args.json:
        print(
            json.dumps(
                {"run": selected, "runs": runs(points), "latest": vars(latest), "comparison": rows},
                indent=2,
            )
        )
        return

    print(
        f"最新: epoch {latest.epoch}  step {latest.step}/{latest.steps}  "
        f"loss {latest.loss:.4f}  text {latest.text:.4f}  "
        f"audio {latest.audio:.4f}  lr {latest.lr:.6f}"
    )
    if len(runs(points)) > 1:
        print(
            f"日志含多个阶段 {runs(points)}；只比 {selected!r}——"
            "其余阶段可训参数与学习率都不同，跨阶段比出来的差异不是 epoch 效应"
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
