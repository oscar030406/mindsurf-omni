"""How close to the rail the archived clips actually sit, before any threshold.

The group OKR asks for clipping detection on the output side (section 2.4).
The detector is one function; the part that needs care is the line it fires on,
and that line is not derivable from first principles. Speech touches full scale
on a plosive, so "any sample at the rail" would fire on healthy audio; a
threshold picked to look reasonable is a threshold nobody can defend when it
starts rejecting good clips.

So this measures first and decides nothing. It reports the distribution of two
quantities over whatever clips it is pointed at, and a later round picks the
line off that distribution -- the same order the CER floor and the latency
budget were settled in.

Peak amplitude is reported beside them on purpose. If a set of clips was
normalised before it was archived, every clipping number is zero for a reason
that has nothing to do with the audio being clean, and the peak column is what
makes that visible instead of silently reassuring.

    python scripts/measure_clipping.py --clips artifacts/codebook_baseline \\
        artifacts/acceptance_ours --output artifacts/clipping-<date>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.service.audio import CLIPPED_AMPLITUDE, clipping_ratio  # noqa: E402

PERCENTILES = (50, 90, 95, 99, 100)


def read_pcm(path: Path) -> tuple[bytes, int]:
    """Raw int16 frames, through the standard library.

    soundfile would hand back floats, and the whole point here is where the
    samples sit relative to the integer rail. wave also keeps this runnable in
    the plain test environment, which has neither torch nor librosa.
    """
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path} is {handle.getsampwidth() * 8}-bit; this reads 16-bit PCM")
        return handle.readframes(handle.getnframes()), handle.getframerate()


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"p{p}": ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))] for p in PERCENTILES
    }


def normalised_already(arm: dict[str, Any], target_peak: float = 0.95) -> bool:
    """Whether these clips look like they have already been peak-normalised.

    A zero clipping ratio means one of two very different things, and the
    difference decides whether the zero is reassuring. ``peak_normalise``
    lands every loud clip on the same peak, so the giveaway is not a low peak
    but an improbably *uniform* one -- checking merely that nothing reached the
    rail would flag healthy audio and miss nothing.
    """
    expected = target_peak * 32768
    peaks = arm["peak"]
    return bool(peaks) and all(abs(peaks[f"p{p}"] - expected) <= expected * 0.02 for p in (50, 100))


def measure(directory: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    runs_ms: list[float] = []
    peaks: list[float] = []
    for path in sorted(directory.glob("*.wav")):
        pcm, rate = read_pcm(path)
        ratio, longest = clipping_ratio(pcm)
        peak = float(max((abs(sample) for sample in memoryview(pcm).cast("h")), default=0))
        run_ms = longest * 1000 / rate if rate else 0.0
        ratios.append(ratio)
        runs_ms.append(run_ms)
        peaks.append(peak)
        rows.append(
            {
                "id": path.stem,
                "clipped_ratio": ratio,
                "longest_run": longest,
                "longest_run_ms": run_ms,
                "peak": peak,
            }
        )
    if not rows:
        raise SystemExit(f"no wav files under {directory}")

    return {
        "clips": len(rows),
        "clipped_ratio": percentiles(ratios),
        "longest_run_ms": percentiles(runs_ms),
        "peak": percentiles(peaks),
        "peak_mean": statistics.mean(peaks),
        "clips_with_any_pinned_sample": sum(1 for value in ratios if value > 0),
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "threshold": CLIPPED_AMPLITUDE,
        "note": "distribution only; no pass/fail line is set by this run",
        "arms": {},
    }
    for directory in args.clips:
        arm = measure(directory)
        report["arms"][directory.name] = arm
        pinned, count = arm["clips_with_any_pinned_sample"], arm["clips"]
        print(f"\n{directory.name}  n={count}  贴顶的片段 {pinned}/{count}")
        peaks = arm["peak"]
        print(f"  峰值      中位 {peaks['p50']:.0f}  最大 {peaks['p100']:.0f}（满幅 32767）")
        print(
            f"  贴顶占比  中位 {arm['clipped_ratio']['p50']:.5f}  "
            f"p95 {arm['clipped_ratio']['p95']:.5f}  最大 {arm['clipped_ratio']['p100']:.5f}"
        )
        runs = arm["longest_run_ms"]
        print(
            f"  最长连续  中位 {runs['p50']:.2f} ms  p95 {runs['p95']:.2f} ms  "
            f"最大 {runs['p100']:.2f} ms"
        )
        if normalised_already(arm):
            print(
                "  **这批音频看起来已经过 peak_normalise**（峰值几乎条条相同且贴着 0.95 满幅）"
                "——归一化会把贴顶抹掉，上面的零读不出「干净」"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n留档 {args.output}")


if __name__ == "__main__":
    main()
