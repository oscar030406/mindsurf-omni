"""Did the prosody difference in the references survive the model? Paired per text.

The result being checked: give the model a reference with the emotion in it and
the output carries pitch, spread and rate. Every instrument here has to state
what it can resolve before it may judge, so the arithmetic is the same
``compare_paired`` every other paired comparison uses, and the protocol carries
one addition a bare before-and-after table cannot have: **a control arm**.

Generation is sampled, so two runs from the *same* reference already differ, and
without knowing by how much, a +20 Hz separation between two different
references means nothing. Run the same reference twice
under different seeds, pass it as an arm, and it must come back
``indistinguishable``. If it does not, nothing else in the run is readable.

Verdict vocabulary: these are the three words the shared comparison emits, and
on an axis with no better or worse direction they read as separation.

    improved            the delta is positive and outside the noise floor
    regressed           the delta is negative and outside the noise floor
    indistinguishable   the delta is inside the noise floor -- no direction

The sign is printed either way, so the word never has to carry the meaning
alone.

    python scripts/measure_prosody.py \\
        --arm calm=<dir>/manifest.json --arm lively=<dir>/manifest.json \\
        --arm control=<dir>/manifest.json --baseline calm \\
        --output artifacts/emotion-harvest-gate.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import compare_paired  # noqa: E402

# Pitch is read as a median over voiced frames only. An unvoiced stretch scored
# as 0 Hz would drag the median toward silence, which is a property of the
# sentence rather than of the speaker's prosody.
F0_FLOOR_HZ = 60.0
F0_CEILING_HZ = 500.0
VOICED_LOW_HZ = 65.0
VOICED_HIGH_HZ = 480.0
MINIMUM_VOICED_FRAMES = 10

# Higher is the direction the emotional arm is expected to move on the first
# two; a livelier delivery is also faster, so its duration goes the other way.
AXES: tuple[tuple[str, bool], ...] = (
    ("f0_median_hz", False),
    ("f0_iqr_hz", False),
    ("seconds", True),
)

# Pitch has a declared line: the labelling pilot judged its per-label F0 against
# 15 Hz and marked every row reported-only for resolving 22.8. Spread and
# duration have never had one, and inventing a number here to make a guardrail
# come out somewhere would be worse than leaving the older behaviour, so they
# stay unset and keep deciding on the noise floor alone.
EFFECTS: dict[str, float] = {"f0_median_hz": 15.0}


def prosody(path: str | Path, rate: int = 24_000) -> dict[str, float] | None:
    """Median pitch, its interquartile spread, and duration -- or None if unvoiced.

    None rather than a number: a clip the pitch tracker cannot follow is not a
    clip with a low pitch, and a run that quietly scores it as one produces a
    separation that came from the tracker.
    """
    import librosa
    import numpy as np

    wave, _ = librosa.load(str(path), sr=rate)
    f0 = librosa.yin(wave, fmin=F0_FLOOR_HZ, fmax=F0_CEILING_HZ, sr=rate)
    voiced = f0[(f0 > VOICED_LOW_HZ) & (f0 < VOICED_HIGH_HZ)]
    if len(voiced) < MINIMUM_VOICED_FRAMES:
        return None
    return {
        "f0_median_hz": float(np.median(voiced)),
        "f0_iqr_hz": float(np.percentile(voiced, 75) - np.percentile(voiced, 25)),
        "seconds": len(wave) / rate,
    }


def read_manifest(path: Path) -> dict[str, str]:
    """Sample id to audio path, skipping the rows a failed generation leaves behind."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(sample["id"]): str(sample["audio_path"])
        for sample in payload.get("samples", [])
        if sample.get("id") is not None and sample.get("audio_path")
    }


def pair_arms(measured: dict[str, dict[str, dict[str, float]]]) -> tuple[list[str], list[str]]:
    """Ids every arm measured, and the ids that fell out -- the second is not optional.

    A comparison over whichever samples happened to survive is a comparison
    over a set nobody chose. Both lists are returned so the report can say how
    many were lost rather than only how many were kept.
    """
    if not measured:
        return [], []
    everywhere = set.intersection(*(set(arm) for arm in measured.values()))
    anywhere = set.union(*(set(arm) for arm in measured.values()))
    return sorted(everywhere), sorted(anywhere - everywhere)


def verdicts(
    measured: dict[str, dict[str, dict[str, float]]], baseline: str, paired: list[str]
) -> dict[str, list[str]]:
    """One line per axis per arm, against the baseline arm, on the shared ids."""
    lines: dict[str, list[str]] = {}
    for label, samples in measured.items():
        if label == baseline:
            continue
        lines[label] = [
            compare_paired(
                axis,
                [samples[i][axis] - measured[baseline][i][axis] for i in paired],
                lower_is_better=lower_is_better,
                effect_of_interest=EFFECTS.get(axis),
            )
            for axis, lower_is_better in AXES
        ]
    return lines


def summarise(samples: dict[str, dict[str, float]], paired: list[str]) -> dict[str, float]:
    return {axis: statistics.fmean(samples[i][axis] for i in paired) for axis, _ in AXES if paired}


def measure(arms: dict[str, Path]) -> dict[str, dict[str, dict[str, float]]]:
    measured: dict[str, dict[str, dict[str, float]]] = {}
    for label, manifest in arms.items():
        rows = read_manifest(manifest)
        scored: dict[str, dict[str, float]] = {}
        for identifier, audio in sorted(rows.items()):
            reading = prosody(audio)
            if reading is not None:
                scored[identifier] = reading
        print(f"  {label:<12} {len(scored)}/{len(rows)} 条可读音高")
        measured[label] = scored
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=MANIFEST",
        help="one generation run. Give at least two, and give a control arm "
        "(the baseline reference under a second seed) or the separation has "
        "nothing to be larger than",
    )
    parser.add_argument("--baseline", help="the arm the others are compared against")
    parser.add_argument("--output", type=Path, help="where to write the report")
    args = parser.parse_args()

    # Split on the last separator: a manifest path may contain one, a label
    # written by hand is more likely to.
    arms: dict[str, Path] = {}
    for entry in args.arm:
        label, _, manifest = entry.rpartition("=")
        if not label or not manifest:
            raise SystemExit(f"--arm wants LABEL=MANIFEST, got {entry!r}")
        arms[label] = Path(manifest)

    if len(arms) < 2:
        raise SystemExit("give at least two arms; one arm has nothing to compare against")

    baseline = args.baseline or next(iter(arms))
    if baseline not in arms:
        raise SystemExit(f"--baseline {baseline} is not one of {sorted(arms)}")

    measured = measure(arms)
    paired, dropped = pair_arms(measured)
    if len(paired) < 2:
        raise SystemExit(f"only {len(paired)} ids are present in every arm; nothing to pair")

    lost = "：" + ", ".join(dropped) if dropped else ""
    print(f"\n配对 {len(paired)} 条，掉出 {len(dropped)} 条{lost}")
    print(f"基线臂 {baseline}\n")

    lines = verdicts(measured, baseline, paired)
    for label, results in lines.items():
        print(f"{label} − {baseline}:")
        for line in results:
            print(f"  {line}")

    report: dict[str, Any] = {
        "baseline": baseline,
        "arms": {label: str(path) for label, path in arms.items()},
        "paired": len(paired),
        "dropped": dropped,
        "verdicts": lines,
        "summary": {label: summarise(samples, paired) for label, samples in measured.items()},
        # The per-sample rows, because a paired verdict cannot be re-derived
        # from a mean and this project has already paid once for keeping only
        # the summary.
        "rows": [
            {"id": identifier, **{label: measured[label][identifier] for label in measured}}
            for identifier in paired
        ],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
