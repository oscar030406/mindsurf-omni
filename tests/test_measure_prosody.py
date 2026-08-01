"""The control arm is the whole protocol, so it is the thing tested.

Generation is sampled: two runs from the same reference already differ in
pitch. A prosody separation between two *different* references is only a
result if it is larger than that, and the way this run can look like it worked
without having is for the comparison to hand back a direction on noise. So the
first test feeds two arms that differ only by sampling jitter and requires the
verdict to be "indistinguishable" -- if that ever stops holding, no number this
script prints is readable.

The rest is the failure this project has already paid for once: samples that
fall out of one arm and are silently dropped from the pairing, leaving a
comparison over a set nobody chose.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.measure_prosody import pair_arms, read_manifest, summarise, verdicts


def _arm(readings: dict[str, tuple[float, float, float]]) -> dict[str, dict[str, float]]:
    return {
        identifier: {"f0_median_hz": f0, "f0_iqr_hz": iqr, "seconds": seconds}
        for identifier, (f0, iqr, seconds) in readings.items()
    }


def _jitter(index: int) -> float:
    """Deterministic small wobble, alternating sign so the mean stays near zero."""
    return (1.7 if index % 2 else -1.9) + (index % 3) * 0.4


def test_two_runs_of_the_same_reference_do_not_separate() -> None:
    """The control arm. A direction here means the instrument is reading sampling noise."""
    base = _arm({f"zh{i:03d}": (150.0 + i, 40.0 + i, 3.0) for i in range(20)})
    rerun = _arm(
        {f"zh{i:03d}": (150.0 + i + _jitter(i), 40.0 + i + _jitter(i) / 2, 3.0) for i in range(20)}
    )

    measured = {"calm": base, "control": rerun}
    paired, _ = pair_arms(measured)
    lines = verdicts(measured, "calm", paired)["control"]

    assert all("indistinguishable" in line for line in lines), lines


def test_a_separation_larger_than_the_jitter_gets_a_direction() -> None:
    """And the sign is the one the reference pair was built to produce."""
    calm = _arm({f"zh{i:03d}": (150.0 + i, 40.0 + i, 3.4) for i in range(20)})
    lively = _arm(
        {
            # +25 Hz pitch, wider spread, and faster -- the three axes the
            # reference experiment moved together.
            f"zh{i:03d}": (175.0 + i + _jitter(i), 55.0 + i, 3.0)
            for i in range(20)
        }
    )

    measured = {"calm": calm, "lively": lively}
    paired, _ = pair_arms(measured)
    lines = verdicts(measured, "calm", paired)["lively"]

    assert "f0_median_hz: improved (+25" in lines[0], lines[0]
    assert "f0_iqr_hz: improved (+15" in lines[1], lines[1]
    # Shorter is "improved" on duration: the livelier arm speaks faster.
    assert "seconds: improved (-0.4" in lines[2], lines[2]


def test_an_id_missing_from_one_arm_is_reported_rather_than_dropped() -> None:
    """A pairing that shrinks in silence is a comparison over an unchosen set."""
    measured = {
        "calm": _arm({"zh000": (150.0, 40.0, 3.0), "zh001": (151.0, 41.0, 3.0)}),
        "lively": _arm({"zh000": (175.0, 55.0, 2.6)}),
    }

    paired, dropped = pair_arms(measured)

    assert paired == ["zh000"]
    assert dropped == ["zh001"]


def test_a_row_whose_generation_produced_no_audio_is_skipped(tmp_path: Path) -> None:
    """evaluate_talker leaves the row behind with an empty path when a sample fails."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"id": "zh000", "audio_path": "zh000.wav"},
                    {"id": "zh001", "audio_path": ""},
                    {"id": "zh002"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert read_manifest(manifest) == {"zh000": "zh000.wav"}


def test_the_summary_averages_only_the_paired_samples() -> None:
    """Averaging an arm over its own extra samples compares two different sets."""
    samples = _arm({"zh000": (150.0, 40.0, 3.0), "zh001": (170.0, 60.0, 4.0)})

    assert summarise(samples, ["zh000"])["f0_median_hz"] == 150.0
