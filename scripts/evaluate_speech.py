"""Score a speech model, and say plainly what the score is not allowed to claim.

Two rules this harness exists to enforce, both learned the expensive way.

CER is computed with a *third-party* recogniser, never our own audio encoder.
Scoring a model with a component of that same model is circular: shared
failure modes cancel, and the number flatters exactly where it should warn.

Every metric is reported with its noise floor, and one that cannot resolve the
effect under test is printed as reported-only. It stays in the report -- hiding
a weak instrument is its own kind of dishonesty -- but it may not decide pass
or fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import (  # noqa: E402
    Measurement,
    assess,
    character_error_rate,
    compare,
)
from mindsurf_omni.evaluation.text_regression import (  # noqa: E402
    assess_text_regression,
)


@dataclass
class Sample:
    """One prompt, what the model should have said, and what it did say."""

    prompt: str
    reference_text: str
    audio_path: Path | None = None
    transcript: str | None = None  # third-party ASR of the generated audio
    utmos: float | None = None


@dataclass
class Report:
    name: str
    samples: list[Sample]
    measurements: dict[str, Measurement] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_size": len(self.samples),
            "measurements": {
                key: {
                    "value": measurement.value,
                    "noise_floor": measurement.noise_floor,
                    "sample_size": measurement.sample_size,
                    "gating_eligible": measurement.gating_eligible,
                    "note": measurement.note,
                }
                for key, measurement in self.measurements.items()
            },
            "not_claimed": sorted(
                key
                for key, measurement in self.measurements.items()
                if not measurement.gating_eligible
            ),
        }


def score(name: str, samples: list[Sample], effects: dict[str, float]) -> Report:
    report = Report(name=name, samples=samples)

    transcribed = [sample for sample in samples if sample.transcript is not None]
    if transcribed:
        report.measurements["cer"] = assess(
            "cer",
            [
                character_error_rate(sample.reference_text, sample.transcript or "")
                for sample in transcribed
            ],
            effect_of_interest=effects.get("cer", 0.05),
        )

    rated = [sample.utmos for sample in samples if sample.utmos is not None]
    if rated:
        report.measurements["utmos"] = assess(
            "utmos", rated, effect_of_interest=effects.get("utmos", 0.3)
        )

    # Silence is a failure mode a CER on the spoken samples alone would miss:
    # a model that emits nothing has nothing to be scored wrong.
    silent = sum(1 for sample in samples if not (sample.transcript or "").strip())
    report.measurements["silent_rate"] = assess(
        "silent_rate",
        [1.0 if not (sample.transcript or "").strip() else 0.0 for sample in samples],
        effect_of_interest=effects.get("silent_rate", 0.05),
    )
    if silent:
        report.measurements["silent_rate"] = Measurement(
            name="silent_rate",
            value=report.measurements["silent_rate"].value,
            noise_floor=report.measurements["silent_rate"].noise_floor,
            sample_size=len(samples),
            gating_eligible=report.measurements["silent_rate"].gating_eligible,
            note=f"{silent} of {len(samples)} produced no speech",
        )
    return report


def load(path: Path) -> list[Sample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(
            Sample(
                prompt=row["prompt"],
                reference_text=row["reference_text"],
                audio_path=Path(row["audio_path"]) if row.get("audio_path") else None,
                transcript=row.get("transcript"),
                utmos=row.get("utmos"),
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path, help="JSONL of scored samples")
    parser.add_argument("--reference", type=Path, help="the model being compared against")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-losses",
        type=Path,
        help="JSON list of strict holdout losses; without it the report cannot "
        "say whether the audio training eroded the text ability",
    )
    parser.add_argument(
        "--cer-effect",
        type=float,
        default=0.05,
        help="the CER difference worth detecting; an instrument that cannot "
        "resolve it is reported but may not judge",
    )
    args = parser.parse_args()

    effects = {"cer": args.cer_effect}
    candidate = score("candidate", load(args.candidate), effects)

    lines = [f"候选 n={len(candidate.samples)}"]
    for _, measurement in sorted(candidate.measurements.items()):
        mark = "有门控资格" if measurement.gating_eligible else f"仅报告（{measurement.note}）"
        lines.append(f"  {measurement}  {mark}")

    payload: dict[str, Any] = {"candidate": candidate.to_json()}

    if args.reference:
        reference = score("reference", load(args.reference), effects)
        payload["reference"] = reference.to_json()
        lines.append("")
        lines.append("对比:")
        for key in sorted(set(candidate.measurements) & set(reference.measurements)):
            lower_is_better = key != "utmos"
            lines.append(
                "  "
                + compare(
                    key,
                    candidate.measurements[key],
                    reference.measurements[key],
                    lower_is_better=lower_is_better,
                )
            )
        payload["comparison"] = {
            key: compare(
                key,
                candidate.measurements[key],
                reference.measurements[key],
                lower_is_better=key != "utmos",
            )
            for key in sorted(set(candidate.measurements) & set(reference.measurements))
        }

    if args.strict_losses:
        losses = json.loads(args.strict_losses.read_text(encoding="utf-8"))
        regression = assess_text_regression(losses)
        lines.append("")
        lines.append(f"文本能力: {regression}")
        if regression.note:
            lines.append(f"  {regression.note}")
        payload["text_regression"] = {
            "value": regression.value,
            "baseline": regression.baseline,
            "difference": regression.difference,
            "threshold": regression.threshold,
            "verdict": regression.verdict,
            "note": regression.note,
        }
    else:
        # Named rather than omitted: a report silent on this has not shown that
        # the audio training left the language ability intact, it has not
        # looked.
        lines.append("")
        lines.append("文本能力: 未测（未提供 --strict-losses）")
        payload["text_regression"] = None

    print("\n".join(lines))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
