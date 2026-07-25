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
    compare_paired,
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
    id: str | None = None  # what lets two runs pair sample-for-sample
    # Which recogniser produced the transcript. whisper-small and paraformer-zh
    # read the same 160 clips 0.08 apart, so two arms judged differently are
    # not comparable and this is what lets that be caught rather than assumed.
    judge: str | None = None


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


def paired_deltas(
    candidate: list[Sample], reference: list[Sample]
) -> dict[str, list[float]] | None:
    """Per-sample differences on shared ids, or None when pairing is invalid.

    Valid pairing needs two things: ids on both sides, and the same reference
    text per id. The second is the load-bearing one -- on the fixed-text
    protocol the texts match and per-item difficulty cancels in the
    subtraction; on free-run output the texts differ and a "pair" would
    subtract two unrelated numbers. Refusing is better than quietly producing
    a smaller, wrong noise floor.
    """
    by_id = {sample.id: sample for sample in reference if sample.id}
    pairs = [(mine, by_id[mine.id]) for mine in candidate if mine.id and mine.id in by_id]
    if len(pairs) < max(2, int(0.8 * min(len(candidate), len(reference)))):
        return None
    if any(mine.reference_text != theirs.reference_text for mine, theirs in pairs):
        return None

    deltas: dict[str, list[float]] = {"cer": [], "utmos": []}
    for mine, theirs in pairs:
        if mine.transcript is not None and theirs.transcript is not None:
            deltas["cer"].append(
                character_error_rate(mine.reference_text, mine.transcript or "")
                - character_error_rate(theirs.reference_text, theirs.transcript or "")
            )
        if mine.utmos is not None and theirs.utmos is not None:
            deltas["utmos"].append(mine.utmos - theirs.utmos)
    return {key: values for key, values in deltas.items() if values}


def judges_of(samples: list[Sample]) -> set[str]:
    """Which recognisers produced these transcripts, as far as the rows say.

    Empty means the rows predate the field. That is unknown, not agreement --
    the caller says so rather than assuming the run it is comparing against
    used the same judge.
    """
    return {sample.judge for sample in samples if sample.judge}


def check_same_judge(candidate: list[Sample], reference: list[Sample]) -> str | None:
    """The reason these two runs may not be compared, or None.

    A CER difference of 0.08 comes free with a change of judge, which is larger
    than most effects this harness is asked to certify. Refusing is the whole
    point: a comparison across judges is not a weaker result, it is a
    measurement of the recognisers.
    """
    for label, samples in (("候选", candidate), ("参照", reference)):
        found = judges_of(samples)
        if len(found) > 1:
            return f"{label}臂里混了两个判官 {sorted(found)}——同一臂内必须同一个"
    mine, theirs = judges_of(candidate), judges_of(reference)
    if mine and theirs and mine != theirs:
        return (
            f"候选臂判官 {sorted(mine)}，参照臂 {sorted(theirs)}——"
            "whisper-small 与 paraformer-zh 在同一批音频上差 0.08，"
            "这不是两个模型的差别，是两个判官的差别"
        )
    return None


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
                id=row.get("id"),
                judge=row.get("judge"),
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
        mismatch = check_same_judge(candidate.samples, reference.samples)
        if mismatch:
            raise SystemExit(f"两臂判官不一致，拒绝比较：{mismatch}")
        if not (judges_of(candidate.samples) and judges_of(reference.samples)):
            # Not fatal: the field is newer than some of the artifacts on disk.
            # Said out loud, because "no judge recorded" reads as "same judge"
            # to everyone who does not know the field exists.
            lines.append("判官未记录（其中一臂没有 judge 字段）——同判官这件事没有被检查")
        payload["reference"] = reference.to_json()
        payload["judge"] = {
            "candidate": sorted(judges_of(candidate.samples)),
            "reference": sorted(judges_of(reference.samples)),
        }
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

        deltas = paired_deltas(candidate.samples, reference.samples)
        if deltas:
            lines.append("")
            lines.append("配对对比（同 id 同文本，逐样本相减）:")
            payload["paired_comparison"] = {}
            for key in sorted(deltas):
                verdict = compare_paired(key, deltas[key], lower_is_better=key != "utmos")
                lines.append("  " + verdict)
                payload["paired_comparison"][key] = verdict
        elif any(sample.id for sample in candidate.samples):
            lines.append("")
            lines.append(
                "配对对比：不可用（id 或文本不对齐——free-run 输出属正常，用上面的非配对对比）"
            )

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
