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
import math
import statistics
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
    # The mean's shape, because the mean alone has misled this project once.
    # Two systems reached similar means by different failure patterns: ours
    # missed characters on nearly every sentence (median 0.2944, 78 of 160 over
    # 0.3), upstream's was accurate except on a few hard ones (median 0.0500,
    # 7 of 160). A mean that improves while the median holds means the
    # catastrophes thinned out, not that the speech got better -- and those two
    # want different fixes.
    shape: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_size": len(self.samples),
            "shape": self.shape,
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


def score(
    name: str, samples: list[Sample], effects: dict[str, float], fold_numbers: bool = False
) -> Report:
    report = Report(name=name, samples=samples)

    transcribed = [sample for sample in samples if sample.transcript is not None]
    if transcribed:
        rates = [
            character_error_rate(sample.reference_text, sample.transcript or "", fold_numbers)
            for sample in transcribed
        ]
        report.measurements["cer"] = assess(
            "cer", rates, effect_of_interest=effects.get("cer", 0.05)
        )
        finite = [rate for rate in rates if math.isfinite(rate)]
        report.shape = {
            "cer_median": statistics.median(finite) if finite else float("nan"),
            # The count, not the fraction: "78 of 160" is what a reader can
            # check against the samples, and it is how the failure shape was
            # first noticed.
            "cer_over_0_3": float(sum(1 for rate in rates if rate > 0.3)),
            "sample_size": float(len(rates)),
        }

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
    candidate: list[Sample], reference: list[Sample], fold_numbers: bool = False
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
                character_error_rate(mine.reference_text, mine.transcript or "", fold_numbers)
                - character_error_rate(theirs.reference_text, theirs.transcript or "", fold_numbers)
            )
        if mine.utmos is not None and theirs.utmos is not None:
            deltas["utmos"].append(mine.utmos - theirs.utmos)
    return {key: values for key, values in deltas.items() if values}


def shape_lines(report: Report) -> list[str]:
    """The median beside the mean, and how many samples are badly wrong.

    Printed rather than left in the JSON: the whole point is that a reader who
    only sees the mean draws the wrong conclusion about what to fix.
    """
    if not report.shape:
        return []
    total = int(report.shape["sample_size"])
    over = int(report.shape["cer_over_0_3"])
    return [
        f"  形状: 中位 CER {report.shape['cer_median']:.4f}，{over}/{total} 超过 0.3"
        + ("（普遍念不准）" if over > total * 0.25 else "（少数难例）")
    ]


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
        "--instrument-only",
        action="store_true",
        help="the model never spoke: the text was read as written, so the CER "
        "belongs to the synthesiser and the judge. Marked in the report because "
        "a floor read as a model score is the best-looking wrong number here",
    )
    parser.add_argument(
        "--cer-effect",
        type=float,
        default=0.05,
        help="the CER difference worth detecting; an instrument that cannot "
        "resolve it is reported but may not judge",
    )
    parser.add_argument(
        "--fold-numerals",
        action="store_true",
        help="write 二零二一 as 2021 on both sides before comparing, so a "
        "synthesiser that correctly reads out a date is not charged a "
        "substitution per digit. Off by default: it moves every CER in this "
        "project, and the acceptance thresholds were calibrated without it",
    )
    args = parser.parse_args()

    effects = {"cer": args.cer_effect}
    candidate = score("candidate", load(args.candidate), effects, args.fold_numerals)

    lines = [f"候选 n={len(candidate.samples)}"]
    if args.fold_numerals:
        lines.insert(0, "阿拉伯数字已折叠（两边都过 cn2an）——与未折叠的历史数字不可比")
    for _, measurement in sorted(candidate.measurements.items()):
        mark = "有门控资格" if measurement.gating_eligible else f"仅报告（{measurement.note}）"
        lines.append(f"  {measurement}  {mark}")
    lines.extend(shape_lines(candidate))

    # In the artifact, not only on the screen: a stored CER whose normalisation
    # is unrecorded is a number nobody can pair against later.
    payload: dict[str, Any] = {
        "candidate": candidate.to_json(),
        "normalisation": {"fold_numerals": args.fold_numerals},
    }
    if args.instrument_only:
        lines.insert(0, "模型没有参与这一轮：下面的 CER 是合成器加判官的底噪，不是模型质量")
        payload["instrument_only"] = True

    if args.reference:
        reference = score("reference", load(args.reference), effects, args.fold_numerals)
        mismatch = check_same_judge(candidate.samples, reference.samples)
        if mismatch:
            raise SystemExit(f"两臂判官不一致，拒绝比较：{mismatch}")
        if not (judges_of(candidate.samples) and judges_of(reference.samples)):
            # Not fatal: the field is newer than some of the artifacts on disk.
            # Said out loud, because "no judge recorded" reads as "same judge"
            # to everyone who does not know the field exists.
            lines.append("判官未记录（其中一臂没有 judge 字段）——同判官这件事没有被检查")
        payload["reference"] = reference.to_json()
        lines.append(f"参照 n={len(reference.samples)}")
        lines.extend(shape_lines(reference))
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

        deltas = paired_deltas(candidate.samples, reference.samples, args.fold_numerals)
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
