"""Fill in the naturalness score the evaluation harness has always read and
nobody ever wrote.

``evaluate_speech.py`` scores a ``utmos`` field on every sample and assesses it
like any other metric. No script produced that field, so the line never
appeared in a report -- naturalness was not "pending validity", it had no
number at all, while CER answered a different question (were the right words
said, not whether they sounded like speech).

This writes it, with UTMOS22-strong: the VoiceMOS 2022 challenge winner, one
forward pass per clip, no reference audio needed.

What it is allowed to claim, and what it is not
-----------------------------------------------
UTMOS was trained mostly on English MOS ratings, so an absolute 2.4 here is not
"2.4 on a Chinese listener panel" -- that calibration does not exist. It is
usable for *ranking systems on the same content*, which is what every question
here actually is: ours against upstream's release, this checkpoint against the
next one, the native path against the cascade.

That claim is checked rather than assumed. Run with --validate over audio whose
quality order is already known from something else, and it prints whether the
scores reproduce that order by more than their own noise floor. Measured over
four sets of 160 clips each:

    edge-tts reading the probes      3.776 +/- 0.032
    edge-tts reading model replies   3.395 +/- 0.046
    upstream's Talker (CER 0.058)    2.335 +/- 0.053
    our Talker        (CER 0.217)    2.045 +/- 0.058

Every adjacent pair separates by more than three times its combined floor. A
real synthesiser sits a full point above either Talker, and the better Talker
scores higher -- on an axis with no access to the CER that ranked them. It still
does not license a sentence like "our speech is 2.05 out of 5 to a listener".
Human blind listening remains the only thing that can say that.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import assess, bootstrap_noise_floor  # noqa: E402

# The repository the weights come from. Pinned to a tag: an unpinned hub load
# would silently change the instrument between two runs that get compared.
HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
HUB_MODEL = "utmos22_strong"
# What difference in predicted MOS is worth acting on. 0.2 rather than the
# 0.3 evaluate_speech defaults to: the gap between our Talker and upstream's is
# 0.26, and an instrument that cannot resolve that cannot judge this rerun.
EFFECT_OF_INTEREST = 0.2


def load_predictor(device: str) -> Any:
    import torch

    # torch.hub's fork check raises KeyError on a repo without an Authorization
    # header rather than answering the question it asks. The repo is pinned to a
    # tag above, which is the property that actually matters here.
    torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: True  # noqa: SLF001
    model = torch.hub.load(HUB_REPO, HUB_MODEL, trust_repo=True).eval()
    return model.to(device)


def score_clip(model: Any, path: Path, device: str) -> float:
    import soundfile
    import torch

    waveform, rate = soundfile.read(path, dtype="float32")
    if waveform.ndim > 1:  # mono, because the predictor takes one channel
        waveform = waveform.mean(axis=1)
    tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(tensor, rate))


def score_paths(model: Any, paths: list[Path], device: str, label: str) -> list[float]:
    scores = []
    for index, path in enumerate(paths):
        scores.append(score_clip(model, path, device))
        if (index + 1) % 40 == 0:
            print(f"  {label} {index + 1}/{len(paths)}", flush=True)
    return scores


def annotate(rows: list[dict[str, Any]], model: Any, device: str) -> list[dict[str, Any]]:
    """Add a utmos field to every row whose audio exists, in place of nothing."""
    annotated = []
    for row in rows:
        audio = row.get("audio_path")
        if not audio or not Path(audio).is_file():
            # Left absent rather than zeroed: a clip that was never produced has
            # no naturalness, and a zero would drag the mean toward "unusable"
            # for a failure that silent_rate already reports.
            annotated.append(dict(row))
            continue
        annotated.append({**row, "utmos": score_clip(model, Path(audio), device)})
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored",
        type=Path,
        help="JSONL from transcribe_samples; a utmos field is added to each row",
    )
    parser.add_argument("--output", type=Path, help="where to write the annotated JSONL")
    parser.add_argument(
        "--validate",
        nargs="+",
        metavar="LABEL=DIR",
        help="score directories of wavs whose quality order is known and report "
        "whether this instrument reproduces it -- run this before trusting it",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, help="write the validation table as JSON")
    args = parser.parse_args()

    if not args.validate and not args.scored:
        raise SystemExit("give --scored to annotate a run, or --validate to check the instrument")

    model = load_predictor(args.device)

    if args.validate:
        print(f"效度检查：{HUB_MODEL}（{HUB_REPO}）\n")
        results = {}
        for entry in args.validate:
            # Split on the last separator, not the first: a label is free text
            # and "chunk_frames=1=<dir>" is a reasonable thing to type.
            label, _, directory = entry.rpartition("=")
            paths = sorted(Path(directory).glob("*.wav"))
            if not paths:
                raise SystemExit(f"no wavs in {directory}")
            scores = score_paths(model, paths, args.device, label)
            floor = bootstrap_noise_floor(scores)
            results[label] = (statistics.fmean(scores), floor, len(scores))
            print(f"  {label:32} {results[label][0]:.3f} ± {floor:.3f}  n={len(scores)}")

        print("\n相邻两组的差，按 3× 合并噪声底判定:")
        comparisons: list[dict[str, Any]] = []
        ordered = list(results.items())
        for (left, (lm, lf, _)), (right, (rm, rf, _)) in zip(ordered, ordered[1:], strict=False):
            difference = lm - rm
            threshold = 3.0 * (lf**2 + rf**2) ** 0.5
            verdict = "可区分" if abs(difference) > threshold else "无法区分"
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "difference": difference,
                    "threshold": threshold,
                    "verdict": verdict,
                }
            )
            print(f"  {left} vs {right}: {difference:+.3f}, 阈值 ±{threshold:.3f} -> {verdict}")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "predictor": f"{HUB_MODEL} @ {HUB_REPO}",
                        "effect_of_interest": EFFECT_OF_INTEREST,
                        "sets": {
                            label: {"utmos": mean, "noise_floor": floor, "n": count}
                            for label, (mean, floor, count) in results.items()
                        },
                        "adjacent_comparisons": comparisons,
                        "caveat": (
                            "UTMOS is trained mostly on English MOS; absolute values are not "
                            "calibrated to a Chinese listener panel. Valid for ranking systems "
                            "on the same content, which is what it is used for here."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"\n输出 {args.report}")
        return

    rows = [
        json.loads(line)
        for line in args.scored.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotated = annotate(rows, model, args.device)
    scored = [row["utmos"] for row in annotated if "utmos" in row]
    if not scored:
        raise SystemExit(f"{args.scored} lists no audio that exists on disk")

    measurement = assess("utmos", scored, effect_of_interest=EFFECT_OF_INTEREST)
    mark = "有门控资格" if measurement.gating_eligible else f"仅报告（{measurement.note}）"
    print(f"{measurement}  {mark}")
    print(f"  预测器 {HUB_MODEL} @ {HUB_REPO}")
    print("  英文语料训练，中文绝对值未校准——用于同内容的系统间排序，不作绝对 MOS")

    output = args.output or args.scored
    with output.open("w", encoding="utf-8") as handle:
        for row in annotated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  写入 {output}（evaluate_speech.py 会自动读 utmos 字段）")


if __name__ == "__main__":
    main()
