"""How well the product's recogniser reads speech, which nobody had measured.

The OKR's first word is 能听 -- "can hear" -- with the criterion "Chinese ASR
CER <= 15%". Every CER in this project until now measured the other
direction: whether the audio we *produced* said the words it was supposed to.
That is the output leg. This is the input leg, and it had no number at all.

The circularity rule does not apply here and it is worth saying why. Scoring
our own generated speech with our own recogniser is circular because shared
failure modes cancel. Scoring the recogniser itself against ground-truth text
is not a comparison between two of our components -- it is a direct
measurement of one of them, and the reference comes from the corpus rather
than from us.

What it cannot do is stand in for real users. Every audio set available here
is synthesised, so this measures the recogniser on synthetic speech at the
corpus's own sample rate. Real far-field human speech is harder, and this
number is a floor on the error rather than an estimate of it. The report says
so rather than leaving the reader to assume otherwise.

    python scripts/measure_asr.py --rows artifacts/asr_probe.jsonl \
        --model-dir ~/omni/minimind-o/model/SenseVoiceSmall --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import (  # noqa: E402
    assess,
    character_error_rate,
)

TARGET_RATE = 16_000  # SenseVoice's frontend; anything else is resampled here


def load_pcm(path: Path) -> bytes:
    """int16 mono at 16 kHz, whatever the file was.

    The corpus audio is 12 kHz, and the contract's input is 16 kHz PCM, so the
    resampling happens somewhere either way. Doing it here keeps it visible
    instead of leaving it to whichever component notices first.
    """
    import numpy as np
    import soundfile as sf

    waveform, rate = sf.read(str(path))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != TARGET_RATE:
        import librosa

        waveform = librosa.resample(waveform.astype(float), orig_sr=rate, target_sr=TARGET_RATE)
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


async def transcribe_all(rows: list[dict[str, Any]], recogniser: Any) -> list[dict[str, Any]]:
    annotated = []
    for index, row in enumerate(rows, start=1):
        text, language = await recogniser.transcribe(load_pcm(Path(row["audio_path"])), TARGET_RATE)
        annotated.append({**row, "asr_transcript": text, "asr_language": language})
        if index % 20 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        required=True,
        type=Path,
        help="JSONL with audio_path and reference_text, the same shape the "
        "generation scripts already write",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="SenseVoice directory. Not needed when every row already carries "
        "an asr_transcript -- see the note on --rows",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="cores to use on CPU. Not all of them: this runs beside training more often than not",
    )
    parser.add_argument("--output", type=Path, help="annotated JSONL")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="the OKR's line. Passing means value plus three noise floors "
        "stays under it, not that the point estimate does",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Rescoring archived rows must not need the recogniser, the model directory,
    # or the machine that has them. The recogniser runs where the weights are,
    # scoring runs where zhconv is, and those have not been the same box once:
    # the first run of this script transcribed all 160 clips and then died on
    # the import, and the rows survived only because they are written before
    # the scoring step.
    if all(row.get("asr_transcript") is not None for row in rows):
        print(f"逐样本行已带转写，跳过识别器（n={len(rows)}）")
        annotated = rows
    else:
        if args.model_dir is None:
            raise SystemExit("这些行还没有 asr_transcript，需要 --model-dir 指向 SenseVoice")
        if args.device == "cpu" and args.cpu_threads > 0:
            import torch

            torch.set_num_threads(args.cpu_threads)

        from mindsurf_omni.service.asr import SenseVoiceRecogniser

        recogniser = SenseVoiceRecogniser(model_dir=args.model_dir, device=args.device)
        annotated = asyncio.run(transcribe_all(rows, recogniser))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in annotated)
            + "\n",
            encoding="utf-8",
        )

    by_language: dict[str, list[float]] = {}
    rates = []
    for row in annotated:
        rate = character_error_rate(row["reference_text"], row["asr_transcript"])
        rates.append(rate)
        by_language.setdefault(row.get("asr_language") or "unknown", []).append(rate)

    lines = [f"识别器 sensevoice-small，n={len(rates)}，音频重采样到 {TARGET_RATE} Hz"]
    measurement = assess("asr_cer", rates, effect_of_interest=args.threshold)
    upper = measurement.value + 3 * measurement.noise_floor
    verdict = "过" if upper < args.threshold else "不过"
    lines.append(f"  {measurement}")
    finite = [rate for rate in rates if math.isfinite(rate)]
    if finite:
        bad = sum(1 for rate in rates if rate > 0.3)
        lines.append(f"  中位 {statistics.median(finite):.4f}，超过 0.3 的 {bad}/{len(rates)}")
    lines.append(f"  判据 CER + 3×噪声底 = {upper:.4f} 对阈值 {args.threshold} —— {verdict}")
    for language, values in sorted(by_language.items()):
        lines.append(f"  判成 {language}: n={len(values)} 均值 {statistics.mean(values):.4f}")
    # Said out loud, because a number without it reads as "the product hears
    # this well" rather than "the recogniser reads this corpus this well".
    lines.append("  这是合成语音上的读数，真人远场更难——它是误差的下界，不是估计")

    print("\n".join(lines))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "recogniser": "sensevoice-small",
                    "sample_size": len(rates),
                    "cer": measurement.value,
                    "noise_floor": measurement.noise_floor,
                    "upper_bound_3sigma": upper,
                    "threshold": args.threshold,
                    "verdict": verdict,
                    "median": statistics.median(finite) if finite else None,
                    "by_language": {
                        language: {"n": len(values), "mean": statistics.mean(values)}
                        for language, values in sorted(by_language.items())
                    },
                    "caveat": "synthetic speech only; a floor on the error rate, not an estimate",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
