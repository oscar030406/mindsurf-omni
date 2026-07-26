"""Pass audio through Mimi and back, to find the ceiling the native path has.

The native path's Talker does not generate waveforms, it generates Mimi codes,
and those codes are decoded by a frozen codec at 12.5 Hz with eight codebooks.
So there is a quality level no native model can exceed no matter how well it
is trained, and it had never been measured -- which means every reading of
"our Talker is 0.96 UTMOS below edge-tts" silently attributes the whole gap to
the Talker.

Encoding real speech and decoding it back is the standard proxy for that
ceiling. What it substitutes is the *encoder*: a native model never encodes
anything, it emits codes directly, so this measures the best code sequence a
perfect model could aim at rather than one it actually produces. The decoder
half -- the part that limits fidelity -- is identical either way.

Deliberately does only the round trip. The three-stage split in EVALUATION.md
already owns transcription, naturalness and scoring, so the output is a
directory plus a manifest those scripts already read:

    python scripts/codec_roundtrip.py --input artifacts/tts_edge \
        --output artifacts/tts_edge_mimi --minimind-root ~/omni/minimind-o
    python scripts/transcribe_samples.py --manifest .../manifest.json \
        --output artifacts/tts_edge_mimi.jsonl --judge paraformer
    python scripts/measure_naturalness.py --scored ... --output ..._mos.jsonl
    python scripts/evaluate_speech.py --candidate ..._mos.jsonl \
        --reference artifacts/tts_edge_mos.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MIMI_RATE = 24_000  # what the codec was trained at; anything else is resampled
CODEBOOKS = 8


def repoint_manifest(
    manifest: dict[str, Any], source: Path, target: Path, frames_per_second: float | None
) -> dict[str, Any]:
    """Point every sample at the rebuilt audio, and stamp what rebuilt it.

    A manifest left pointing at the input would make the downstream report read
    the original clips while claiming to describe the round trip -- silently,
    and in the direction that flatters.
    """
    for sample in manifest.get("samples", []):
        if sample.get("audio_path"):
            sample["audio_path"] = str(target / Path(sample["audio_path"]).name)
    manifest.setdefault("generated_by", {})["codec_roundtrip"] = {
        "codec": "mimi",
        "codebooks": CODEBOOKS,
        "sample_rate": MIMI_RATE,
        "frames_per_second": frames_per_second,
        "source": str(source),
        "note": "encode+decode of real speech: a ceiling proxy for the native path, "
        "not a model reading -- no model produced these codes",
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="directory of wavs + manifest.json"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="cores on CPU. Not all of them: this runs beside training more often than not",
    )
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    if args.device == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    from transformers import MimiModel

    root = args.minimind_root.expanduser().resolve()
    mimi = MimiModel.from_pretrained(str(root / "model" / "mimi")).eval().float().to(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    clips = sorted(args.input.glob("*.wav"))
    if not clips:
        raise SystemExit(f"no wavs in {args.input}")

    frames_total = seconds_total = 0.0
    for index, clip in enumerate(clips, start=1):
        waveform, rate = sf.read(str(clip))
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        if rate != MIMI_RATE:
            import librosa

            waveform = librosa.resample(waveform.astype(float), orig_sr=rate, target_sr=MIMI_RATE)
        audio = torch.tensor(waveform, dtype=torch.float32, device=args.device)
        with torch.no_grad():
            codes = mimi.encode(audio.reshape(1, 1, -1)).audio_codes[:, :CODEBOOKS]
            restored = mimi.decode(codes).audio_values.squeeze().cpu().numpy()
        # Written at the codec's own rate, not the input's: resampling the
        # output too would fold a second resampler's error into the reading.
        sf.write(str(args.output / clip.name), np.clip(restored, -1.0, 1.0), MIMI_RATE)
        frames_total += codes.shape[-1]
        seconds_total += len(waveform) / MIMI_RATE
        if index % 20 == 0:
            print(f"  {index}/{len(clips)}", flush=True)

    manifest_path = args.input / "manifest.json"
    if manifest_path.is_file():
        rate_hz = round(frames_total / seconds_total, 3) if seconds_total else None
        rewritten = repoint_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")), args.input, args.output, rate_hz
        )
        (args.output / "manifest.json").write_text(
            json.dumps(rewritten, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(f"  没有 {manifest_path}，只写了音频（下游脚本要 manifest）")

    print(f"{len(clips)} 条重建到 {args.output}")
    if seconds_total:
        print(f"  帧率 {frames_total / seconds_total:.3f} 帧/秒（Mimi 标称 12.5）")
    print("  这是**天花板代理**，不是模型读数：真的原生模型直接吐码，不编码任何东西")


if __name__ == "__main__":
    main()
