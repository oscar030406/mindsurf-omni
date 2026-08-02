"""Mix noise into clean speech at controlled SNRs, to give the ASR a stress axis.

Every ASR number this project has was measured on clean synthesised speech, and
the report that carries it says so: "synthetic speech only; a floor on the error
rate, not an estimate". A voice product is used in a kitchen, a car, a room with
other people. Between 0.0315 on clean speech and whatever the real number is,
there is a curve nobody has drawn.

This draws it. Clean clips in, noisy copies out at each requested SNR, and the
rows file the ASR harness already reads.

**The noise is synthetic, and that is a real limit.** Gaussian noise is
stationary and spectrally flat; babble made from other speech is neither, and
babble is what actually breaks recognisers because it competes for the same
frequencies and the same attention. Real recorded noise would be better --
AISHELL-5 ships 40 hours of in-car recordings -- and this is deliberately built
so the noise source can be swapped for files without touching the mixing.

SNR is computed per clip against that clip's own RMS, so a quiet clip and a loud
one get the same ratio rather than the same absolute noise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def scale_for_snr(speech_rms: float, noise_rms: float, snr_db: float) -> float:
    """Gain on the noise that puts it snr_db below this clip's speech.

    Per clip rather than per batch: the clips differ in level, and a single
    absolute noise level would mean the quiet ones are tested harder than the
    loud ones while the report claims one SNR for all of them.
    """
    if noise_rms <= 0 or speech_rms <= 0:
        return 0.0
    return float(speech_rms / (noise_rms * math.pow(10.0, snr_db / 20.0)))


def babble(samples: int, rng: Any, sources: list[Any]) -> Any:
    """Overlapping speech, which is the noise that actually breaks recognisers.

    Built by summing random excerpts of other clips in the set. Not as good as
    recorded babble -- the voices are all one synthesiser -- but far closer to
    the failure mode than white noise, and it costs no new data.
    """
    import numpy

    mixed = numpy.zeros(samples, dtype="float32")
    for _ in range(6):
        source = sources[rng.randrange(len(sources))]
        if len(source) < samples:
            source = numpy.tile(source, samples // max(1, len(source)) + 1)
        start = rng.randrange(max(1, len(source) - samples))
        mixed += source[start : start + samples]
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="speak_texts manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="directory for the noisy rows")
    parser.add_argument(
        "--snr", type=float, nargs="+", default=[20.0, 15.0, 10.0, 5.0, 0.0], help="dB"
    )
    parser.add_argument("--kind", choices=("white", "babble"), default="babble")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    import random

    import numpy
    import soundfile

    rng = random.Random(args.seed)
    generator = numpy.random.default_rng(args.seed)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent
    clips = []
    for sample in manifest["samples"]:
        if not sample.get("audio_path"):
            continue
        path = root / Path(sample["audio_path"]).name
        if not path.is_file():
            continue
        audio, rate = soundfile.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        clips.append((sample, audio, rate))
    if not clips:
        raise SystemExit(f"{args.manifest} named no readable audio next to it")
    print(f"读到 {len(clips)} 条干净语音")

    pool = [audio for _, audio, _ in clips]
    for snr in args.snr:
        directory = args.output / f"snr{snr:g}"
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for sample, audio, rate in clips:
            noise = (
                generator.normal(0.0, 1.0, len(audio)).astype("float32")
                if args.kind == "white"
                else babble(len(audio), rng, pool)
            )
            speech_rms = float(numpy.sqrt(numpy.mean(audio**2)))
            noise_rms = float(numpy.sqrt(numpy.mean(noise**2)))
            mixed = audio + noise * scale_for_snr(speech_rms, noise_rms, snr)
            # Clipping here would add a distortion the SNR label does not
            # describe, so scale the whole clip down instead of letting it rail.
            peak = float(numpy.max(numpy.abs(mixed)))
            if peak > 1.0:
                mixed = mixed / peak
            path = directory / f"{sample['id']}.wav"
            soundfile.write(str(path), mixed, rate)
            rows.append(
                {
                    "id": sample["id"],
                    "audio_path": str(path.name),
                    "reference_text": sample["reference_text"],
                    "snr_db": snr,
                    "noise": args.kind,
                }
            )
        (directory / "rows.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"  SNR {snr:g} dB -> {directory}")

    print("\n噪声是合成的（同一个合成器的语音叠出来的 babble），")
    print("**这是估计不是实测**：真实环境的噪声非平稳、频谱不同，量出来的曲线会不一样。")


if __name__ == "__main__":
    main()
