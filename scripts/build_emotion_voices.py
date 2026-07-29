"""Emotional variants of the built-in voices, made from each voice's own audio.

The native path turned out to support emotion without any training: prosody
rides the same reference channel as identity, so a reference clip that sounds
excited produces speech that sounds excited. See
docs/experiments/2026-07-29-emotion-through-the-reference.md.

One constraint from that measurement decides the whole design. Emotion and
identity are entangled in the speaker vector -- the same synthetic voice
re-rendered with different prosody scored a CAM++ cosine of 0.33 against
itself, well under the 0.56-0.65 the clone metric treats as the same person.
So a voice's emotional variant cannot be borrowed from another speaker. It has
to be built from that voice's own reference audio, which is what this does:
decode the stored codes, move the prosody, re-encode, re-embed.

The first attempt at this shipped nothing, and the reason is worth keeping.
librosa's pitch_shift is resampling-based: it moves the whole frequency axis,
so formants travel with F0 and a large enough shift turns the speaker into
somebody else. Every one of the ten variants came back under the identity
floor, 0.06 to 0.44. A formant-preserving vocoder (WORLD, praat) is the right
tool and neither is installed on the box, and the box is mid-training in the
one virtualenv that could install them, so this takes the route that needs no
new package:

* **The identity vector is not rebuilt.** Each variant keeps the voice's
  original ``spk_emb`` and changes only ``ref_codes``. Identity and prosody
  reach the model through two inputs; anchoring one and moving the other is a
  one-line answer to the entanglement that the 0.33 cosine measured. This is
  the change that matters most and it removes the failure above by
  construction.
* **A rate-only arm.** Phase-vocoder time stretching does not touch the
  frequency axis, so it cannot move formants. Rate is a genuine emotional cue
  on its own -- the reference experiment moved duration 12/12 -- so this arm is
  the formant-safe fallback if pitch turns out to cost too much.
* **Shifts are computed per voice and capped.** A fixed semitone count means a
  different Hz effect on a 117 Hz voice than on a 260 Hz one; a fixed Hz target
  means a different semitone count, and on the low voices a destructive one.
  The target is in Hz, converted per voice, and refused past the cap.

Two quiet failures are measured rather than assumed:

* **The codec eats the manipulation.** Mimi at 12.5 Hz and 8 codebooks is a
  narrow pipe, and the manipulated waveform is out of its training
  distribution. Everything is reported after encode-decode, not before.
* **The pipeline has its own floor.** Decoding and re-encoding alone costs
  cosine: this project measured the clone ceiling at 0.8409. So an identity
  control -- zero shift, unit rate, through every step -- runs first, and the
  variants are read against it rather than against 1.0.

**What this file does not decide, on purpose.** Because the identity vector is
anchored, the identity content of the manipulated codes is no longer
disqualifying -- the model reads identity from the vector it was given and
prosody from the codes. Measuring the codes' own embedding and gating on it
would be testing something the model never sees, which is the mistake the first
build made in the other direction. So those numbers are reported and nothing is
refused on them. The criterion that decides is the clone score of *generated*
audio against the *base* voice's stored embedding, and it needs the model.
Until that has run, the output is a candidate, not a deliverable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# Reported, not enforced. A variant's codes losing speaker information is
# expected once the identity vector is anchored, and how much of it matters is
# a question only generation can answer. Kept as a number so a later run can
# correlate it with the clone score instead of guessing.
IDENTITY_FRACTION = 0.85

# Without formant preservation, shifts past this are the measured failure: the
# first build asked for 4 semitones and produced a different speaker.
MAX_SEMITONES = 2.0

# Targets in Hz on the OUTPUT, doubled for the ~0.5 transfer ratio, then
# converted per voice. The cascade serves happy at +25 Hz and care at -10 Hz
# (EDGE_PROSODY), so the reference has to move +50 and -20.
EMOTIONS: dict[str, dict[str, float]] = {
    "happy": {"hz": 50.0, "rate": 1.16},
    "care": {"hz": -20.0, "rate": 0.84},
    # Formant-safe arms: no pitch shift at all, only rate.
    "fast": {"hz": 0.0, "rate": 1.20},
    "slow": {"hz": 0.0, "rate": 0.80},
}


def semitones_for(base_f0: float, target_hz: float) -> tuple[float, bool]:
    """How far to shift this particular voice, and whether the cap bit.

    A semitone is a ratio and a Hz target is a difference, so the conversion is
    per voice or it is wrong on one end of the range: +50 Hz is 6.2 semitones
    on a 117 Hz voice and 3.0 on a 260 Hz one.
    """
    import math

    if target_hz == 0.0 or not math.isfinite(base_f0) or base_f0 <= 0:
        return 0.0, False
    wanted = 12.0 * math.log2((base_f0 + target_hz) / base_f0)
    capped = min(MAX_SEMITONES, max(-MAX_SEMITONES, wanted))
    return capped, capped != wanted


def f0_median(wave: Any, rate: int) -> float:
    """Median voiced F0, or nan when there is not enough voiced audio to judge."""
    import librosa
    import numpy

    f0 = librosa.yin(wave.astype("float32"), fmin=60, fmax=500, sr=rate)
    voiced = f0[(f0 > 65) & (f0 < 480)]
    if len(voiced) < 10:
        return float("nan")
    return float(numpy.median(voiced))


def spectral_envelope(magnitude: Any, quefrency: int = 40) -> Any:
    """The slow part of the log spectrum -- the vocal tract, not the harmonics.

    Cepstral liftering: a log spectrum is the sum of a slowly varying envelope
    (formants) and a fast-varying excitation (the harmonic comb). Keeping only
    the low quefrency coefficients keeps the first and discards the second.
    """
    import numpy

    log_magnitude = numpy.log(numpy.maximum(magnitude, 1e-8))
    cepstrum = numpy.fft.irfft(log_magnitude, axis=0)
    cepstrum[quefrency:-quefrency] = 0.0
    return numpy.exp(numpy.real(numpy.fft.rfft(cepstrum, axis=0)))


def manipulate(wave: Any, rate: int, semitones: float, speed: float, formants: bool = True) -> Any:
    """Move pitch and rate, and by default leave the formants where they were.

    librosa's pitch shift is resampling-based: it scales the whole frequency
    axis, so the vocal tract moves with the pitch and a shift big enough to
    carry emotion turns the speaker into someone else. That is not a theory --
    the first build of this pack asked for four semitones and every one of ten
    variants came back under the identity floor, 0.06 to 0.44.

    The correction is standard and needs no new package. Shift as before, then
    divide out the shifted envelope and multiply the original one back in.
    Harmonics stay where the shift put them; formants return to where the
    speaker's own vocal tract had them. Frames line up one to one because
    pitch_shift preserves length.
    """
    import librosa
    import numpy

    shifted = librosa.effects.pitch_shift(wave, sr=rate, n_steps=semitones)
    if formants and semitones != 0.0:
        # 1024 at 24 kHz is 42 ms: long enough to resolve a 110 Hz voice's
        # harmonics, short enough not to smear the stops Chinese needs.
        n_fft, hop = 1024, 256
        before = librosa.stft(wave, n_fft=n_fft, hop_length=hop)
        after = librosa.stft(shifted, n_fft=n_fft, hop_length=hop)
        frames = min(before.shape[1], after.shape[1])
        correction = spectral_envelope(numpy.abs(before[:, :frames])) / numpy.maximum(
            spectral_envelope(numpy.abs(after[:, :frames])), 1e-8
        )
        # float32 all the way out: the correction is built from numpy defaults
        # and would otherwise hand the codec a float64 waveform, which fails on
        # the first convolution rather than anywhere informative.
        shifted = librosa.istft(
            after[:, :frames] * correction, hop_length=hop, length=len(wave)
        ).astype("float32")
    return librosa.effects.time_stretch(shifted, rate=speed).astype("float32")


def build(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import MimiModel

    from scripts.measure_voice_clone import Embedder, cosine

    codec = MimiModel.from_pretrained(str(args.codec)).eval()
    embedder = Embedder(args.minimind_root, "cpu")
    import librosa

    voices: dict[str, Any] = {}
    for pack in ("voices.pt", "voices_unseen.pt"):
        path = args.voices / pack
        if path.is_file():
            voices.update(torch.load(str(path), map_location="cpu", weights_only=False))
    if args.only:
        voices = {name: entry for name, entry in voices.items() if name in set(args.only)}
    if not voices:
        raise SystemExit(f"no voices under {args.voices}")

    out: dict[str, Any] = {}
    report: dict[str, Any] = {}
    weak: list[str] = []

    for name, entry in sorted(voices.items()):
        codes = entry["ref_codes"]
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        with torch.no_grad():
            original = codec.decode(codes)[0][0, 0].numpy()
        base_f0 = f0_median(original, 24_000)
        base_emb = embedder(
            torch.tensor(librosa.resample(original, orig_sr=24_000, target_sr=16_000))
        )

        def through_pipeline(wave: Any) -> tuple[Any, Any, float, Any]:
            """Everything the model will actually be handed, and what it reads."""
            with torch.no_grad():
                new_codes = codec.encode(torch.tensor(wave)[None, None, :]).audio_codes[0][:8]
                decoded = codec.decode(new_codes.unsqueeze(0))[0][0, 0].numpy()
            emb = embedder(
                torch.tensor(librosa.resample(decoded, orig_sr=24_000, target_sr=16_000))
            )
            return new_codes, decoded, f0_median(decoded, 24_000), emb

        # The identity control: every step, no manipulation. Its cosine is what
        # the codec round trip costs on its own, and every variant is read
        # against it rather than against 1.0.
        control_codes, _, control_f0, control_emb = through_pipeline(original)
        control_cosine = cosine(base_emb, control_emb)
        report[name] = {
            "reference_f0": base_f0,
            "control": {"identity_cosine": control_cosine, "f0": control_f0},
            "variants": {},
        }
        print(
            f"{name:<12} 参考 F0 {base_f0:6.1f}  "
            f"恒等对照余弦 {control_cosine:.4f}（这条流水线的地板）"
        )

        for emotion, move in EMOTIONS.items():
            semitones, capped = semitones_for(base_f0, move["hz"])
            moved = manipulate(
                original, 24_000, semitones, move["rate"], formants=not args.no_formant_correction
            )
            new_codes, _, variant_f0, new_emb = through_pipeline(moved)
            identity = cosine(base_emb, new_emb)
            fraction = identity / control_cosine if control_cosine > 0 else 0.0

            # Codebook 0 is the semantic one. It moving as much as the acoustic
            # codebooks would mean the manipulation changed what is being said,
            # not how -- which no amount of identity cosine would reveal.
            if move["rate"] == 1.0:
                widths = min(control_codes.shape[-1], new_codes.shape[-1])
                churn: Any = [
                    float((control_codes[i, :widths] != new_codes[i, :widths]).float().mean())
                    for i in range(new_codes.shape[0])
                ]
            else:
                # Time stretching slides every frame, so a frame-aligned
                # comparison reports ~100% for any rate change and means
                # nothing. Left null rather than printed as a finding.
                churn = None

            record = {
                "f0": variant_f0,
                "f0_shift": variant_f0 - base_f0,
                "identity_cosine": identity,
                "fraction_of_control": fraction,
                "semitones": semitones,
                "semitone_cap_bit": capped,
                "rate": move["rate"],
                "seconds": len(new_codes[0]) / 12.5,
                "codebook_churn": churn,
            }
            report[name]["variants"][emotion] = record
            key = f"{name}_{emotion}"
            if fraction < IDENTITY_FRACTION:
                # Noted, not refused: see the module docstring. The vector is
                # anchored, so this number describes the codes, not the voice
                # the model will produce.
                weak.append(key)
            # The identity vector is NOT rebuilt: the voice's own stays, and
            # only the codes carry the prosody. This is the whole fix.
            out[key] = {"ref_codes": new_codes.contiguous(), "spk_emb": entry["spk_emb"]}
            print(
                f"  {key:<20} {semitones:+5.2f} 半音{'(封顶)' if capped else '     '} "
                f"F0 {variant_f0:6.1f} ({record['f0_shift']:+6.1f})  "
                f"身份 {identity:.4f} = 对照的 {fraction:.0%}  "
                + (f"码本0 变动 {churn[0]:.0%}" if churn else "码本变动 n/a（时长变了）")
                + ("  (码带身份弱)" if fraction < IDENTITY_FRACTION else "")
            )
        out[f"{name}_neutral"] = {"ref_codes": entry["ref_codes"], "spk_emb": entry["spk_emb"]}

    if weak:
        print(
            f"\n{len(weak)} 个变体的码带身份不到对照的 {IDENTITY_FRACTION:.0%}——"
            "已记录，未拒绝（身份走 spk_emb，锚定未动）"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(args.output))
    print(f"\n写入 {args.output}：{len(out)} 个条目")
    return {
        "formant_correction": not args.no_formant_correction,
        "identity_fraction_required": IDENTITY_FRACTION,
        "max_semitones": MAX_SEMITONES,
        "codes_with_weak_identity": weak,
        "gate_not_applied_here": (
            "the criterion that decides is the clone score of GENERATED audio against the "
            "base voice's stored embedding; this screen can reject, not pass"
        ),
        "voices": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", type=Path, required=True, help="dir with voices*.pt")
    parser.add_argument("--codec", type=Path, required=True, help="mimi directory")
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="limit to these voice names")
    parser.add_argument(
        "--no-formant-correction",
        action="store_true",
        help="shift by resampling alone, which moves the vocal tract with the pitch; "
        "here to measure what the correction is worth, not to be used",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = build(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
