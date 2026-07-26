"""Speaker similarity for in-context voice cloning, and the ceiling it is read against.

KPI-3 asks for voice cloning. The architecture does it by conditioning: a
voice is a reference Mimi code strip plus a 192-dimensional CAM++ vector,
placed in the audio buffer, and the Talker's weights never change. So
measuring it means asking whether the generated speech carries the reference
speaker's identity, which is a cosine between CAM++ embeddings.

The number that comes out cannot be read against 1.0, and that is the point
of ``--ceiling``. Upstream's stored ``spk_emb`` was extracted from the
original reference recording, while anything the model generates has to come
back out through eight Mimi codebooks at 12.5 Hz. Decoding the stored
``ref_codes`` and re-extracting gives what a *perfect* cloner would score --
measured at 0.79 to 0.90 depending on the voice. Upstream's report gives
0.6472 seen and 0.5654 unseen with no ceiling stated, so those read as
"roughly 60% of the way there" when they are closer to 77% of what the codec
allows.

This mirrors ``measure_naturalness.py --validate``: prove the instrument, then
let it judge. An instrument whose ceiling is unmeasured is the same failure
this project has already paid for twice -- a CER floor that was three quarters
judge behaviour, and a repetition score that rewarded silence.

The pipeline is upstream's own, copied from ``webui/web_demo.py`` rather than
reinvented: 16 kHz mel (n_fft 512, win 400, hop 160, 80 mels, 20-7600 Hz,
slaney), log, mean-subtracted over time, then CAM++ at embedding size 192.
Upstream ships the *generation* side of its clone evaluation
(``eval_omni.py --mode 3``) but not the scorer, so this is a reconstruction of
it -- if their table used a different front-end the absolute values shift, and
the ceiling shifts with them.

    python scripts/measure_voice_clone.py --ceiling --minimind-root ~/omni/minimind-o
    python scripts/measure_voice_clone.py --score artifacts/clone_eval \
        --minimind-root ~/omni/minimind-o
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import assess, bootstrap_noise_floor  # noqa: E402

# Upstream's front-end, value for value. A different mel window or a missing
# mean subtraction would still produce plausible cosines, which is exactly why
# these are stated rather than defaulted.
MEL = {
    "sample_rate": 16000,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "n_mels": 80,
    "f_min": 20,
    "f_max": 7600,
    "norm": "slaney",
    "mel_scale": "slaney",
}
CAMPPLUS = {
    "feat_dim": 80,
    "embedding_size": 192,
    "growth_rate": 32,
    "bn_size": 4,
    "init_channels": 128,
    "config_str": "batchnorm-relu",
    "memory_efficient": True,
}
SEEN_PACK, UNSEEN_PACK = "voices.pt", "voices_unseen.pt"


def load_voices(root: Path) -> dict[str, dict[str, Any]]:
    """Every voice with the split it belongs to, seen first."""
    import torch

    voices: dict[str, dict[str, Any]] = {}
    for pack, split in ((SEEN_PACK, "seen"), (UNSEEN_PACK, "unseen")):
        path = root / "model" / "speaker" / pack
        if not path.is_file():
            continue
        for name, entry in torch.load(str(path), map_location="cpu").items():
            voices[name] = {
                "split": split,
                "ref_codes": entry["ref_codes"],
                "spk_emb": entry["spk_emb"].float(),
            }
    if not voices:
        raise SystemExit(f"no voice packs under {root / 'model' / 'speaker'}")
    return voices


class Embedder:
    """CAM++ on 16 kHz audio, built once and reused."""

    def __init__(self, root: Path, device: str) -> None:
        import torch
        import torchaudio
        from modelscope.models.audio.sv.DTDNN import CAMPPlus  # type: ignore[import-not-found]

        self.torch = torch
        self.device = device
        model = CAMPPlus(**CAMPPLUS)
        state = torch.load(
            str(root / "model" / "campplus" / "campplus_cn_common.pt"), map_location="cpu"
        )
        model.load_state_dict({key: value.float() for key, value in state.items()})
        self.model = model.eval().to(device)
        self.mel = torchaudio.transforms.MelSpectrogram(**MEL).to(device)

    def __call__(self, waveform_16k: Any) -> Any:
        with self.torch.no_grad():
            mel = self.mel(waveform_16k.unsqueeze(0).to(self.device))
            feature = mel.clamp(min=1e-10).log().transpose(1, 2)
            feature = feature - feature.mean(dim=1, keepdim=True)
            return self.model(feature).squeeze(0).cpu()


def cosine(left: Any, right: Any) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left, right, dim=0))


def run_ceiling(args: argparse.Namespace) -> dict[str, Any]:
    """What a perfect cloner would score, which is not 1.0.

    Decodes each stored reference strip back to audio and re-embeds it. The
    gap from 1.0 is what eight Mimi codebooks at 12.5 Hz cost a speaker
    embedding, and every generated clip pays it too.
    """
    import torch
    import torchaudio
    from transformers import MimiModel

    root = args.minimind_root.expanduser().resolve()
    voices = load_voices(root)
    embedder = Embedder(root, args.device)
    mimi = MimiModel.from_pretrained(str(root / "model" / "mimi")).eval().float().to(args.device)
    resample = torchaudio.transforms.Resample(24000, MEL["sample_rate"])

    rows = []
    for name, voice in sorted(
        voices.items(), key=lambda item: (item[1]["split"] != "seen", item[0])
    ):
        with torch.no_grad():
            decoded = mimi.decode(
                voice["ref_codes"].unsqueeze(0).to(args.device)
            ).audio_values.squeeze()
        similarity = cosine(embedder(resample(decoded.cpu())), voice["spk_emb"])
        rows.append(
            {
                "voice": name,
                "split": voice["split"],
                "reference_seconds": round(decoded.shape[-1] / 24000, 2),
                "ceiling": similarity,
            }
        )
        seconds = rows[-1]["reference_seconds"]
        print(f"  {name:<10} {voice['split']:<6} ref {seconds:>5.2f}s  上限 {similarity:.4f}")

    values = [row["ceiling"] for row in rows]
    print(
        f"\n上限 均值 {statistics.mean(values):.4f}  最低 {min(values):.4f}（"
        f"{min(rows, key=lambda row: row['ceiling'])['voice']}）  最高 {max(values):.4f}"
    )
    print("  这是「完美克隆」能拿到的分，不是 1.0——差额是 Mimi 8 码本 12.5 Hz 对说话人身份的损耗")
    return {"mode": "ceiling", "voices": rows, "mean": statistics.mean(values)}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    """Score generated clips against the voice each was conditioned on.

    Filenames carry the voice, because a clip whose conditioning cannot be
    recovered from the artifact is a clip that cannot be scored twice -- the
    same reason every other harness here writes its provenance down.
    """
    import soundfile as sf
    import torch
    import torchaudio

    root = args.minimind_root.expanduser().resolve()
    voices = load_voices(root)
    embedder = Embedder(root, args.device)
    pattern = re.compile(args.name_pattern)

    by_voice: dict[str, list[float]] = {}
    unmatched = []
    resamplers: dict[int, Any] = {}
    for path in sorted(args.score.glob("*.wav")):
        match = pattern.search(path.stem)
        if not match or match.group("voice") not in voices:
            unmatched.append(path.name)
            continue
        name = match.group("voice")
        waveform, rate = sf.read(str(path))
        audio = torch.tensor(waveform, dtype=torch.float32)
        if audio.ndim > 1:
            audio = audio.mean(dim=1)
        if rate != MEL["sample_rate"]:
            if rate not in resamplers:
                resamplers[rate] = torchaudio.transforms.Resample(rate, MEL["sample_rate"])
            audio = resamplers[rate](audio)
        by_voice.setdefault(name, []).append(cosine(embedder(audio), voices[name]["spk_emb"]))

    if not by_voice:
        raise SystemExit(
            f"{args.score} 里没有能匹配 {args.name_pattern!r} 且在音色包里的 wav；"
            f"看到的前几个: {unmatched[:5]}"
        )

    rows = []
    for name, values in sorted(
        by_voice.items(), key=lambda item: (voices[item[0]]["split"] != "seen", item[0])
    ):
        rows.append(
            {
                "voice": name,
                "split": voices[name]["split"],
                "n": len(values),
                "similarity": statistics.mean(values),
                "noise_floor": bootstrap_noise_floor(values) if len(values) > 1 else float("inf"),
            }
        )
        split, mean = rows[-1]["split"], rows[-1]["similarity"]
        print(f"  {name:<10} {split:<6} n={len(values):<3} 相似度 {mean:.4f}")

    payload: dict[str, Any] = {"mode": "score", "voices": rows, "unmatched": unmatched}
    for split in ("seen", "unseen"):
        values = [
            value for row in rows if row["split"] == split for value in by_voice[row["voice"]]
        ]
        if not values:
            continue
        measurement = assess(f"clone_{split}", values, effect_of_interest=args.effect)
        payload[split] = {
            "value": measurement.value,
            "noise_floor": measurement.noise_floor,
            "n": measurement.sample_size,
            "gating_eligible": measurement.gating_eligible,
            "note": measurement.note,
        }
        mark = "有门控资格" if measurement.gating_eligible else f"仅报告（{measurement.note}）"
        print(f"\n{split}: {measurement}  {mark}")

    if args.ceiling_report and args.ceiling_report.is_file():
        stored = json.loads(args.ceiling_report.read_text(encoding="utf-8"))
        ceilings = {row["voice"]: row["ceiling"] for row in stored.get("voices", [])}
        print("\n对上限归一（相似度 ÷ 该音色的编解码上限）:")
        for row in rows:
            ceiling = ceilings.get(row["voice"])
            if ceiling:
                row["ceiling"] = ceiling
                row["normalised"] = row["similarity"] / ceiling
                name, got, norm = row["voice"], row["similarity"], row["normalised"]
                print(f"  {name:<10} {got:.4f} / {ceiling:.4f} = {norm:.3f}")
        print("  **归一化会改变排名**：原始分低的音色可能只是它的上限低")
    elif args.ceiling_report:
        print(
            f"\n（没有 {args.ceiling_report}，先跑 --ceiling，否则这些分只能对 1.0 读，那是错的）"
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument(
        "--ceiling",
        action="store_true",
        help="measure what a perfect cloner would score, by decoding the stored "
        "reference strips and re-embedding them. Run this before trusting any score",
    )
    parser.add_argument("--score", type=Path, help="directory of generated wavs")
    parser.add_argument(
        "--name-pattern",
        default=r"clone-(?P<voice>[a-z_]+)-",
        help="regex with a 'voice' group, matched against each wav's stem. The "
        "default follows eval_omni.py's clone-<voice>-<index> naming",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="cores on CPU. Not all of them: this runs beside training more often than not",
    )
    parser.add_argument("--effect", type=float, default=0.05)
    parser.add_argument(
        "--ceiling-report",
        type=Path,
        default=Path("artifacts/voice-clone-ceiling.json"),
        help="a --ceiling run's JSON, used to normalise scores against the codec limit",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.ceiling and not args.score:
        raise SystemExit("给 --ceiling 验仪器，或给 --score <目录> 打分")

    if args.device == "cpu" and args.cpu_threads > 0:
        import torch

        torch.set_num_threads(args.cpu_threads)

    payload = run_ceiling(args) if args.ceiling else run_score(args)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
