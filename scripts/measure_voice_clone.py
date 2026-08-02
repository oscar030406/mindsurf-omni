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

from mindsurf_omni.evaluation.metrics import (  # noqa: E402
    assess,
    bootstrap_noise_floor,
    cluster_bootstrap_noise_floor,
    resolvable_effect,
)

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


SPLITS = ("seen", "unseen", "external")


def load_voices(root: Path, extra: tuple[Path, ...] = ()) -> dict[str, dict[str, Any]]:
    """Every voice with the split it belongs to, seen first.

    ``extra`` takes packs built outside upstream's speaker directory -- the
    harvested emotional references are the first of them. They land in their
    own split rather than joining ``unseen``: the certified bands were measured
    on upstream's twelve voices, and a pack of three corpus speakers averaged
    into that number would move it without anyone choosing to.
    """
    import torch

    packs = [(root / "model" / "speaker" / SEEN_PACK, "seen")]
    packs.append((root / "model" / "speaker" / UNSEEN_PACK, "unseen"))
    # A directory is the shape harvest_emotion_voices.py writes, a file is what
    # someone types when they have one pack and no directory to put it in.
    packs += [(path / SEEN_PACK if path.is_dir() else path, "external") for path in extra]

    voices: dict[str, dict[str, Any]] = {}
    for path, split in packs:
        if not path.is_file():
            if split == "external":
                raise SystemExit(f"--voice-pack {path} 不存在")
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


def identify(
    embedding: Any, voice: str, voices: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Which of the pack's voices this clip sounds most like, and by how much.

    The similarity to a clip's own voice is a number on a scale nobody set: a
    voice whose codec ceiling is 0.79 and one whose ceiling is 0.90 cannot be
    held to the same line, and a band drawn at "the tier mean minus the mean's
    own noise" rejects half the tier by construction -- see section 15 of
    docs/experiments/2026-07-30-emotion-harvest-gate.md, which is what this
    exists to replace.

    Asking instead which voice the clip is closest to needs no chosen
    threshold: the boundary comes from the other voices in the pack. The
    margin over the nearest impostor is what is left of the continuous
    quantity, and unlike the raw similarity it is comparable across voices.

    Candidates are upstream's twelve. An external pack joins ``voices`` for
    scoring but must not join the line-up -- adding candidates makes the task
    harder without anyone choosing that -- so a clip conditioned on an external
    voice has no identification to report.
    """
    candidates = {name: entry for name, entry in voices.items() if entry["split"] != "external"}
    if voice not in candidates:
        return None
    scores = {name: cosine(embedding, entry["spk_emb"]) for name, entry in candidates.items()}
    nearest = max(scores, key=lambda name: scores[name])
    others = [score for name, score in scores.items() if name != voice]
    impostor = max(others, default=float("-inf"))
    return {
        "nearest": nearest,
        "hit": nearest == voice,
        "margin": scores[voice] - impostor,
        "candidates": len(candidates),
    }


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
    voices = load_voices(root, tuple(args.voice_pack or ()))
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
        embedding = embedder(resample(decoded.cpu()))
        similarity = cosine(embedding, voice["spk_emb"])
        rows.append(
            {
                "voice": name,
                "split": voice["split"],
                "reference_seconds": round(decoded.shape[-1] / 24000, 2),
                "ceiling": similarity,
            }
        )
        # Whether a perfect cloner would even be identified. If the round trip
        # of a voice's own reference lands nearer some other voice, then a low
        # hit rate on generated audio is the line-up's property and not the
        # model's, and no verdict may be read from it.
        recognised = identify(embedding, name, voices)
        if recognised is not None:
            rows[-1]["ceiling_nearest"] = recognised["nearest"]
            rows[-1]["ceiling_hit"] = recognised["hit"]
            rows[-1]["ceiling_margin"] = recognised["margin"]
        seconds = rows[-1]["reference_seconds"]
        print(f"  {name:<10} {voice['split']:<6} ref {seconds:>5.2f}s  上限 {similarity:.4f}")

    values = [row["ceiling"] for row in rows]
    print(
        f"\n上限 均值 {statistics.mean(values):.4f}  最低 {min(values):.4f}（"
        f"{min(rows, key=lambda row: row['ceiling'])['voice']}）  最高 {max(values):.4f}"
    )
    print("  这是「完美克隆」能拿到的分，不是 1.0——差额是 Mimi 8 码本 12.5 Hz 对说话人身份的损耗")

    payload = {"mode": "ceiling", "voices": rows, "mean": statistics.mean(values)}
    judged = [row for row in rows if "ceiling_hit" in row]
    if judged:
        hits = sum(1 for row in judged if row["ceiling_hit"])
        payload["identification_ceiling"] = {"n": len(judged), "hits": hits}
        print(f"\n上限臂的 rank-1 认人 {hits}/{len(judged)}")
        for row in judged:
            if not row["ceiling_hit"]:
                nearest, margin = row["ceiling_nearest"], row["ceiling_margin"]
                print(f"  连完美克隆都认错 {row['voice']} → {nearest}（裕度 {margin:+.4f}）")
        if hits < len(judged):
            print("  **认错的那些音色，生成侧的认人率读不出模型的好坏**——线阵本身分不开它们")
    return payload


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    """Score generated clips against the voice each was conditioned on.

    Two generators write these directories and they record the conditioning
    differently. evaluate_talker.py names clips by text id and stamps the voice
    in the manifest; upstream's eval_omni.py --mode 3 puts it in the filename
    and writes no manifest. The manifest wins when present -- it is the
    authoritative record and a filename is a convention -- and the pattern is
    the fallback. A clip whose conditioning cannot be recovered from the
    artifact is a clip nobody can score twice.
    """
    import soundfile as sf
    import torch
    import torchaudio

    root = args.minimind_root.expanduser().resolve()
    voices = load_voices(root, tuple(args.voice_pack or ()))
    embedder = Embedder(root, args.device)
    pattern = re.compile(args.name_pattern)

    by_voice: dict[str, list[float]] = {}
    # Per-clip rows as well as aggregates: two arms scored on the same voices
    # and the same texts can be paired clip by clip, and a paired verdict is
    # the only kind this project lets decide anything. Aggregates alone would
    # force a comparison of means with no threshold.
    samples: list[dict[str, Any]] = []
    unmatched = []
    resamplers: dict[int, Any] = {}
    for directory in args.score:
        manifest_voice = None
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            stamped = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_voice = (stamped.get("generated_by") or {}).get("voice")
            if manifest_voice and manifest_voice not in voices:
                raise SystemExit(
                    f"{directory} 的 manifest 记的音色 {manifest_voice!r} 不在音色包里"
                )
            if manifest_voice:
                print(f"{directory.name}: 音色取自 manifest：{manifest_voice}")
            else:
                print(f"{directory.name}: manifest 里 voice 是空的——**无条件生成**，量不出克隆")
        for path in sorted(directory.glob("*.wav")):
            if manifest_voice:
                name = manifest_voice
            else:
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
            embedding = embedder(audio)
            similarity = cosine(embedding, voices[name]["spk_emb"])
            by_voice.setdefault(name, []).append(similarity)
            row = {
                "id": f"{name}/{path.stem}",
                "voice": name,
                "split": voices[name]["split"],
                "similarity": similarity,
            }
            recognised = identify(embedding, name, voices)
            if recognised is not None:
                row["nearest"] = recognised["nearest"]
                row["hit"] = recognised["hit"]
                row["margin"] = recognised["margin"]
            samples.append(row)

    if not by_voice:
        raise SystemExit(
            f"{args.score} 里没有可打分的 wav。manifest 没记音色时按 "
            f"{args.name_pattern!r} 匹配文件名；看到的前几个: {unmatched[:5]}"
        )

    rows = []
    for name, values in sorted(
        by_voice.items(), key=lambda item: (SPLITS.index(voices[item[0]]["split"]), item[0])
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

    payload: dict[str, Any] = {
        "mode": "score",
        "voices": rows,
        "samples": samples,
        "unmatched": unmatched,
    }

    identified = [sample for sample in samples if "hit" in sample]
    if identified:
        hits = sum(1 for sample in identified if sample["hit"])
        # Clustered by voice, because that is the unit that varies: a voice is
        # almost all right or almost all wrong, so twelve voices carry twelve
        # units of information and not two hundred and forty. Resampling rows
        # would report a floor the data does not support.
        by_name: dict[str, list[float]] = {}
        for sample in identified:
            by_name.setdefault(sample["voice"], []).append(float(sample["hit"]))
        floor = cluster_bootstrap_noise_floor(list(by_name.values()))
        resolvable = resolvable_effect(floor)
        eligible = resolvable <= args.identification_effect
        payload["identification"] = {
            "n": len(identified),
            "voices": len(by_name),
            "hits": hits,
            "rank1": hits / len(identified),
            "noise_floor": floor,
            "resolves": resolvable,
            "effect_of_interest": args.identification_effect,
            "gating_eligible": eligible,
            "margin_mean": statistics.mean(sample["margin"] for sample in identified),
            "misses": [
                {"id": sample["id"], "nearest": sample["nearest"], "margin": sample["margin"]}
                for sample in identified
                if not sample["hit"]
            ],
        }
        summary = payload["identification"]
        mark = (
            "有门控资格"
            if eligible
            else f"仅报告（按音色重抽只分辨得了 {resolvable:.4f}，"
            f"关心的是 {args.identification_effect:.4f}）"
        )
        print(
            f"\nrank-1 认人 {hits}/{len(identified)} = {summary['rank1']:.4f} "
            f"± {floor:.4f}（按音色重抽，{len(by_name)} 个音色）  {mark}"
        )
        print(f"  裕度均值 {summary['margin_mean']:.4f}")
        for miss in summary["misses"][:5]:
            print(f"  认错 {miss['id']} → {miss['nearest']}（裕度 {miss['margin']:+.4f}）")
    for split in SPLITS:
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
    parser.add_argument(
        "--score",
        type=Path,
        nargs="+",
        help="directories of generated wavs. One per voice is the normal shape, "
        "because evaluate_talker.py takes a single --voice per run and its clips "
        "are named by text id, so two voices in one directory would collide",
    )
    parser.add_argument(
        "--name-pattern",
        default=r"clone-(?P<voice>[a-z_]+)-",
        help="regex with a 'voice' group, matched against each wav's stem. The "
        "default follows eval_omni.py's clone-<voice>-<index> naming",
    )
    parser.add_argument(
        "--voice-pack",
        type=Path,
        nargs="+",
        help="voices.pt built outside upstream's model/speaker -- a directory "
        "or the file itself. Scored in its own 'external' split, because the "
        "seen and unseen bands were certified on upstream's twelve voices",
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
        "--identification-effect",
        type=float,
        default=0.10,
        help="the rank-1 difference the identification axis has to resolve to be "
        "allowed to gate. Written down before the first cluster-bootstrap run, in "
        "the fourth-round queue of docs/ACTION_PLAN.md; do not tune it to a result",
    )
    parser.add_argument(
        "--ceiling-report",
        type=Path,
        default=Path("artifacts/voice-clone-ceiling.json"),
        help="a --ceiling run's JSON, used to normalise scores against the codec limit",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.ceiling and not args.score:
        raise SystemExit("给 --ceiling 验仪器，或给 --score <目录…> 打分")

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
