"""Does a batch come back in one voice, and at a human speaking rate.

The fourth meeting reported three symptoms from the cascade's local
synthesiser: the speaker changes between sentences, some clips stretch into a
squeal, and punctuation gets read aloud. The first one has a mechanical cause
-- ``VoxCPMSynthesiser`` was built with no prompt clip, and VoxCPM without a
prompt draws a speaker per call -- and this is the instrument that says whether
pointing it at a fixed reference actually fixed it.

Two numbers per arm, both per-utterance so two arms can be compared rather than
described:

* **Timbre spread.** CAM++ embeddings for every clip, cosine over every pair.
  One voice reads high and tight; a new speaker per call reads low and wide.
  The floor comes from resampling utterances rather than pairs -- the pairs are
  not independent, and a bootstrap over them would report a confidence the data
  does not have.
* **Speaking rate.** Characters of the requested text over the seconds of audio
  produced. A clip that stretches has a rate far below the arm's own centre,
  and one that races sits far above; both are the shape the "拖长气泡" report
  describes. Not a listening test -- it catches the gross ones only.

Clipping is measured by ``measure_clipping.py`` on the same directories and is
deliberately not repeated here.

    python scripts/measure_voice_consistency.py \
        --minimind-root ~/omni/minimind-o --device cuda \
        --arm edge=artifacts/tts_edge --arm voxcpm_noref=artifacts/tts_voxcpm_noref \
        --arm voxcpm_ref=artifacts/tts_voxcpm_ref \
        --report artifacts/voice-consistency-2026-08-15.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
# The repository root as well: run as a script, sys.path[0] is scripts/ rather
# than the working directory, so the sibling module is not importable by its
# package name -- which pytest hides, because it puts the root there itself.
sys.path.insert(0, str(_ROOT))

from scripts.measure_voice_clone import Embedder  # noqa: E402

# 判据，跑之前写死。
#
# 一条参考锁住音色：带参考那一臂，句与句之间的说话人余弦中位数。
FIXED_VOICE = 0.90
# 无参考臂要低到这个数以下，才算把「逐句换人声」这个病因坐实——两条都过，
# 才是「病因找对了并且修好了」，只过一条说明测的不是这件事。
SCATTERED_VOICE = 0.60
# 语速的人类区间（字/秒）。中文朗读常态 4–6，放宽到这个范围之外才算粗大缺陷。
RATE_FLOOR, RATE_CEILING = 2.0, 12.0


def load_arm(directory: Path) -> list[dict[str, Any]]:
    """The clips of one arm, with the text each was asked to say."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{directory} 没有 manifest.json，认不出每条要说的是什么")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for sample in manifest.get("samples", []):
        path = directory / f"{sample['id']}.wav"
        if not sample.get("audio_path") or not path.is_file():
            rows.append({**sample, "path": None})
            continue
        rows.append({**sample, "path": path})
    if not rows:
        raise SystemExit(f"{directory} 的 manifest 里没有样本")
    return rows


def embed_all(rows: list[dict[str, Any]], embedder: Embedder) -> list[Any]:
    import soundfile as sf
    import torch
    import torchaudio

    resamplers: dict[int, Any] = {}
    embeddings = []
    for row in rows:
        if row["path"] is None:
            continue
        waveform, rate = sf.read(str(row["path"]), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        audio = torch.from_numpy(waveform)
        if rate != 16_000:
            if rate not in resamplers:
                resamplers[rate] = torchaudio.transforms.Resample(rate, 16_000)
            audio = resamplers[rate](audio)
        embeddings.append(embedder(audio))
    return embeddings


def spread(embeddings: list[Any], draws: int = 1000, seed: int = 20260815) -> dict[str, Any]:
    """How alike the clips of one arm sound, and how firm that number is.

    Resampling utterances rather than pairs: with n clips there are n(n-1)/2
    cosines and they share terms, so treating them as independent samples would
    shrink the floor by roughly the square root of n for free.
    """
    import numpy
    import torch

    if len(embeddings) < 2:
        return {"n": len(embeddings), "pairs": 0}
    # One matrix product rather than n(n-1)/2 calls: at 160 clips a draw is
    # 12,720 cosines and a thousand draws is twelve million, which is minutes of
    # Python for a number that is one BLAS call.
    stacked = torch.stack([embedding.detach().float().reshape(-1) for embedding in embeddings])
    stacked = stacked / stacked.norm(dim=1, keepdim=True).clamp(min=1e-12)
    similarity = (stacked @ stacked.T).cpu().numpy()
    count = similarity.shape[0]
    upper = numpy.triu_indices(count, 1)
    pairs = similarity[upper]

    generator = numpy.random.default_rng(seed)
    means = []
    for _ in range(draws):
        picked = generator.integers(0, count, count)
        block = similarity[numpy.ix_(picked, picked)]
        # A clip resampled twice would otherwise contribute its own 1.0, which
        # is not a pair of utterances.
        distinct = picked[:, None] != picked[None, :]
        if distinct.any():
            means.append(float(block[distinct].mean()))
    return {
        "n": count,
        "pairs": int(pairs.size),
        "mean": float(pairs.mean()),
        "median": float(numpy.median(pairs)),
        "minimum": float(pairs.min()),
        "p05": float(numpy.quantile(pairs, 0.05)),
        "noise_floor": statistics.stdev(means) if len(means) > 1 else None,
    }


def speaking_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Characters asked for over seconds produced, and the clips that are nowhere near."""
    rates = []
    outliers = []
    for row in rows:
        seconds = row.get("audio_seconds") or 0.0
        characters = len(row.get("reference_text", ""))
        if seconds <= 0 or characters == 0:
            continue
        rate = characters / seconds
        rates.append(rate)
        if rate < RATE_FLOOR or rate > RATE_CEILING:
            outliers.append({"id": row["id"], "chars_per_second": rate, "seconds": seconds})
    silent = [row["id"] for row in rows if row["path"] is None]
    return {
        "n": len(rates),
        "median": statistics.median(rates) if rates else None,
        "minimum": min(rates) if rates else None,
        "maximum": max(rates) if rates else None,
        "outliers": sorted(outliers, key=lambda item: item["chars_per_second"])[:10],
        "outlier_rate": len(outliers) / len(rates) if rates else None,
        "silent": silent,
    }


# The names of the marks, which a synthesiser reading punctuation aloud says
# instead of pausing. Compared against the requested text rather than flagged
# outright: a reply may genuinely be about commas.
PUNCTUATION_WORDS = ("逗号", "句号", "问号", "感叹号", "顿号", "冒号", "分号", "省略号", "破折号")


def read_punctuation(rows: list[dict[str, Any]], transcripts: dict[str, str]) -> dict[str, Any]:
    """Clips where a judge heard the name of a mark that nobody asked for.

    The third symptom from the meeting. It needs a transcript, so it is
    reported only for arms that have one -- and an arm without one says so
    rather than reporting zero, which would read as "checked, clean".
    """
    caught = []
    for row in rows:
        heard = transcripts.get(row["id"])
        if heard is None:
            continue
        wanted = row.get("reference_text", "")
        spoken = [word for word in PUNCTUATION_WORDS if word in heard and word not in wanted]
        if spoken:
            caught.append({"id": row["id"], "words": spoken, "transcript": heard[:60]})
    return {"n": len(transcripts), "clips": caught, "rate": len(caught) / max(1, len(transcripts))}


def verdict(arms: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Read against the lines above, on the arm names the fix is about."""
    found = {}
    reference = arms.get("voxcpm_ref", {}).get("timbre", {}).get("median")
    unprompted = arms.get("voxcpm_noref", {}).get("timbre", {}).get("median")
    if reference is None or unprompted is None:
        return {"音色": "缺臂，判不了（需要 voxcpm_ref 与 voxcpm_noref 两臂）"}
    if reference >= FIXED_VOICE and unprompted < SCATTERED_VOICE:
        found["音色"] = (
            f"病因坐实并且修好了：无参考 {unprompted:.4f} < {SCATTERED_VOICE}，"
            f"带参考 {reference:.4f} ≥ {FIXED_VOICE}"
        )
    elif reference >= FIXED_VOICE:
        found["音色"] = (
            f"带参考那臂音色是固定的（{reference:.4f}），但无参考臂 {unprompted:.4f} 并不散——"
            "「逐句换人声」这个归因在这批句子上没坐实"
        )
    else:
        found["音色"] = f"没修好：带参考 {reference:.4f} < {FIXED_VOICE}"
    worse = [
        name
        for name, arm in arms.items()
        if name.startswith("voxcpm") and (arm["rate"]["outlier_rate"] or 0) > 0
    ]
    found["语速"] = (
        "没有落在人类区间外的片段" if not worse else f"语速离群出现在：{', '.join(worse)}"
    )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimind-root", required=True, type=Path)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="名字=目录",
        help="一臂一个，名字进报告。voxcpm_ref / voxcpm_noref 这两个名字有判据",
    )
    parser.add_argument(
        "--transcripts",
        action="append",
        default=[],
        metavar="名字=打分行",
        help="可选，一臂一个：独立判官转写的 jsonl，用来查「念标点」那个症状",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    heard: dict[str, dict[str, str]] = {}
    for entry in args.transcripts:
        name, _, path = entry.partition("=")
        heard[name] = {
            row["id"]: row.get("transcript") or ""
            for row in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())
        }

    embedder = Embedder(args.minimind_root.expanduser().resolve(), args.device)
    arms: dict[str, dict[str, Any]] = {}
    for entry in args.arm:
        if "=" not in entry:
            raise SystemExit(f"--arm 要写成 名字=目录，收到 {entry!r}")
        name, _, directory = entry.partition("=")
        rows = load_arm(Path(directory))
        print(f"{name}: {len(rows)} 条")
        arms[name] = {
            "directory": directory,
            "timbre": spread(embed_all(rows, embedder)),
            "rate": speaking_rate(rows),
            # None rather than an empty result: "no transcript" and "checked,
            # nothing found" are different answers to the same question.
            "read_punctuation": read_punctuation(rows, heard[name]) if name in heard else None,
        }

    report = {
        "arms": arms,
        "thresholds": {
            "fixed_voice": FIXED_VOICE,
            "scattered_voice": SCATTERED_VOICE,
            "rate_floor": RATE_FLOOR,
            "rate_ceiling": RATE_CEILING,
        },
        "verdicts": verdict(arms),
        "caveat": (
            "CAM++ 量的是说话人身份，不是好不好听；语速离群只抓粗大缺陷。"
            "「达不达得到 edge 那一档」要人耳，这里只给数"
        ),
    }
    for name, arm in arms.items():
        timbre, rate = arm["timbre"], arm["rate"]
        print(
            f"{name}: 句间余弦中位 {timbre.get('median', float('nan')):.4f}"
            f"（均值 {timbre.get('mean', float('nan')):.4f}"
            f" ± {timbre.get('noise_floor') or float('nan'):.4f}，最低 "
            f"{timbre.get('minimum', float('nan')):.4f}），"
            f"语速中位 {rate['median']:.2f} 字/秒，离群 {len(rate['outliers'])}"
            f"，没出声 {len(rate['silent'])}"
        )
    for name, line in report["verdicts"].items():
        print(f"判据 {name}：{line}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
