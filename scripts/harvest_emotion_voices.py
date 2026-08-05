"""Emotional references harvested from real audio instead of manufactured from it.

The manufactured route is dead. Signal-processing a neutral reference into an
emotional one destroys the clip before the model ever sees it: 15 of 20 variants
land above CER 0.6, and the damage is the phase vocoder rather than the pitch
shift, so formant correction, which works, does not reach it.

This takes the other key. Every corpus row carries a speaker vector beside its
reference clip, and the corpus repeats speakers heavily: 80% of rows have a
nearest neighbour above 0.90 cosine, against a median of 0.79 between random
rows. So one speaker's clips already span a range of pitch and rate, recorded
that way, and references in different prosodies exist on disk. Nothing is
processed. The clip that is calmer is simply a different clip.

Three things are checked rather than assumed, because each is a way this could
look like it worked and not have:

* **Same speaker.** Clusters form above 0.93, well past the corpus's own 99th
  percentile of 0.9172, and the pair's own cosine is reported. The manufactured
  variants scored 0.32 to 0.64 here; a codec round trip alone scores 0.95.
* **Still intelligible.** The clips go through the same judge every gating
  comparison here uses. This is the check that killed the manufactured pack and
  it costs one CPU pass.
* **The right language.** A reference conditions voice and prosody rather than
  content, but an English reference driving Chinese generation is an untested
  mismatch, so clusters whose clips do not transcribe as Chinese are skipped.

And one thing about *where* to read, which the first version got wrong twice.

Reading from the start of the file returned four English clusters, and the first
explanation -- that the corpus is mostly English -- was a sampling artifact.
Sampling every fifth of the 102 row groups shows the English ones are scattered,
not blocked: groups 0, 25, 60, 65 and 85 come back at a 0.00 Han share while
the rest sit between 0.85 and 0.97. The head of the file happens to be English,
which is the only reason the first run saw nothing else.

The same scan turned up something worth knowing independently: only 73.2% of
rows carry a speaker vector at all, and groups 60 through 80 carry none. With
the dataset's own 50% dropout of reference codes on top, A2A saw speaker
conditioning on roughly a third of its samples.

So ``--row-groups`` is not a convenience. Reading the head of a structured file
is the bug it fixes, and the groups that are both Chinese and conditioned are
5, 10, 15, 20, 30 through 55, and 90 onward.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# Above the corpus's 99th percentile of pairwise similarity (0.9172), so a
# cluster has to be tighter than chance rather than merely typical.
SAME_SPEAKER = 0.93
# A reference this short conditions on almost nothing once the model right-
# aligns it into the prompt.
MINIMUM_FRAMES = 60


def is_chinese(text: str, share: float = 0.5) -> bool:
    """Whether a transcript is mostly Han characters.

    Not a language identifier -- a threshold on what the judge returned. An
    English clip comes back as ASCII words and fails this by a wide margin,
    which is the only distinction that has to be reliable.
    """
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return False
    han = sum(1 for c in stripped if "一" <= c <= "鿿")
    return han / len(stripped) >= share


def unpack(flat: list[int]) -> Any:
    """Corpus reference codes are stored frame-major: eight values per frame.

    Taken from omni_dataset.py rather than guessed; the other reading produces
    a plausible tensor of noise.
    """
    import torch

    frames = len(flat) // 8
    return torch.tensor(
        [[flat[i * 8 + j] for i in range(frames)] for j in range(8)], dtype=torch.long
    )


def median_f0(wave: Any, rate: int = 24_000) -> float:
    import librosa
    import numpy

    f0 = librosa.yin(wave, fmin=60, fmax=500, sr=rate)
    voiced = f0[(f0 > 65) & (f0 < 480)]
    return float(numpy.median(voiced)) if len(voiced) >= 10 else float("nan")


def cluster(bank: Any, minimum: int) -> list[list[int]]:
    """Greedy speaker clusters, largest first."""
    similarity = bank @ bank.T
    similarity.fill_diagonal_(-1)
    used: set[int] = set()
    found: list[list[int]] = []
    for seed in similarity.max(dim=1).values.argsort(descending=True).tolist():
        if seed in used:
            continue
        members = [
            index
            for index in (similarity[seed] > SAME_SPEAKER).nonzero().flatten().tolist()
            if index not in used
        ]
        if len(members) >= minimum:
            used.update([seed, *members])
            found.append([seed, *members])
    found.sort(key=len, reverse=True)
    return found


def harvest(args: argparse.Namespace) -> dict[str, Any]:
    import librosa
    import pyarrow.parquet as pq
    import soundfile
    import torch
    from transformers import MimiModel

    from mindsurf_omni.service.asr import SenseVoiceRecogniser  # noqa: F401  (import check)

    codec = MimiModel.from_pretrained(str(args.codec)).eval()
    handle = pq.ParquetFile(str(args.parquet))

    wanted = args.row_groups or list(range(handle.num_row_groups))
    rows: list[tuple[list[int], Any]] = []
    for group in wanted:
        table = handle.read_row_group(group, columns=["ref_audios", "spk_emb"])
        for ref, emb in zip(
            table.column("ref_audios").to_pylist(),
            table.column("spk_emb").to_pylist(),
            strict=False,
        ):
            if ref and emb and len(emb) == 192 and len(ref) >= 8 * MINIMUM_FRAMES:
                rows.append((ref, torch.tensor(emb, dtype=torch.float32)))
        if len(rows) >= args.sample:
            break
    rows = rows[: args.sample]
    print(f"取样 {len(rows)} 行（行组 {wanted[:4]}{'…' if len(wanted) > 4 else ''}）", flush=True)

    bank = torch.stack([emb / emb.norm() for _, emb in rows])
    clusters = cluster(bank, args.minimum_clips)
    print(f"说话人簇（余弦 > {SAME_SPEAKER}）：{len(clusters)} 个", flush=True)

    from funasr import AutoModel

    judge = AutoModel(model="paraformer-zh", disable_update=True)
    args.output.mkdir(parents=True, exist_ok=True)

    pack: dict[str, Any] = {}
    report: list[dict[str, Any]] = []
    for number, members in enumerate(clusters[: args.clusters]):
        measured = []
        for index in members[: args.clips_per_cluster]:
            codes = unpack(rows[index][0])
            with torch.no_grad():
                wave = codec.decode(codes.unsqueeze(0))[0][0, 0].numpy()
            pitch = median_f0(wave)
            if pitch == pitch:
                measured.append((pitch, index, codes, wave))
        if len(measured) < 4:
            continue
        measured.sort()
        calm, lively = measured[0], measured[-1]

        transcripts = {}
        for tag, (_, _, _, wave) in (("calm", calm), ("lively", lively)):
            sixteen = librosa.resample(wave, orig_sr=24_000, target_sr=16_000)
            path = args.output / f"speaker{number}_{tag}.wav"
            soundfile.write(path, sixteen, 16_000)
            transcripts[tag] = judge.generate(input=str(path))[0]["text"]

        chinese = all(is_chinese(text) for text in transcripts.values())
        pair = float(bank[calm[1]] @ bank[lively[1]])
        entry = {
            "speaker": number,
            "clips": len(measured),
            "f0_calm": calm[0],
            "f0_lively": lively[0],
            "f0_spread": lively[0] - calm[0],
            "pair_cosine": pair,
            "chinese": chinese,
            "transcripts": transcripts,
        }
        report.append(entry)
        mark = "" if chinese else "  ** 跳过：参考不是中文 **"
        print(
            f"speaker{number}: {len(measured)} 条  F0 {calm[0]:.0f}->{lively[0]:.0f} "
            f"(跨度 {lively[0] - calm[0]:.0f} Hz)  说话人余弦 {pair:.4f}{mark}",
            flush=True,
        )
        if not chinese:
            continue
        for tag, (_, index, codes, _) in (("calm", calm), ("lively", lively)):
            pack[f"speaker{number}_{tag}"] = {
                "ref_codes": codes,
                "spk_emb": rows[index][1],
            }

    torch.save(pack, str(args.output / "voices.pt"))
    print(f"\n写入 {args.output / 'voices.pt'}：{len(pack)} 个条目")
    return {"same_speaker_bar": SAME_SPEAKER, "speakers": report, "entries": len(pack)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument(
        "--row-groups",
        type=int,
        nargs="*",
        help="which row groups to read. The file is blocked by language -- 0-25 are "
        "English, 51 onward Chinese -- so reading from the start returns one language",
    )
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--minimum-clips", type=int, default=8)
    parser.add_argument("--clips-per-cluster", type=int, default=16)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = harvest(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
