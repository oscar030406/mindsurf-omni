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
import math
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
# The tail of the reference, and only the tail, is what the model reads.
# speak_forced right-aligns the strip so it ends just before the assistant's
# first position and drops whatever does not fit, which on a typical prompt
# leaves about forty frames. A twenty-four second clip therefore contributes
# its last three seconds, while an F0 measured over the whole clip describes
# twenty-one seconds the model never sees. So clips are cut to their tail
# before anything is measured, at the length the shipped packs already use
# (5.9 to 9.4 seconds).
MAXIMUM_FRAMES = 125


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
    # Batched, and spread across the whole file rather than taken from the front.
    #
    # Two things force this. The corpus as published is a single row group of
    # 414,024 rows, so reading a group whole would materialise every reference
    # strip in the file as Python lists -- tens of gigabytes to sample a few
    # thousand. And the rows are blocked by language, which is why --row-groups
    # exists at all; with one row group that flag has nothing to select, so a
    # prefix of this file is one language and a sample drawn from it would
    # describe the prefix rather than the corpus. The same mistake was caught
    # once already when the T2A corpus turned out to be stored in language
    # blocks and nobody knew until it was checked.
    batch_size = 2048
    total_batches = max(1, math.ceil(handle.metadata.num_rows / batch_size))
    per_batch = max(1, math.ceil(args.sample / total_batches))
    for batch in handle.iter_batches(
        batch_size=batch_size, row_groups=wanted, columns=["ref_audios", "spk_emb"]
    ):
        taken = 0
        for ref, emb in zip(
            batch.column("ref_audios").to_pylist(),
            batch.column("spk_emb").to_pylist(),
            strict=False,
        ):
            if taken >= per_batch:
                break
            if ref and emb and len(emb) == 192 and len(ref) >= 8 * MINIMUM_FRAMES:
                rows.append((ref, torch.tensor(emb, dtype=torch.float32)))
                taken += 1
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
            codes = unpack(rows[index][0])[:, -MAXIMUM_FRAMES:]
            with torch.no_grad():
                # .float() before .numpy(): the codec's output dtype follows
                # whatever torch and transformers negotiate, and a newer pair
                # hands back float16 here. librosa's resampler takes float32,
                # float64, int16 or int32 and raises on float16, so the run dies
                # several minutes in, after the clustering, on a machine where
                # the same script worked before.
                wave = codec.decode(codes.unsqueeze(0))[0][0, 0].float().numpy()
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
        help="which row groups to read. Only useful on a repartitioned copy: the "
        "corpus as published is one row group, and the rows inside it are blocked "
        "by language, so the sample is spread across the whole file instead",
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
