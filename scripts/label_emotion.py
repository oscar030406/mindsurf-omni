"""Label the training corpus's assistant audio with emotion2vec.

Native emotion has one honest route left. Prosody in this model rides the
speaker vector, and that vector *is* identity -- two independently designed
gates both said the price is a voice that no longer identifies as itself. The
way out is to stop making emotion share an input with identity and give it one
of its own, which needs a corpus that says what emotion each clip carries. Ours
does not: there is no emotion column, and that part was always true.

emotion2vec can write that column. This script is the pilot that decides
whether the full pass is worth ten hours of card, and the criterion is written
before it runs:

    fewer than 10% of clips carrying a non-neutral label kills the route.

That threshold is not about accuracy. A conditioning token learns nothing from
a column that is one value, so a corpus that is overwhelmingly neutral cannot
teach emotion no matter how good the labeller is -- and finding that out costs
one pilot instead of one night.

The audio is stored as interleaved Mimi codes, not waveform, so every clip pays
a decode before it can be heard. That decode is the throughput question this
pilot also answers.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

CODEBOOKS = 8
MIMI_RATE = 24_000
EMOTION_RATE = 16_000

# Interleaved tokens kept per clip, i.e. 30 seconds at 8 codebooks and 12.5 Hz.
#
# Not a quality choice, a memory one, and it was set by a crash: the corpus runs
# to 23,080 tokens (230 s) at the maximum, and Mimi's decode allocates with the
# square of the length -- one such clip asked for 11.15 GiB and took three
# workers down with it. The pilot never saw one because it sampled a thousand
# rows and the tail is 0.05% of the corpus.
#
# Truncating rather than skipping keeps every row: emotion is audible in the
# first half-minute, and a clip is labelled from its opening either way. What is
# lost is emotion that only appears late in a long clip, which the report counts
# so the size of that blind spot is on the record.
MAX_TOKENS = 3000


def deinterleave(tokens: list[int], codebooks: int = CODEBOOKS) -> list[list[int]]:
    """One flat list back into the eight codebook rows the codec expects.

    The corpus stores frames interleaved -- codebook 0 of frame 0, codebook 1 of
    frame 0, and so on -- which is what the training loader undoes at
    omni_dataset.py's ``for i in range(0, len(tokens) - 7, 8)``. Reading it as
    eight contiguous blocks instead would decode to noise without erroring,
    so this mirrors the loader rather than inventing a second convention.
    """
    rows: list[list[int]] = [[] for _ in range(codebooks)]
    for start in range(0, len(tokens) - (codebooks - 1), codebooks):
        for offset in range(codebooks):
            rows[offset].append(tokens[start + offset])
    return rows


def pitch_median(wave16: Any) -> float | None:
    """Median voiced F0, the same band measure_prosody.py uses."""
    import librosa
    import numpy

    track = librosa.yin(wave16, fmin=60, fmax=500, sr=EMOTION_RATE)
    voiced = track[(track > 65) & (track < 480)]
    return float(numpy.median(voiced)) if voiced.size else None


def all_rows(path: Path, shard: int, shards: int) -> Any:
    """Every row of the assigned row groups, one at a time.

    Sharding is by row group rather than by row so each worker opens the file
    at different offsets and never reads a group another worker is reading.
    102 groups over a handful of workers divides evenly enough that no worker
    finishes far ahead of the others.

    This yields instead of returning a list: the corpus is 414,024 rows and
    materialising them would cost more memory than the machine has spare while
    several workers run.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(str(path))
    for group in range(handle.metadata.num_row_groups):
        if group % shards != shard:
            continue
        table = handle.read_row_group(group, columns=["answer_audios"])
        turns_column = table.column("answer_audios")
        for index in range(table.num_rows):
            turns = turns_column[index].as_py()
            if not turns or not turns[0]:
                continue
            yield {"row_group": group, "index": index, "tokens": turns[0]}


def pick_rows(path: Path, wanted: int, seed: int) -> list[dict[str, Any]]:
    """Spread the sample over row groups instead of taking a prefix.

    The T2A corpus turned out to be stored in language blocks -- English row
    groups, then Chinese, then English again -- and nobody knew until it was
    checked. A prefix of this file
    could be one language, one speaker, or one source, and a label distribution
    measured on that would describe the prefix rather than the corpus.
    """
    import random

    import pyarrow.parquet as pq

    handle = pq.ParquetFile(str(path))
    groups = handle.metadata.num_row_groups
    per_group = max(1, wanted // groups + 1)
    rng = random.Random(seed)

    picked: list[dict[str, Any]] = []
    for group in range(groups):
        if len(picked) >= wanted:
            break
        table = handle.read_row_group(group, columns=["answer_audios", "conversations"])
        rows = table.num_rows
        if not rows:
            continue
        for index in sorted(rng.sample(range(rows), min(per_group, rows))):
            entry = table.slice(index, 1).to_pylist()[0]
            turns = entry.get("answer_audios") or []
            if not turns or not turns[0]:
                continue
            picked.append({"row_group": group, "index": index, "tokens": turns[0]})
            if len(picked) >= wanted:
                break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True, help="mimi directory")
    parser.add_argument(
        "--labeller",
        type=Path,
        required=True,
        help="local emotion2vec snapshot. A path, not a hub id: this machine has "
        "been without outbound network for 33 hours once, and a run that resolves "
        "a name at load time dies there",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--all",
        action="store_true",
        help="label the whole corpus instead of a sample. Use with --shard to "
        "split it across workers",
    )
    parser.add_argument(
        "--shard",
        default="0/1",
        help="i/N -- this worker takes row groups where group %% N == i. Sharding "
        "by group, not by row, so no two workers read the same group",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="torch threads for THIS worker. The default is deliberately not all "
        "of them: several workers run at once, and torch grabbing every core each "
        "has already cost this project 8 CPU hours for 35 minutes of work",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="interleaved tokens kept per clip. Bounds the codec's decode, which "
        "allocates with the square of the length and has already OOMed the card",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prosody",
        action="store_true",
        help="also measure each clip's median F0, so the label distribution can be "
        "checked against an acoustic correlate instead of taken on trust",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch
    import torchaudio
    from funasr import AutoModel
    from transformers import MimiModel

    torch.set_num_threads(args.cpu_threads)

    shard, shards = (int(part) for part in args.shard.split("/"))
    if not 0 <= shard < shards:
        raise SystemExit(f"--shard {args.shard} is not of the form i/N with 0 <= i < N")

    if args.all:
        print(f"全量，分片 {shard}/{shards}，torch 线程 {args.cpu_threads}", flush=True)
        picked: Any = all_rows(args.parquet, shard, shards)
    else:
        print(f"取样 {args.limit} 条，跨行组分散", flush=True)
        picked = pick_rows(args.parquet, args.limit, args.seed)
        print(f"  拿到 {len(picked)} 条", flush=True)
        if not picked:
            raise SystemExit(f"{args.parquet} 里没有可用的 answer_audios")

    mimi = MimiModel.from_pretrained(str(args.codec)).eval().float().to(args.device)
    resample = torchaudio.transforms.Resample(MIMI_RATE, EMOTION_RATE)
    labeller = AutoModel(model=str(args.labeller), disable_update=True, device=args.device)

    rows: list[dict[str, Any]] = []
    decode_seconds: list[float] = []
    label_seconds: list[float] = []
    audio_seconds: list[float] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    truncated = 0
    with args.output.open("w", encoding="utf-8") as sink:
        for position, item in enumerate(picked):
            started = time.perf_counter()
            tokens = item["tokens"]
            if len(tokens) > args.max_tokens:
                tokens = tokens[: args.max_tokens]
                truncated += 1
            codes = torch.tensor(deinterleave(tokens), dtype=torch.long)
            with torch.no_grad():
                wave = mimi.decode(codes.unsqueeze(0).to(args.device)).audio_values.squeeze()
            wave16 = resample(wave.detach().cpu().float())
            decoded_at = time.perf_counter()

            result = labeller.generate(
                wave16.numpy(), granularity="utterance", extract_embedding=False
            )
            finished = time.perf_counter()

            best = result[0]
            scores = list(best["scores"])
            labels = list(best["labels"])
            top = max(range(len(scores)), key=lambda i: scores[i])
            row = {
                "row_group": item["row_group"],
                "index": item["index"],
                "label": labels[top],
                "confidence": float(scores[top]),
                "audio_seconds": float(wave16.shape[-1]) / EMOTION_RATE,
            }
            if args.prosody:
                # The pilot's third question is whether the labels mean anything,
                # and the honest check for that is a listening test we cannot run
                # here. This is the substitute: pitch is the acoustic correlate
                # this project already has an instrument for, so if the labels
                # track nothing in F0 they are not tracking the audio either.
                # Weak evidence by design -- speaker sex moves F0 far more than
                # emotion does -- so it can catch a labeller that is guessing,
                # not confirm one that is right.
                row["f0_median"] = pitch_median(wave16.numpy())
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

            decode_seconds.append(decoded_at - started)
            label_seconds.append(finished - decoded_at)
            audio_seconds.append(row["audio_seconds"])
            # The full pass streams, so there is no total to count towards and
            # the rate is what tells a watcher whether it is on schedule.
            if (position + 1) % 500 == 0:
                rate = (position + 1) / (time.perf_counter() - run_started)
                sink.flush()
                print(f"  {position + 1} 条  {rate:.1f} 条/秒", flush=True)

    # emotion2vec's labels are bilingual ("生气/angry"); the English half is the
    # stable key, and 其他/other plus 未知/unk are not emotions -- counting them
    # as non-neutral would pass the gate on clips the labeller declined to call.
    counts = collections.Counter(row["label"] for row in rows)
    neutral = sum(count for label, count in counts.items() if "neutral" in label)
    unusable = sum(count for label, count in counts.items() if "other" in label or "unk" in label)
    emotional = len(rows) - neutral - unusable
    per_clip = statistics.mean(decode_seconds) + statistics.mean(label_seconds)

    import pyarrow.parquet as pq

    corpus_rows = pq.ParquetFile(str(args.parquet)).metadata.num_rows

    report = {
        "sampled": len(rows),
        "truncated_clips": truncated,
        "max_tokens": args.max_tokens,
        "corpus_rows": corpus_rows,
        "labels": dict(counts.most_common()),
        "emotional_fraction": emotional / len(rows),
        "neutral_fraction": neutral / len(rows),
        "unusable_fraction": unusable / len(rows),
        "seconds_per_clip": {
            "decode": statistics.mean(decode_seconds),
            "label": statistics.mean(label_seconds),
            "total": per_clip,
        },
        "audio_seconds_median": statistics.median(audio_seconds),
        "full_pass_hours": corpus_rows * per_clip / 3600,
        "gate": {
            "written_before_the_run": "emotional_fraction below 0.10 kills the route",
            "verdict": "pass" if emotional / len(rows) >= 0.10 else "fail",
        },
    }
    if args.prosody:
        pitched = [row for row in rows if row.get("f0_median")]
        by_label: dict[str, list[float]] = {}
        for row in pitched:
            by_label.setdefault(row["label"], []).append(row["f0_median"])
        report["f0_median_by_label"] = {
            label: {"n": len(values), "f0": statistics.median(values)}
            for label, values in sorted(by_label.items(), key=lambda kv: -len(kv[1]))
        }
        print("\n逐标签 F0 中位（弱证据：说话人性别对 F0 的影响远大于情绪）:")
        for label, entry in report["f0_median_by_label"].items():
            print(f"  {label:<18} n={entry['n']:<4} F0 {entry['f0']:.1f} Hz")
    print(f"\n标签分布: {dict(counts.most_common())}")
    print(
        f"有情绪 {report['emotional_fraction']:.3f}  中性 {report['neutral_fraction']:.3f}  "
        f"不可用 {report['unusable_fraction']:.3f}"
    )
    print(f"每条 {per_clip:.3f} s（解码 {report['seconds_per_clip']['decode']:.3f}）")
    print(f"全量 {corpus_rows} 条外推 {report['full_pass_hours']:.1f} 小时")
    print(f"判据（跑前写死）：有情绪 < 0.10 判死 —— {report['gate']['verdict']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"报告 {args.report}")
    sys.exit(0 if report["gate"]["verdict"] == "pass" else 2)


if __name__ == "__main__":
    main()
