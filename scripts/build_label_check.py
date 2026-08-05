"""Lay out a blind pack for checking emotion2vec's labels by ear.

The labelling pilot answered two of its three questions from a keyboard --
how many clips carry a non-neutral label, and how long a full pass costs. The
third one, whether the labels are right, needs ears, and the automatic stand-in
could not settle it: per-label F0 ordered the way you would expect but every
contrast came back reported-only.

So this builds the pack a person can actually score. The shape follows the MOS
pack that already exists (anonymous tokens, one sheet per rater, a key nobody
opens until the sheets are in), because a second convention is a second thing to
explain.

What it measures is PRECISION, not recall: clips are drawn stratified over the
labels emotion2vec assigned, so the result reads "when it says 开心, how often
does a person agree". It cannot say how many happy clips it filed as neutral --
that would need a sample drawn without reference to the labels, and at 68%
neutral most of it would be spent confirming neutrals.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

# The closed set the rater picks from. Deliberately the same words the labeller
# uses, minus the three that are too rare to check (disgusted / fearful / other
# were 7 clips in 1000) -- a rater offered an option that never appears will
# start reaching for it.
CHOICES = ("中立", "开心", "生气", "吃惊", "难过", "说不上来")

LABEL_TO_CHOICE = {
    "neutral": "中立",
    "happy": "开心",
    "angry": "生气",
    "surprised": "吃惊",
    "sad": "难过",
}

README = """情绪标签盲核

**这不是打分，是听完说出你听到的情绪。** 我们用 emotion2vec 给训练语料自动打了
情绪标签，那批标签**没有人验过**；自动替代品（逐标签 F0）分辨不了，所以要耳朵。

1. 用耳机，安静环境。每段约 10 秒，是模型训练语料里助手那一侧的音频。
2. 打开你自己那个 raterN 文件夹里的 raterN.csv，按 001、002…… 顺序听
   同一个文件夹里的 001.wav、002.wav——**编号就是行号，不用找文件**。
3. emotion_heard 列**只能填这六个之一**：
   中立 / 开心 / 生气 / 吃惊 / 难过 / 说不上来
   **听不出来就填「说不上来」，不要猜。** 猜出来的一致率是假的，
   而这一轮要判的恰恰是「这批标签能不能信」。
4. 判断只靠声音——语速、音高、力度、语气。**不要因为内容像好消息就填开心。**
5. note 列可选：录音有杂音、听不清、像是两个人说话，都写一句。
6. 不要打开 key.json——那是揭盲用的，看了这次核验就作废。
7. 有的片段会重复出现，这是故意的，照常填不要回头找。

三个人各自独立填，填完把 raterN.csv 交回来，用
`python scripts/build_label_check.py score --pack <这个目录>` 出结果。
"""


def token_for(entry: dict[str, Any], salt: str) -> str:
    raw = f"{salt}:{entry['row_group']}:{entry['index']}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def pick(rows: list[dict[str, Any]], per_label: int, seed: int) -> list[dict[str, Any]]:
    """Stratified over the assigned label, so每 class gets its own precision.

    Proportional sampling would spend two thirds of a fifty-clip budget on
    neutral and leave six clips for 难过, which is the class most likely to be
    wrong and the one a proportional sample can say least about.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        english = row["label"].split("/")[-1]
        if english in LABEL_TO_CHOICE:
            by_label[english].append(row)

    picked: list[dict[str, Any]] = []
    for english in LABEL_TO_CHOICE:
        pool = by_label.get(english, [])
        if not pool:
            print(f"  ⚠ {english} 一条都没有，跳过")
            continue
        if len(pool) < per_label:
            print(f"  ⚠ {english} 只有 {len(pool)} 条，少于 {per_label}")
        picked.extend(rng.sample(pool, min(per_label, len(pool))))
    rng.shuffle(picked)
    return picked


def build(args: argparse.Namespace) -> None:
    import soundfile
    import torch
    from transformers import MimiModel

    from scripts.label_emotion import EMOTION_RATE, MIMI_RATE, deinterleave

    rows = [
        json.loads(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"读到 {len(rows)} 条标签")
    picked = pick(rows, args.per_label, args.seed)
    print(f"分层抽出 {len(picked)} 条")

    import pyarrow.parquet as pq
    import torchaudio

    handle = pq.ParquetFile(str(args.parquet))
    mimi = MimiModel.from_pretrained(str(args.codec)).eval().float().to(args.device)
    resample = torchaudio.transforms.Resample(MIMI_RATE, EMOTION_RATE)

    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for item in picked:
        table = handle.read_row_group(item["row_group"], columns=["answer_audios"])
        turns = table.slice(item["index"], 1).to_pylist()[0]["answer_audios"]
        codes = torch.tensor(deinterleave(turns[0]), dtype=torch.long)
        with torch.no_grad():
            wave = mimi.decode(codes.unsqueeze(0).to(args.device)).audio_values.squeeze()
        token = token_for(item, args.salt)
        soundfile.write(
            str(audio_dir / f"{token}.wav"),
            resample(wave.detach().cpu().float()).numpy(),
            EMOTION_RATE,
        )
        entries.append(
            {
                "token": token,
                "label": item["label"],
                "choice": LABEL_TO_CHOICE[item["label"].split("/")[-1]],
                "row_group": item["row_group"],
                "index": item["index"],
            }
        )

    # Repeats measure whether one rater agrees with themselves. Without them a
    # low agreement rate cannot be split into "the labels are wrong" and "this
    # task is too hard to answer twice the same way".
    rng = random.Random(args.seed + 1)
    repeats = rng.sample(entries, min(args.repeats, len(entries)))
    order = entries + repeats
    rng.shuffle(order)

    for rater in range(1, args.raters + 1):
        path = args.output / f"rater{rater}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["token", "emotion_heard", "note"])
            for position, entry in enumerate(order):
                writer.writerow([f"{entry['token']}#{position}", "", ""])

    (args.output / "README.txt").write_text(README, encoding="utf-8")
    (args.output / "key.json").write_text(
        json.dumps(
            {
                "entries": entries,
                "repeat_tokens": [entry["token"] for entry in repeats],
                "choices": list(CHOICES),
                "measures": "precision of the assigned label, not recall",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"写好 {args.output}：{len(order)} 行 × {args.raters} 人，音频 {len(entries)} 段")
    print("**key.json 评分员不要看**")


def score(args: argparse.Namespace) -> None:
    key = json.loads((args.pack / "key.json").read_text(encoding="utf-8"))
    truth = {entry["token"]: entry["choice"] for entry in key["entries"]}

    # One sheet per rater folder, first column being that rater's play order --
    # the clips are numbered so a rater plays them straight through. The key
    # carries the position-to-token map per rater.
    layout = key.get("raters") or {}
    sheets = sorted(args.pack.glob("rater*/rater*.csv")) or sorted(args.pack.glob("rater*.csv"))
    filled: dict[str, list[str]] = collections.defaultdict(list)
    for sheet in sheets:
        rater = sheet.stem.replace("rater", "")
        positions = {row["position"]: row["token"] for row in layout.get(rater, [])}
        with sheet.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                answer = (row.get("emotion_heard") or "").strip()
                if not answer:
                    continue
                key_cell = (row.get("音频文件") or next(iter(row.values())) or "").strip()
                position = key_cell.removesuffix(".wav")
                filled[positions.get(position, "") or position.split("#")[0]].append(answer)
    if not filled:
        raise SystemExit(f"{args.pack} 里的 raterN.csv 还是空的，没人填")

    agree: collections.Counter[str] = collections.Counter()
    total: collections.Counter[str] = collections.Counter()
    unsure = 0
    for token, answers in filled.items():
        assigned = truth.get(token)
        if assigned is None:
            continue
        for answer in answers:
            if answer == "说不上来":
                unsure += 1
                continue
            total[assigned] += 1
            if answer == assigned:
                agree[assigned] += 1

    print(f"{len(sheets)} 份表，{sum(total.values())} 个可判判断，说不上来 {unsure} 个\n")
    for assigned in sorted(total, key=lambda label: -total[label]):
        share = agree[assigned] / total[assigned]
        print(f"  标为 {assigned:<4} n={total[assigned]:<4} 人耳同意 {share:.1%}")
    overall = sum(agree.values()) / sum(total.values()) if sum(total.values()) else 0.0
    print(f"\n整体一致率 {overall:.1%}（六选一盲猜约 16.7%，五个真标签盲猜 20%）")
    print("**这是精确率不是召回率**：抽样是按已分配的标签分层的，")
    print("它答不了「有多少开心被记成了中立」。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("build")
    make.add_argument("--labels", type=Path, required=True, help="pilot jsonl from label_emotion")
    make.add_argument("--parquet", type=Path, required=True)
    make.add_argument("--codec", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--per-label", type=int, default=10)
    make.add_argument("--raters", type=int, default=3)
    make.add_argument("--repeats", type=int, default=5)
    make.add_argument("--device", default="cuda")
    make.add_argument("--seed", type=int, default=20260801)
    make.add_argument("--salt", default="emotion-label-check")

    read = sub.add_parser("score")
    read.add_argument("--pack", type=Path, required=True)

    args = parser.parse_args()
    (build if args.command == "build" else score)(args)


if __name__ == "__main__":
    main()
