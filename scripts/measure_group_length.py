"""扫 TRAINED_LENGTH 这个旋钮，用一把不会奖励「少删」的尺子。

这个旋钮扫过一次，结论作废了。那次拿字错率和内容保留去比五个档位，而那批评测
目标是「转写内容减必删语气词」——**删得少的臂在这两个判据上天生得高分**。切得越
碎删得越多（4.12% → 5.27%），于是表看着单调，量的却是「哪种删得少」。更糟的是
「不切」那档有 19/200 片超出模型见过的长度，模型答空、FLOOR 兜底把整片退回原文，
一个字没删，白拿高分。

所以这一版换判据，并且换数据。

**判据不需要人工标签，也不偏袒任何方向。** 两半各自记，互相拉扯：

* 召回：真的该删的东西，删掉了多少。「真的该删」只认两类——嗯 / 呃（CS2W 上人标
  为口语词 78/78 和 43/45），以及相邻的整段重复。少删，这个数直接掉。
* 误删：没有任何理由支持的删除有多少。不在任何词表词里、不在任何重复里的字被删
  掉了，就记一次。多删，这个数直接涨。

中间那批两栖词（那个 / 就是 / 然后 / 对 …，CS2W 上只有两成上下是口语词）**不打分，
整段屏蔽掉**——真值判不了的位置不该进分母。屏蔽在所有档位上是同一套，所以它不会
造出趋势，只会让每个档位的绝对值都偏低一点。

**尺子自己先受检。** `--calibrate` 拿 CS2W 的人工逐字标签当真值，量这把探测器本身
的精确率和召回率。探测器多准，下面的数就只能读到多细。上一次栽在「判据看不见它
要找的东西」，所以这次先验尺子再用尺子。

**数据用真的长口述。** artifacts/polish_train/pairs_long.jsonl 的 1200 条来自 RAMC
的长段落，SenseVoice 转写，中位 254 字、最长 4277 字、1125 条超过 160。只用它的
source——它的 target 是从人工转写来的另一份文本（1200 条里只有 44 条是 source 的
子序列，连「皇帝 / 黄帝」这种识别错都不一样），拿来算删除会把识别错当成不流利。

**顺带记下机制。** 每个档位都记 FLOOR 兜底触发了几片、退回了多少字、worth_polishing
跳过了几片、真正进模型的字有多少、切出来的片有多长。假设是「上次那个单调趋势是
兜底制造的」——兜底率随档位怎么走，这个假设当场就能被证伪。

    python scripts/measure_group_length.py --data artifacts/polish_train/pairs_long.jsonl \\
        --checkpoint out/sft_polish6_768.pth --minimind-root ~/omni/minimind-o \\
        --lengths 80,120,160,240,320,0 --report artifacts/group-length.json

    python scripts/measure_group_length.py --calibrate <cs2w>/test.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import (  # noqa: E402
    DOUBLE_DUTY,
    VOCABULARY,
    Polisher,
    dropped,
    group_sentences,
    worth_polishing,
)

# The two words CS2W marks as filler nearly every time it sees them. Everything
# else in VOCABULARY earns its keep as content often enough that a deletion of
# it is not evidence either way -- see DOUBLE_DUTY's note in polish.py.
CERTAIN_FILLERS = ("嗯", "呃")

# A repetition has to be at least two characters. One is not a repetition in
# Chinese: 天天 is a word and 看看 is a word.
REPEAT_SHORTEST = 2
REPEAT_LONGEST = 5


def filler_occurrences(source: str) -> list[range]:
    """Where the two unambiguous fillers sit."""
    out = []
    for word in CERTAIN_FILLERS:
        start = source.find(word)
        while start != -1:
            out.append(range(start, start + len(word)))
            start = source.find(word, start + 1)
    return out


def repetition_occurrences(source: str) -> list[range]:
    """Adjacent exact repetitions, longest match first, no overlaps.

    The span returned covers **both** copies. Which copy an arm deletes is its
    own business and difflib reports whichever aligns, so the test downstream is
    "at least one copy's worth of characters went", not "these exact indices".
    """
    taken: set[int] = set()
    out: list[range] = []
    for size in range(REPEAT_LONGEST, REPEAT_SHORTEST - 1, -1):
        for start in range(len(source) - 2 * size + 1):
            span = range(start, start + 2 * size)
            if any(index in taken for index in span):
                continue
            if source[start : start + size] != source[start + size : start + 2 * size]:
                continue
            taken.update(span)
            out.append(span)
    return sorted(out, key=lambda span: span.start)


def unscorable(source: str) -> set[int]:
    """Positions the detector cannot call, masked out of both classes.

    Every double-duty word, plus one-character doublings. Deleting one of these
    is neither right nor wrong as far as this ruler can tell, and scoring a
    coin flip as if it were truth is how the last three readings went wrong.
    """
    out: set[int] = set()
    for word in DOUBLE_DUTY:
        start = source.find(word)
        while start != -1:
            out.update(range(start, start + len(word)))
            start = source.find(word, start + 1)
    for index in range(len(source) - 1):
        if source[index] == source[index + 1]:
            out.update((index, index + 1))
    return out


# Marks are not scored either way. ``tidy`` deletes punctuation a deletion
# stranded, which is right and is not in ``justified``; leaving marks in would
# book every one of those as over-deletion. Worse, how much gets stranded rises
# with how much was deleted, so the bias would track the arm rather than sit
# still, and a bias that tracks the arm is a trend.
MARKS = set("，。！？；：、,.!?;:—… \n\u3000")


def justified(source: str) -> set[int]:
    """Every position some rule could defend deleting.

    Wider than the gold on purpose: a deletion outside this set has nothing
    speaking for it, which is what makes it countable as over-deletion without
    a human label.
    """
    out: set[int] = set()
    for span in filler_occurrences(source):
        out.update(span)
    for span in repetition_occurrences(source):
        out.update(span)
    for word in VOCABULARY:
        start = source.find(word)
        while start != -1:
            out.update(range(start, start + len(word)))
            start = source.find(word, start + 1)
    return out | unscorable(source) | {i for i, c in enumerate(source) if c in MARKS}


def reached_model(source: str, longest: int) -> int:
    """Characters the model actually saw, which is what over-deletion divides by.

    A piece the gate skipped and a piece that came back unpolished cannot
    contribute a wrong deletion, so counting them in the denominator makes an
    arm look cleaner exactly where it stopped working.
    """
    pieces = group_sentences(source, longest) if longest else [source]
    return sum(len(p) for p in pieces if p.strip() and worth_polishing(p))


def is_subsequence(part: str, whole: str) -> bool:
    walk = iter(whole)
    return all(character in walk for character in part)


def score(source: str, output: str, target: str | None = None) -> dict[str, int]:
    """One transcript's counts. Summed across the set, never averaged per row.

    ``target`` is scored only where it is a real deletion of the source --
    injected sets are, and then every deleted position is known rather than
    detected. That is the only way to answer the question the detector cannot:
    the double-duty words (那个 / 就是 / 然后) are masked out of the detector's
    gold because a corpus says they are content most of the time, and the
    injected sets plant them as filler at a fixed rate. A reading that masks
    them cannot see a change that lands on them.

    Positional, so a repetition scores against whichever copy difflib aligned.
    That ambiguity is identical across arms and was present in the reading this
    is being compared against.
    """
    gone = dropped(source, output)
    fillers = filler_occurrences(source)
    repeats = repetition_occurrences(source)
    return {
        "filler_total": len(fillers),
        "filler_removed": sum(1 for span in fillers if all(i in gone for i in span)),
        "repeat_total": len(repeats),
        # Half the span is one copy. At least that many characters gone from
        # inside it means a copy was taken, whichever copy difflib aligned.
        "repeat_removed": sum(1 for span in repeats if len(gone & set(span)) >= len(span) // 2),
        "over_deleted": len(gone - justified(source)),
        "deleted": len(gone),
        "chars": len(source),
        "chars_to_model": 0,
        "truth_labelled": 0,
        "truth_total": 0,
        "truth_hit": 0,
        "truth_claimed": 0,
        "truth_total_ambiguous": 0,
        "truth_hit_ambiguous": 0,
        "truth_total_plain": 0,
        "truth_hit_plain": 0,
    }


def with_truth(counts: dict[str, int], source: str, output: str, target: str) -> None:
    """Fill in the exact-label half, when the set carries exact labels."""
    if not target or not is_subsequence(target, source):
        return
    gold = dropped(source, target)
    gone = dropped(source, output)
    counts["truth_labelled"] = 1
    counts["truth_total"] = len(gold)
    counts["truth_hit"] = len(gone & gold)
    counts["truth_claimed"] = len(gone)
    # The same recall, split by whether the character sits inside a word the
    # detector refuses to call. Whatever the two rulers disagree about has to
    # be on one side of this line.
    ambiguous = unscorable(source)
    inside = gold & ambiguous
    outside = gold - ambiguous
    counts["truth_total_ambiguous"] = len(inside)
    counts["truth_hit_ambiguous"] = len(gone & inside)
    counts["truth_total_plain"] = len(outside)
    counts["truth_hit_plain"] = len(gone & outside)


def totals(rows: list[dict[str, int]]) -> dict[str, float]:
    out = {key: sum(row[key] for row in rows) for key in rows[0]}
    out["filler_recall"] = (
        out["filler_removed"] / out["filler_total"] if out["filler_total"] else 0.0
    )
    out["repeat_recall"] = (
        out["repeat_removed"] / out["repeat_total"] if out["repeat_total"] else 0.0
    )
    seen = out["chars_to_model"] or out["chars"]
    out["over_per_1k"] = 1000.0 * out["over_deleted"] / seen if seen else 0.0
    out["deleted_share"] = out["deleted"] / out["chars"] if out["chars"] else 0.0
    if out["truth_labelled"]:
        out["truth_recall"] = out["truth_hit"] / out["truth_total"] if out["truth_total"] else 0.0
        out["truth_precision"] = (
            out["truth_hit"] / out["truth_claimed"] if out["truth_claimed"] else 0.0
        )
        recall, precision = out["truth_recall"], out["truth_precision"]
        out["truth_f1"] = (
            2 * recall * precision / (recall + precision) if recall + precision else 0.0
        )
        out["truth_recall_ambiguous"] = (
            out["truth_hit_ambiguous"] / out["truth_total_ambiguous"]
            if out["truth_total_ambiguous"]
            else 0.0
        )
        out["truth_recall_plain"] = (
            out["truth_hit_plain"] / out["truth_total_plain"] if out["truth_total_plain"] else 0.0
        )
    return out


def calibrate(path: Path) -> dict[str, float]:
    """The detector against CS2W's human labels, before anyone reads it.

    Scored the way the sweep scores: **per occurrence**, not per character. The
    first version of this function scored per character and read 0.5213
    precision, which looked like a broken detector and was a broken framing --
    where a speaker says 我我, deleting either 我 gives the same text, the human
    deleted one and the detector had named the other, and one agreement was
    counted as two errors. Split apart, fillers land at 0.9835 and repetitions
    at 0.9296. It is the same shape as the punctuation check that cost this
    stage a night: the ruler could not see the thing it was counting.

    Recall against the human marks is genuinely low and is meant to be. A real
    speaker's restart ("值得大家应，值得大家注意的") is not an exact copy, and
    the detector only finds exact copies. What it claims is a high-precision
    subset of what a person would mark -- which is what a gold subset has to be,
    and why the sweep below reads recall *on this subset* rather than on
    disfluency in general.
    """
    filler_total = filler_agreed = 0
    repeat_total = repeat_agreed = 0
    human_marked = detector_claimed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "disfluency_label" in row:
            source, labels = row["content"], row["disfluency_label"]
            if len(source) != len(labels):
                continue
            gold = {index for index, mark in enumerate(labels) if mark in "12"}
        else:
            # The converted pool. cs2w_to_pairs.py builds the target as the
            # source minus the marked characters in order and nothing else, so
            # the diff recovers the marks -- checked: all 6501 targets are
            # subsequences of their source.
            source, target = row["source"], row["target"]
            gold = dropped(source, target)
        human_marked += len(gold)
        for span in filler_occurrences(source):
            filler_total += 1
            detector_claimed += len(span)
            if any(index in gold for index in span):
                filler_agreed += 1
        for span in repetition_occurrences(source):
            repeat_total += 1
            detector_claimed += len(span) // 2
            if len(gold & set(span)) >= len(span) // 2:
                repeat_agreed += 1

    filler_precision = filler_agreed / filler_total if filler_total else 0.0
    repeat_precision = repeat_agreed / repeat_total if repeat_total else 0.0
    claimed = filler_total + repeat_total
    agreed = filler_agreed + repeat_agreed
    print(
        f"探测器 vs 人工标签（按处算）：嗯/呃 {filler_agreed}/{filler_total} = "
        f"{filler_precision:.4f}，重复 {repeat_agreed}/{repeat_total} = {repeat_precision:.4f}，"
        f"合计 {agreed}/{claimed} = {agreed / claimed:.4f}"
    )
    print(
        f"  人一共标了 {human_marked} 个字，探测器只认领 {detector_claimed} 个——"
        "它找的是整段复制，真人的改口不是整段复制，所以覆盖面小是设计如此。"
    )
    return {
        "filler_occurrences": filler_total,
        "filler_precision": filler_precision,
        "repeat_occurrences": repeat_total,
        "repeat_precision": repeat_precision,
        "precision": agreed / claimed if claimed else 0.0,
        "human_marked_chars": human_marked,
        "detector_claimed_chars": detector_claimed,
    }


def shape(sources: list[str], longest: int) -> dict[str, float]:
    """What the grouper produces, before any model runs."""
    pieces: list[str] = []
    for text in sources:
        pieces.extend(group_sentences(text, longest) if longest else [text])
    lengths = [len(piece) for piece in pieces]
    skipped = [piece for piece in pieces if not (piece.strip() and worth_polishing(piece))]
    return {
        "pieces": len(pieces),
        "piece_median": statistics.median(lengths),
        "piece_p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
        "piece_max": max(lengths),
        "over_160": sum(1 for x in lengths if x > 160),
        "skipped_pieces": len(skipped),
        "skipped_chars": sum(len(piece) for piece in skipped),
        "chars_to_model": sum(len(p) for p in pieces if p.strip() and worth_polishing(p)),
    }


async def run_arm(
    polisher: Polisher,
    sources: list[str],
    longest: int,
    targets: list[str] | None = None,
    dump: Path | None = None,
) -> dict:
    targets = targets or [""] * len(sources)
    polisher.group_longest = longest if longest else 10**9
    polisher.floored = polisher.floored_chars = 0
    polisher.emptied = polisher.emptied_chars = 0
    rows: list[dict[str, int]] = []
    written: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, text in enumerate(sources):
        output = await polisher.polish(text)
        if dump is not None:
            written.append({"source": text, "polished": output})
        row = score(text, output)
        row["chars_to_model"] = reached_model(text, polisher.group_longest)
        with_truth(row, text, output, targets[index])
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(f"    {index + 1}/{len(sources)}", flush=True)
    out = totals(rows)
    out["seconds"] = time.perf_counter() - started
    # Off the Polisher, counted where the fallback happens. A piece that fell
    # back is identical to one the model left alone, so it cannot be recovered
    # from the output afterwards.
    out["floored_pieces"] = polisher.floored
    out["floored_chars"] = polisher.floored_chars
    out["emptied_pieces"] = polisher.emptied
    out["emptied_chars"] = polisher.emptied_chars
    out["unpolished_chars"] = polisher.floored_chars + polisher.emptied_chars
    if dump is not None:
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in written) + "\n",
            encoding="utf-8",
        )
    out["unpolished_share"] = (
        out["unpolished_chars"] / out["chars_to_model"] if out["chars_to_model"] else 0.0
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--calibrate", type=Path, help="CS2W test.jsonl，先验尺子")
    parser.add_argument("--lengths", default="80,120,160,240,320,0", help="0 表示不切")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shape-only", action="store_true", help="不动模型，只看切出来的形状")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--minimind-root", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=_ROOT / "assets/tokenizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tagger", type=Path)
    parser.add_argument("--tagger-backbone", type=Path)
    parser.add_argument("--tagger-threshold", type=float, default=0.4)
    parser.add_argument(
        "--merge", default="veto", help="veto / union / intersection，两条臂怎么合"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--dump", type=Path, help="每档把 source/polished 写一份，供否定词等安全判据读"
    )
    args = parser.parse_args()

    report: dict = {}
    if args.calibrate:
        report["calibration"] = calibrate(args.calibrate)
    if not args.data:
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
        return

    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x.strip()]
    kept = [r for r in rows if r.get("split", args.split) == args.split and r["source"].strip()][
        : args.limit
    ]
    sources = [r["source"] for r in kept]
    targets = [r.get("target", "") for r in kept]
    lengths = [int(x) for x in args.lengths.split(",")]
    total = sum(len(s) for s in sources)
    middle = statistics.median([len(s) for s in sources])
    print(f"{len(sources)} 段真长口述，{total} 字，中位 {middle:.0f} 字")

    report["shape"] = {}
    print(
        f"\n{'档位':>6}{'片数':>8}{'片中位':>8}{'片P95':>8}{'片最长':>8}{'>160':>7}{'跳过片':>8}{'进模型字':>10}"
    )
    for longest in lengths:
        got = shape(sources, longest)
        report["shape"][str(longest)] = got
        name = "不切" if not longest else str(longest)
        print(
            f"{name:>6}{got['pieces']:>8}{got['piece_median']:>8.0f}{got['piece_p95']:>8}"
            f"{got['piece_max']:>8}{got['over_160']:>7}{got['skipped_pieces']:>8}{got['chars_to_model']:>10}"
        )

    if not args.shape_only:
        if not (args.checkpoint and args.minimind_root):
            raise SystemExit("要跑模型就得给 --checkpoint 和 --minimind-root")
        polisher = Polisher(
            checkpoint=args.checkpoint,
            tokenizer_dir=args.tokenizer,
            minimind_root=args.minimind_root,
            device=args.device,
            tagger=args.tagger,
            tagger_backbone=args.tagger_backbone,
            tagger_threshold=args.tagger_threshold,
            merge_mode=args.merge,
        )
        polisher.load()
        report["arms"] = {}
        print(
            f"\n{'档位':>6}{'口语词召回':>12}{'重复召回':>10}{'误删/千字':>11}{'删除率':>9}{'退回字占比':>12}{'停住片':>8}{'答空片':>8}{'秒':>8}{'真标签召回':>12}{'真标签精确':>12}{'两栖词召回':>14}{'非两栖召回':>14}"
        )

        async def every_arm() -> None:
            # One event loop for the whole sweep. A fresh asyncio.run per arm
            # leaves the polisher's worker bound to the loop that just closed and
            # the next arm hangs with the card at 0%, which reads as slowness
            # rather than as a hang. The first version of this script did that.
            for longest in lengths:
                print(f"  跑 {longest or '不切'} …", flush=True)
                spill = args.dump / f"arm{longest}.jsonl" if args.dump else None
                got = await run_arm(polisher, sources, longest, targets, spill)
                report["arms"][str(longest)] = got
                name = "不切" if not longest else str(longest)
                print(
                    f"{name:>6}{got['filler_recall']:>12.4f}{got['repeat_recall']:>10.4f}"
                    f"{got['over_per_1k']:>11.2f}{got['deleted_share']:>9.4f}"
                    f"{got['unpolished_share']:>12.4f}{got['floored_pieces']:>8}"
                    f"{got['emptied_pieces']:>8}{got['seconds']:>8.0f}"
                    + (
                        f"{got['truth_recall']:>10.4f}{got['truth_precision']:>10.4f}"
                        f"{got['truth_recall_ambiguous']:>12.4f}"
                        f"{got['truth_recall_plain']:>12.4f}"
                        f"  ({got['truth_total_ambiguous']}/{got['truth_total_plain']})"
                        if "truth_recall" in got
                        else ""
                    ),
                    flush=True,
                )
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(
                        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
                    )

        asyncio.run(every_arm())

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n写到 {args.report}")


if __name__ == "__main__":
    main()
