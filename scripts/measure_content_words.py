"""Did the polisher take the filler out, and did it leave the content words in.

Both numbers or neither. Seven of the nine words the vocabulary calls filler are
ordinary content most of the time -- 这个 in 这个模块 is not a hesitation -- and a
score that only counts filler clearance rewards deleting all of them, which is
what the shipped arms do. A score that only counts content-word survival rewards
deleting nothing. So this prints the pair, on every ruler it has.

Three rulers, three modes:

    --distribution
        How often a corpus of human-marked transcription calls each vocabulary
        word filler. No model, no GPU. This is what decides which words are in
        ``polish.DOUBLE_DUTY``, so it is recounted here rather than frozen into
        a comment.

    --cs2w DUMP
        Score an already-decoded dump against those human marks: how much of
        what a person deleted the arm deleted, and how much of what a person
        kept the arm deleted anyway. No GPU.

    --probes
        Run the real polish chain over ``configs/polish_probes_zh_v1.jsonl``.
        Needs the weights. Prints exact-match, content-word wrong-deletion rate
        and filler clearance separately, because the probe set is built so that
        deleting everything and deleting nothing each fail half of it.

    python scripts/measure_content_words.py --distribution
    python scripts/measure_content_words.py --cs2w artifacts/polish_train/val_cs2w_polish6.jsonl
    PYTHONPATH=src python scripts/measure_content_words.py --probes \
        --checkpoint D:/environment/models/mindsurf-local/sft_polish6_768.pth \
        --minimind-root D:/environment/models/minimind-o-repo --report artifacts/probes.json
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import (  # noqa: E402
    BRIDGING_FILLERS,
    DOUBLE_DUTY,
    LEADING_FILLERS,
)

SINGLE_DUTY = tuple(
    word for word in (*LEADING_FILLERS, *BRIDGING_FILLERS) if word not in DOUBLE_DUTY
)
PROBES = _ROOT / "configs/polish_probes_zh_v1.jsonl"
CS2W = _ROOT / "artifacts/polish_train/val_cs2w_polish6.jsonl"


def deleted(source: str, output: str) -> set[int]:
    """Indices of ``source`` missing from ``output``, by alignment."""
    kept: set[int] = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
        None, source, output, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            kept.update(range(i1, i2))
    return set(range(len(source))) - kept


def occurrences(source: str, words: tuple[str, ...]) -> list[tuple[int, int]]:
    spans = []
    for word in words:
        start = source.find(word)
        while start != -1:
            spans.append((start, start + len(word)))
            start = source.find(word, start + 1)
    return spans


def _cs2w_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CS2W.open(encoding="utf-8")]


def distribution() -> None:
    """How often a person marks each vocabulary word as something to delete."""
    rows = _cs2w_rows()
    counted = 0
    stats: dict[str, list[int]] = {
        word: [0, 0] for word in (*LEADING_FILLERS, *BRIDGING_FILLERS)
    }
    for row in rows:
        source, colloquial, repeats = (
            row["content"],
            row["colloquial_word_label"],
            row["disfluency_label"],
        )
        if len(colloquial) != len(source) or len(repeats) != len(source):
            continue
        counted += 1
        marked = {index for index, flag in enumerate(colloquial) if flag != "0"}
        marked |= {index for index, flag in enumerate(repeats) if flag != "0"}
        for word in stats:
            for start, end in occurrences(source, (word,)):
                stats[word][0] += 1
                stats[word][1] += all(index in marked for index in range(start, end))

    print(f"CS2W, {counted} human-marked sentences")
    print(f"{'word':<10}{'seen':>6}{'marked':>8}{'share':>9}   {'in DOUBLE_DUTY':>14}")
    for word, (seen, marked) in stats.items():
        share = f"{marked / seen:.3f}" if seen else "n/a"
        print(f"{word:<10}{seen:>6}{marked:>8}{share:>9}   {str(word in DOUBLE_DUTY):>14}")
    thin = [word for word, (seen, _) in stats.items() if seen < 10]
    if thin:
        print(f"\nToo few occurrences to decide from this corpus: {' '.join(thin)}")


def against_cs2w(dump: Path) -> dict[str, float]:
    """Deletion precision and recall against the human marks, split by word class.

    The marks are read off the dump's own rows rather than joined in by ``id``:
    CS2W repeats an id, so a join silently drops a sentence and scores another
    one against the wrong labels.
    """
    hit = miss = false = characters = sentences = 0
    wrong_double = wrong_single = 0
    for line in dump.open(encoding="utf-8"):
        row = json.loads(line)
        source = row["content"]
        colloquial, repeats = row["colloquial_word_label"], row["disfluency_label"]
        if len(colloquial) != len(source):
            continue
        characters += len(source)
        sentences += 1
        should = {index for index, flag in enumerate(colloquial) if flag != "0"}
        should |= {index for index, flag in enumerate(repeats) if flag != "0"}
        drop = deleted(source, row["polished"])
        hit += len(drop & should)
        miss += len(should - drop)
        wrong = drop - should
        false += len(wrong)
        for words, target in ((DOUBLE_DUTY, "double"), (SINGLE_DUTY, "single")):
            inside = set()
            for start, end in occurrences(source, words):
                inside |= set(range(start, end))
            if target == "double":
                wrong_double += len(wrong & inside)
            else:
                wrong_single += len(wrong & inside)
    report = {
        "sentences": sentences,
        "characters": characters,
        "deleted_right": hit,
        "missed": miss,
        "deleted_wrong": false,
        "precision": hit / (hit + false) if hit + false else 0.0,
        "recall": hit / (hit + miss) if hit + miss else 0.0,
        "wrong_deletion_rate": false / characters if characters else 0.0,
        "wrong_on_double_duty": wrong_double,
        "wrong_on_single_duty": wrong_single,
    }
    print(f"{dump.name}: {characters} characters over {sentences} human-marked sentences")
    print(f"  deleted right {hit}   missed {miss}   deleted wrong {false}")
    print(f"  precision {report['precision']:.4f}   recall {report['recall']:.4f}")
    print(f"  a person kept it and we deleted it: {false} = {report['wrong_deletion_rate']:.2%}")
    print(f"    of which on double-duty words {wrong_double}, on single-duty words {wrong_single}")
    return report


def score_probes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Both directions on the probe set, so neither extreme can win."""
    exact = sum(1 for row in rows if row["got"] == row["wanted"])
    content_seen = content_lost = 0
    filler_seen = filler_gone = 0
    for row in rows:
        source, got = row["source"], row["got"]
        drop = deleted(source, got)
        # A double-duty word the wanted answer keeps is content; every character
        # of it that went missing is a wrong deletion.
        for start, end in occurrences(source, DOUBLE_DUTY):
            word = source[start:end]
            if row["wanted"].count(word) < source.count(word) and source.count(word) > 1:
                continue  # a stutter: the wanted answer keeps fewer copies on purpose
            if word not in row["wanted"]:
                continue
            content_seen += end - start
            content_lost += len(drop & set(range(start, end)))
        for start, end in occurrences(source, SINGLE_DUTY):
            if source[start:end] in row["wanted"]:
                continue
            filler_seen += end - start
            filler_gone += len(drop & set(range(start, end)))
    return {
        "probes": len(rows),
        "exact": exact,
        "exact_rate": exact / len(rows) if rows else 0.0,
        "content_characters": content_seen,
        "content_deleted": content_lost,
        "content_loss_rate": content_lost / content_seen if content_seen else 0.0,
        "filler_characters": filler_seen,
        "filler_deleted": filler_gone,
        "filler_clearance": filler_gone / filler_seen if filler_seen else 0.0,
    }


async def run_probes(args: argparse.Namespace) -> list[dict[str, Any]]:
    from mindsurf_omni.service.polish import Polisher

    kwargs: dict[str, Any] = {
        "checkpoint": Path(args.checkpoint),
        "tokenizer_dir": Path(args.tokenizer_dir),
        "minimind_root": Path(args.minimind_root),
        "device": args.device,
    }
    if args.tagger:
        kwargs.update(
            tagger=Path(args.tagger),
            tagger_backbone=Path(args.tagger_backbone),
            tagger_threshold=args.tagger_threshold,
        )
    polisher = Polisher(**kwargs)
    polisher.load()
    rows = []
    for line in Path(args.probes).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        probe = json.loads(line)
        probe["got"] = await polisher.polish(probe["source"])
        rows.append(probe)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", action="store_true")
    parser.add_argument("--cs2w", type=Path)
    parser.add_argument("--probes", nargs="?", const=str(PROBES))
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer-dir", default=str(_ROOT / "assets/tokenizer"))
    parser.add_argument("--minimind-root")
    parser.add_argument("--tagger")
    parser.add_argument("--tagger-backbone")
    parser.add_argument("--tagger-threshold", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    if args.distribution:
        distribution()
    if args.cs2w:
        report["cs2w"] = against_cs2w(args.cs2w)
    if args.probes:
        rows = asyncio.run(run_probes(args))
        report["probes"] = score_probes(rows)
        report["rows"] = rows
        summary = report["probes"]
        print(f"{args.probes}: {summary['probes']} probes")
        print(f"  exactly the wanted answer {summary['exact']}/{summary['probes']}")
        print(
            f"  content words deleted  {summary['content_deleted']}/"
            f"{summary['content_characters']} = {summary['content_loss_rate']:.2%}"
        )
        print(
            f"  filler cleared         {summary['filler_deleted']}/"
            f"{summary['filler_characters']} = {summary['filler_clearance']:.2%}"
        )
        for row in rows:
            print(f"  {'OK ' if row['got'] == row['wanted'] else 'BAD'} {row['id']}  "
                  f"{row['source']}  ->  {row['got']}")
    if not (args.distribution or args.cs2w or args.probes):
        parser.error("pick one of --distribution, --cs2w, --probes")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
