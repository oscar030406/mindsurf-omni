"""What this stage deletes, scored against people rather than against an injector.

Every headline number so far is measured on 986 held-out sentences whose filler
we planted ourselves, and that target has a hole in it: the injector picks
uniformly from nine words, seven of which are double-duty, so the 这个 it planted
is labelled "delete this" while the 这个 in 这个模块 is not -- same spelling, same
slot, and only the injection record can tell them apart. Every fix that taught
the stage to keep a double-duty word in a content position moved those numbers
the wrong way.

CS2W does not have that hole. 723 real transcripts, each carrying a per-character
judgement from a person about what is disfluency and what is a word they meant,
and no one planted anything. The stage is scored against that judgement here.

Three questions, and the third is the one that has never been answered:

* **Did it remove what people marked?** Recall over the marked characters.
* **Did it keep what they did not?** Every deletion of an unmarked character,
  counted -- this is the number a user sees as a broken sentence.
* **What are the unmarked deletions made of?** Not a rate but a list: which
  characters, how often, in what context. The last reading of this said 61.9%
  of deletions were explained by neither the vocabulary nor a repetition, and
  stopped there; the 219 characters behind that number went to a file that was
  never committed, so nobody has ever seen what they were.

    python scripts/measure_against_human_labels.py \\
        --rows artifacts/polish_train/val_cs2w_polish6.jsonl \\
        --checkpoint out/sft_polish6_768.pth --minimind-root ../minimind-o \\
        --tagger out/polish_tagger_unionchar.pt \\
        --tagger-backbone out/polish_tagger_unionchar_backbone.pth \\
        --report artifacts/human-labels.json

``--stored`` scores the ``polished`` column instead of running the model, which
is how a past run is re-read rather than re-done. Say which one produced it: the
column in this corpus is from 2026-08-16 and the stage has changed since.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import (  # noqa: E402
    ALWAYS_FILLER,
    DOUBLE_DUTY,
    PUNCTUATION,
    VOCABULARY,
    Polisher,
)
from scripts.measure_content_words import deleted  # noqa: E402

# How much of the sentence around an unexplained deletion to keep, so the list
# can be read rather than counted. Wide enough to see the word it came out of.
CONTEXT = 8

# How far a repetition's other copy may sit before this stops calling it one.
# A restart puts a few characters between the two -- 是用两只是用两只 -- and
# zero would see only the touching kind. Bounded so that a common character
# recurring later in a long sentence is not read as a repetition of it.
GAP = 6


def marked(row: dict[str, Any]) -> set[int] | None:
    """Character positions a person marked as disfluency or colloquial.

    None when the two label strings do not line up with the text, which happens
    in this corpus and is a dropped row rather than a guess.
    """
    source = row["content"]
    colloquial, repeats = row["colloquial_word_label"], row["disfluency_label"]
    if len(colloquial) != len(source) or len(repeats) != len(source):
        return None
    return {index for index, flag in enumerate(colloquial) if flag != "0"} | {
        index for index, flag in enumerate(repeats) if flag != "0"
    }


def explain(source: str, index: int) -> str:
    """Why this deletion might be defensible, in one word.

    The buckets are the ones the last reading used, so the two are comparable,
    plus the one it did not have: whether the character sits inside a
    double-duty word, which is the class the stage was later taught to keep.
    """
    char = source[index]
    if char in PUNCTUATION:
        return "punctuation"
    if char in ALWAYS_FILLER:
        return "interjection"
    # Both lists, because they are not nested. 那种 那些 那 就 对 were added to
    # DOUBLE_DUTY when the stage was taught to keep them and were never in
    # VOCABULARY, so checking only the second called every one of them
    # unexplained -- 我都是买的那种大版 came out in that pile, which is exactly
    # the class this bucket exists to separate.
    for word in sorted({*VOCABULARY, *DOUBLE_DUTY}, key=len, reverse=True):
        start = max(0, index - len(word) + 1)
        if any(source[at : at + len(word)] == word for at in range(start, index + 1)):
            return "double_duty" if word in DOUBLE_DUTY else "vocabulary"
    # A repetition, adjacent or not. Both shapes, because a restart is the
    # commonest one in real speech and the adjacent-only test called it
    # unexplained: 国人吃饭是用两只是用两只手 has the speaker starting the
    # phrase over, and the copy that goes is four characters from its twin, not
    # touching it. Reading that as an unexplained deletion was the first
    # version of this bucket and it put most of the residue in the wrong pile.
    for size in range(1, 7):
        here = source[index - size + 1 : index + 1] if index + 1 >= size else ""
        if not here:
            continue
        for start in range(
            max(0, index - size - GAP + 1), min(len(source) - size, index + GAP) + 1
        ):
            if start != index - size + 1 and source[start : start + size] == here:
                return "repetition"
    return "unexplained"


def score(rows: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    hit = miss = wrong = characters = 0
    counted = 0
    buckets: collections.Counter[str] = collections.Counter()
    unexplained: collections.Counter[str] = collections.Counter()
    examples: list[dict[str, str]] = []

    for row, output in zip(rows, outputs, strict=True):
        should = marked(row)
        if should is None:
            continue
        counted += 1
        source = row["content"]
        characters += len(source)
        drop = deleted(source, output)
        hit += len(drop & should)
        miss += len(should - drop)
        for index in sorted(drop - should):
            wrong += 1
            why = explain(source, index)
            buckets[why] += 1
            if why == "unexplained":
                unexplained[source[index]] += 1
                if len(examples) < 40:
                    left = max(0, index - CONTEXT)
                    examples.append(
                        {
                            "char": source[index],
                            "context": source[left : index + CONTEXT],
                            "at": str(index),
                        }
                    )

    return {
        "sentences": counted,
        "characters": characters,
        # Against people, not against an injector: the marked characters are
        # what a person said should go.
        "recall": hit / (hit + miss) if hit + miss else None,
        "marked_total": hit + miss,
        "deleted_unmarked": wrong,
        "deleted_unmarked_per_thousand": 1000 * wrong / characters if characters else None,
        # What the unmarked deletions are made of. The last reading gave only
        # the share of the last bucket and threw the characters away.
        "unmarked_by_kind": dict(buckets.most_common()),
        "unexplained_characters": dict(unexplained.most_common(20)),
        "unexplained_examples": examples,
    }


async def run(polisher: Polisher, rows: list[dict[str, Any]]) -> list[str]:
    """One event loop for all of them; a fresh asyncio.run per row hangs the next."""
    out = []
    for index, row in enumerate(rows, 1):
        out.append(await polisher.polish(row["content"], "zh"))
        if index % 100 == 0:
            print(f"  {index}/{len(rows)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument(
        "--replay",
        type=Path,
        help="re-bucket a saved report's outputs instead of running anything",
    )
    parser.add_argument(
        "--stored",
        action="store_true",
        help="score the polished column instead of running the model",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tagger", type=Path)
    parser.add_argument("--tagger-backbone", type=Path)
    parser.add_argument("--tagger-threshold", type=float, default=0.4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.replay:
        saved = json.loads(args.replay.read_text(encoding="utf-8"))
        outputs = saved["outputs"]
        produced = f"{saved['outputs_from']}（从 {args.replay} 重读）"
    elif args.stored:
        outputs = [row["polished"] for row in rows]
        produced = f"{args.rows} 的 polished 列（不是今天的模型）"
    else:
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
        )
        polisher.load()
        outputs = asyncio.run(run(polisher, rows))
        produced = str(args.checkpoint)

    report = {
        "rows": str(args.rows),
        "outputs_from": produced,
        # The model's answers, so a change to the buckets can be re-read
        # without occupying the card again. The first two rounds of this
        # measurement each cost a GPU run for an arithmetic fix.
        "outputs": outputs,
        **score(rows, outputs),
    }
    print()
    print(f"{report['sentences']} 句 / {report['characters']} 字，输出来自 {produced}")
    print(f"  人标了该删的      {report['marked_total']} 处，删掉 {report['recall']:.4f}")
    print(
        f"  删了人没标的      {report['deleted_unmarked']} 字"
        f"（每千字 {report['deleted_unmarked_per_thousand']:.2f}）"
    )
    for kind, count in report["unmarked_by_kind"].items():
        share = count / max(report["deleted_unmarked"], 1)
        print(f"      {kind:14s} {count:5d}  {share:.1%}")
    if report["unexplained_characters"]:
        top = "、".join(f"{c}×{n}" for c, n in list(report["unexplained_characters"].items())[:8])
        print(f"  说不出理由的头几个字：{top}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n写到 {args.report}")


if __name__ == "__main__":
    main()
