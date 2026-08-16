"""Combine two arms' deletions without retraining either of them.

The generator and the tagger get things wrong in different places, so what one
leaves in the other often takes out. Both arms already wrote, per sentence, the
transcript they were given and the text they produced; the deletions are
recoverable from that pair, and combining them is set arithmetic rather than a
seventh training run. Five minutes buys a whole row of the frontier.

* **Union** deletes what either arm deleted. Cleans the most and keeps the
  least -- the previous round's best working point was a union.
* **Intersection** deletes only what both agreed on. Keeps the most content,
  and is the only shape that ever cleared the retention line.
* **Veto** goes the other way: keep the first arm's deletion only where the
  later arms agree, so the head's confidence is used to protect content rather
  than to remove more of it. Two kinds of deletion are exempt from the veto,
  and both for the same reason as above -- a filler word the head cannot have
  missed, and an exact adjacent repetition, which a per-token head cannot see
  at all and would therefore veto every time.
* **Vocabulary union** takes the first arm's deletions whole, and from the
  later arms only the spans that spell a filler word. It exists because the
  complementarity is not symmetric and the asymmetry can be named: measured on
  986 sentences, the tagger clears 0.983 of the vocabulary filler and 0.437 of
  the repetition, while the generator reads 0.914 and 0.603. A per-token head
  recognises 嗯 from the token; it cannot compare two spans. So take the head's
  verdict where it is strong and ignore it everywhere else, instead of paying
  its content damage across the whole sentence.

Deletions are read off the alignment rather than trusted from a field: an arm
under a copy constraint only ever deletes, so source-to-output is exactly a
drop set, and reading it back means this works for any arm that writes the two
strings.

    python scripts/merge_polish_arms.py --arm artifacts/polish_train/val_a.jsonl \
        --arm artifacts/polish_train/val_b.jsonl --mode union \
        --output artifacts/polish_train/val_union.jsonl --report artifacts/polish-eval-union.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.evaluation.metrics import character_error_rate  # noqa: E402
from mindsurf_omni.service.polish import (  # noqa: E402
    BRIDGING_FILLERS,
    LEADING_FILLERS,
    tidy,
)
from scripts.measure_polish import content_kept, filler_removed, invented  # noqa: E402

# Longest first, so 你知道吧 is matched before 你知道 would be.
VOCABULARY = tuple(sorted((*LEADING_FILLERS, *BRIDGING_FILLERS), key=len, reverse=True))


def dropped(source: str, output: str) -> set[int]:
    """Indices of ``source`` the arm did not carry into ``output``.

    A replaced span counts as dropped: under the copy constraint an arm cannot
    substitute, so anything the alignment calls a replacement is a deletion the
    matcher paired with unrelated context.
    """
    kept = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
        None, source, output, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            kept.update(range(i1, i2))
    return set(range(len(source))) - kept


def vocabulary_spans(source: str, drop: set[int]) -> set[int]:
    """The part of ``drop`` that spells a whole filler word.

    Whole rather than partial: half a filler is the defect this is meant to
    avoid, not one to import. 你知道吧 deleted down to 你知道 reads worse than
    leaving it alone.
    """
    kept: set[int] = set()
    for word in VOCABULARY:
        start = source.find(word)
        while start != -1:
            span = range(start, start + len(word))
            if all(index in drop for index in span):
                kept.update(span)
            start = source.find(word, start + 1)
    return kept


def repetition_spans(source: str, drop: set[int], shortest: int = 2, longest: int = 5) -> set[int]:
    """Deletions that remove one copy of an exact adjacent repetition.

    Exempt from the veto because a per-token head cannot represent the
    judgement at all -- it reads one position at a time and 时间时间 looks like
    two ordinary words. Measured: the tagger clears 0.437 of the injected
    repetition against the generator's 0.603, and letting it veto that work
    costs the whole difference.

    Two characters and up, because one is not a repetition in Chinese: 今天天气
    holds 天天 and 看看 is an ordinary word. Measured both ways over 986 sentences
    -- a floor of one reads 0.9083 filler clearance against two's 0.9055, same
    retention to four places -- so the shorter floor buys 0.003 that is inside
    the noise and pays for it with a class of false positive that is not.
    """
    found: set[int] = set()
    for size in range(shortest, longest + 1):
        for start in range(len(source) - 2 * size + 1):
            span = range(start, start + size)
            if not all(index in drop for index in span):
                continue
            if source[start : start + size] == source[start + size : start + 2 * size]:
                found.update(span)
    return found


# tidy lives in the service now rather than here. It was written for this
# script and then the service shipped without it for a round, so the offline
# arms were tidied and the product was not -- the one place that has to do it
# is the one a user reads. Imported rather than copied so the next thing added
# to it reaches both.


def reached(source: str, output: str) -> int:
    """How far into ``source`` the output got, matching greedily in order.

    Under the copy constraint the output is a subsequence of the source, so
    this is exact. Used to tell a deletion from an absence: a generative arm
    that emits its stop token early leaves the whole tail looking deleted, and
    on a 184-character dictation that was 100 characters of "opinion" it never
    formed. Measured by hand, the generator returned 82 of 184 characters and
    stopped mid-phrase.
    """
    pointer = 0
    for char in output:
        while pointer < len(source) and source[pointer] != char:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer


def merge(rows: list[dict[str, Any]], mode: str) -> str:
    source = rows[0]["source"]
    drops = [dropped(source, row["polished"]) for row in rows]
    # Past where an arm stopped, it has no opinion. Without this the veto reads
    # a truncated arm as agreeing to delete the entire tail.
    for index, row in enumerate(rows):
        stopped = reached(source, row["polished"])
        if stopped < len(source):
            drops[index] = {position for position in drops[index] if position < stopped}
    if mode == "union":
        combined = set.union(*drops)
    elif mode == "intersection":
        combined = set.intersection(*drops)
    elif mode == "veto":
        exempt = repetition_spans(source, drops[0])
        for drop in drops:
            exempt |= vocabulary_spans(source, drop)
        combined = set.intersection(*drops) | exempt
    else:
        # The first arm whole, the rest only where they spell a filler.
        combined = set(drops[0])
        for drop in drops[1:]:
            combined |= vocabulary_spans(source, drop)
    return tidy("".join(char for index, char in enumerate(source) if index not in combined))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, action="append", type=Path, help="two or more")
    parser.add_argument(
        "--mode",
        choices=("union", "intersection", "vocabulary-union", "veto"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if len(args.arm) < 2:
        raise SystemExit("--arm 至少要给两次，不然没有可合的")

    arms = []
    for path in args.arm:
        arms.append(
            {
                json.loads(line)["id"]: json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        )
    shared = sorted(set.intersection(*(set(arm) for arm in arms)))
    if not shared:
        raise SystemExit("这几条臂没有共同的句子，合不了")

    written = []
    for key in shared:
        rows = [arm[key] for arm in arms]
        # A sentence the arms were given differently is not the same sentence,
        # and merging their deletions by index would silently cross the two.
        if len({row["source"] for row in rows}) != 1:
            raise SystemExit(f"{key} 在两条臂里的转写不一样，索引对不上")
        text = merge(rows, args.mode)
        arrived, removed = filler_removed(rows[0], text)
        written.append(
            {
                **rows[0],
                "polished": text,
                "cer_before": character_error_rate(rows[0]["target"], rows[0]["source"]),
                "cer_after": character_error_rate(rows[0]["target"], text),
                "content_kept": content_kept(rows[0]["target"], text),
                "invented": invented(rows[0]["target"], text),
                "filler_arrived": arrived,
                "filler_removed": removed,
            }
        )

    arrived = sum(row["filler_arrived"] for row in written)
    summary = {
        "mode": args.mode,
        "arms": [str(path) for path in args.arm],
        "n": len(written),
        "cer_before": statistics.fmean(row["cer_before"] for row in written),
        "cer_after": statistics.fmean(row["cer_after"] for row in written),
        "filler_removed_rate": (
            sum(row["filler_removed"] for row in written) / arrived if arrived else 0.0
        ),
        "content_kept": statistics.fmean(row["content_kept"] for row in written),
        "invented": statistics.fmean(row["invented"] for row in written),
        "empty": sum(1 for row in written if not row["polished"].strip()),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in written)
            + "\n",
            encoding="utf-8",
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )
    print(
        f"{args.mode} {len(written)} 句：CER {summary['cer_after']:.4f}"
        f"（输入端 {summary['cer_before']:.4f}）、口语词清除 {summary['filler_removed_rate']:.4f}、"
        f"内容保留 {summary['content_kept']:.4f}、编造 {summary['invented']:.4f}、"
        f"空 {summary['empty']}"
    )


if __name__ == "__main__":
    main()
