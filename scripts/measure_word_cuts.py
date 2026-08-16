"""How often an arm's deletion cuts through the middle of a word.

The four acceptance numbers charge one character for 客户那边 becoming 客户边,
and a reader sees a word that does not exist. That is the same shape as the
stranded punctuation and the stranded particle before it: a defect the criteria
price at roughly zero and a person notices immediately. So it gets its own
reading rather than an argument.

Segmentation is jieba's, which is a judgement and not ground truth -- 表表 comes
back as one word, and 还是还是 as two. The number is therefore a rate to compare
arms with, not a defect count to trust absolutely.

    python scripts/measure_word_cuts.py \\
        --arm generative=artifacts/polish_train/val_bigholdout_polish6.jsonl \\
        --arm veto=artifacts/polish_train/val_bigholdout_vetochar_t0.5.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import (  # noqa: E402
    BRIDGING_FILLERS,
    LEADING_FILLERS,
    dropped,  # noqa: E402
)

VOCABULARY = (*LEADING_FILLERS, *BRIDGING_FILLERS)


def word_spans(text: str) -> list[tuple[int, int]]:
    import jieba

    jieba.setLogLevel(logging.ERROR)
    spans, cursor = [], 0
    for word in jieba.cut(text):
        spans.append((cursor, cursor + len(word)))
        cursor += len(word)
    return spans


def cuts(source: str, output: str) -> list[str]:
    """Words the deletion entered and did not finish, as the surviving fragment.

    A cut that spells a filler is not counted: jieba glues 就是 to its
    neighbour in 就是说, and deleting the filler out of that is the vocabulary
    doing its job rather than a word being broken.
    """
    drop = dropped(source, output)
    broken = []
    for start, end in word_spans(source):
        inside = [index for index in range(start, end) if index in drop]
        if not inside or len(inside) == end - start:
            continue
        cut = "".join(source[index] for index in inside)
        if any(word in cut for word in VOCABULARY):
            continue
        broken.append("".join(source[index] for index in range(start, end) if index not in drop))
    return broken


def both_copies_gone(source: str, output: str, shortest: int = 2, longest: int = 5) -> list[str]:
    """Repetitions where the deletion took every copy instead of one.

    A repetition is one copy too many, so removing it should leave one behind.
    Taking both is content loss wearing a deletion's clothes, and the criteria
    charge it the same two characters they charge a correct removal -- measured
    on a dictated note, 因为，因为上午张老师有课 came back with no 因为 at all
    and the causal link with it.

    A repeated filler is exempt: both copies of 就是就是 are filler, so taking
    both is the vocabulary doing its job. Without the exemption this counts
    them and the number says 1.52% where the real rate is a fifth of that --
    measured, and the examples were 其实 / 这个 / 就是 straight down the list.

    Read against the source rather than tracked through the merge: any arm that
    writes the two strings can be asked.
    """
    drop = dropped(source, output)
    lost = []
    for size in range(shortest, longest + 1):
        start = 0
        while start + 2 * size <= len(source):
            first = range(start, start + size)
            second = range(start + size, start + 2 * size)
            unit = source[start : start + size]
            if unit == source[start + size : start + 2 * size] and all(
                index in drop for index in (*first, *second)
            ):
                if not any(word in unit for word in VOCABULARY):
                    lost.append(unit)
                start += 2 * size
            else:
                start += 1
    return lost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name=path.jsonl")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--examples", type=int, default=6)
    arguments = parser.parse_args()

    report: dict[str, Any] = {}
    for entry in arguments.arm:
        name, _, path = entry.partition("=")
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        broken = [(row["source"], cuts(row["source"], row["polished"])) for row in rows]
        hit = [(source, pieces) for source, pieces in broken if pieces]
        total = sum(len(pieces) for _, pieces in broken)
        wiped = [both_copies_gone(row["source"], row["polished"]) for row in rows]
        wiped_hit = [pieces for pieces in wiped if pieces]
        report[name] = {
            "sentences": len(rows),
            "sentences_with_a_cut": len(hit),
            "rate": len(hit) / max(1, len(rows)),
            "cuts": total,
            "examples": [piece for _, pieces in hit[: arguments.examples] for piece in pieces][
                : arguments.examples
            ],
            "sentences_with_a_repetition_wiped": len(wiped_hit),
            "repetition_wiped_rate": len(wiped_hit) / max(1, len(rows)),
            "repetition_wiped_examples": [
                piece for pieces in wiped_hit[: arguments.examples] for piece in pieces
            ][: arguments.examples],
        }
        print(
            f"{name:<24} {len(hit):>4}/{len(rows)} 句被切穿词 "
            f"({len(hit) / max(1, len(rows)):.2%})，共 {total} 处",
            flush=True,
        )
        print(f"{'':24} 例：{report[name]['examples']}", flush=True)
        print(
            f"{'':24} 重复被整对删掉 {len(wiped_hit)}/{len(rows)} 句 "
            f"({len(wiped_hit) / max(1, len(rows)):.2%})，例："
            f"{report[name]['repetition_wiped_examples']}",
            flush=True,
        )

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"报告 {arguments.report}")


if __name__ == "__main__":
    main()
