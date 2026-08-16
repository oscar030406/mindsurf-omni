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

from mindsurf_omni.service.polish import BRIDGING_FILLERS, LEADING_FILLERS  # noqa: E402
from scripts.merge_polish_arms import dropped  # noqa: E402

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
        if any(word in cut or cut in word for word in VOCABULARY):
            continue
        broken.append("".join(source[index] for index in range(start, end) if index not in drop))
    return broken


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
        report[name] = {
            "sentences": len(rows),
            "sentences_with_a_cut": len(hit),
            "rate": len(hit) / max(1, len(rows)),
            "cuts": total,
            "examples": [piece for _, pieces in hit[: arguments.examples] for piece in pieces][
                : arguments.examples
            ],
        }
        print(
            f"{name:<24} {len(hit):>4}/{len(rows)} 句被切穿词 "
            f"({len(hit) / max(1, len(rows)):.2%})，共 {total} 处",
            flush=True,
        )
        print(f"{'':24} 例：{report[name]['examples']}", flush=True)

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"报告 {arguments.report}")


if __name__ == "__main__":
    main()
