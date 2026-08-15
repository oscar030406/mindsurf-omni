"""How far into the transcript the polisher got before it stopped.

Found by hand rather than by any of the four criteria. Dictating a 240-word
paragraph into the service came back cut off mid-phrase -- 81 characters out of
240 in, ending inside 识别错 -- and dictating 288 came back at 231. The model
emits its stop token early on input longer than it was trained on, and a
dictation tool that silently drops the second half of what you said is not
shippable at any content-retention number.

None of CER, filler clearance, content retention or invention can see it. They
are read on corpus sentences capped at 160 characters and read aloud by
edge-tts; the held-out set's median is 43. Truncation lives past the end of the
measurement.

The copy constraint makes the probe exact: the output is always a subsequence
of the transcript, so walking the transcript alongside the output says how much
of it was consumed. Stopping at 0.99 is a polisher that finished; stopping at
0.34 is one that gave up.

    python scripts/measure_polish_truncation.py --rows artifacts/polish_train/val_x.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# The share of the transcript below which an output is called truncated. Not a
# tuned number: an arm that consumed less than nine tenths of what it was given
# has dropped a clause, and one that dropped a clause is the failure this file
# exists to count.
FINISHED = 0.90


def consumed(source: str, output: str) -> float:
    """Share of ``source`` the output reached, matching greedily in order.

    The same walk ``polish.subsequence_pointer`` does at decode time, on
    characters rather than tokens so the number is readable.
    """
    if not source:
        return 1.0
    pointer = 0
    for char in output:
        while pointer < len(source) and source[pointer] != char:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer / len(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, action="append", type=Path, help="val_*.jsonl")
    args = parser.parse_args()

    print(f"{'臂':<34}{'n':>5}{'消耗中位':>10}{'截断句':>8}{'占比':>8}{'最短那条':>10}")
    for path in args.rows:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        shares = [consumed(row["source"], row["polished"]) for row in rows]
        cut = [value for value in shares if value < FINISHED]
        print(
            f"{path.stem:<34}{len(rows):>5}{statistics.median(shares):>10.4f}"
            f"{len(cut):>8}{len(cut) / len(rows):>8.1%}{min(shares):>10.3f}"
        )
    print(f"\n「截断」= 输出只走到转写的 {FINISHED:.0%} 以前。复制约束下输出必是转写的子序列，")
    print("所以这个指针是精确的，不是估计。四条判据都看不到这一维。")


if __name__ == "__main__":
    main()
