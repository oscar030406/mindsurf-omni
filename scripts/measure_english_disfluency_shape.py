"""What English speakers' disfluencies are made of, and which we can touch.

The Chinese version of this question turned up the answer that the vocabulary
was not the hole -- repetition was half the job and the injector made the wrong
shape. English has never been asked. The rule arm scores 265/300 on injected
fillers and repeats and 1/300 on Disfl-QA as written, and the standing decision
is that restarts are not ours to fix. This sizes that decision: of everything a
human takes out, how much is filler, how much is repetition, and how much is
the restart we have already declined.

Disfl-QA (google-research-datasets/Disfl-QA, CC BY 4.0), dev split.

    python scripts/measure_english_disfluency_shape.py --data dev.json
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import re
from pathlib import Path

# The set the rule arm already knows, from service/polish.py.
FILLERS = {"um", "uh", "er", "erm", "uhh", "hmm", "mm", "eh", "ah", "oh"}
WORD = re.compile(r"[\w']+|[^\w\s]")


def words(text: str) -> list[str]:
    return WORD.findall(text)


def deleted(disfluent: str, original: str) -> list[tuple[int, int]]:
    a, b = words(disfluent), words(original)
    return [
        (i1, i2)
        for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        if tag in ("delete", "replace")
    ]


def classify(tokens: list[str], i1: int, i2: int, slack: int = 4) -> str:
    span = [t.lower() for t in tokens[i1:i2]]
    if not span:
        return "空"
    if all(t in FILLERS or not t.isalnum() for t in span):
        return "填充音"
    right = [t.lower() for t in tokens[i2 : i2 + len(span) + slack]]
    left = [t.lower() for t in tokens[max(0, i1 - len(span) - slack) : i1]]
    joined = " ".join(span)
    if joined in " ".join(right) or joined in " ".join(left):
        return "重复"
    if any(t in FILLERS for t in span):
        return "填充音夹着内容"
    return "改口（说了一半重来）"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    rows = list(data.values()) if isinstance(data, dict) else data
    rng = random.Random(args.seed)

    real: collections.Counter[str] = collections.Counter()
    chance: collections.Counter[str] = collections.Counter()
    for row in rows:
        tokens = words(row["disfluent"])
        for i1, i2 in deleted(row["disfluent"], row["original"]):
            real[classify(tokens, i1, i2)] += 1
            # Same length, elsewhere in the same sentence, kept by the human.
            n = i2 - i1
            if len(tokens) > n:
                j1 = rng.randrange(0, len(tokens) - n + 1)
                chance[classify(tokens, j1, j1 + n)] += 1

    total = sum(real.values())
    ctotal = sum(chance.values()) or 1
    print(f"{len(rows)} 对，{total} 个删除跨度\n")
    print(f"{'类别':<18}{'真删':>8}{'占比':>9}{'随机跨度同判':>14}")
    for name, n in real.most_common():
        print(f"{name:<18}{n:>8}{n / total:>9.1%}{chance[name] / ctotal:>14.1%}")

    doable = real["填充音"] + real["重复"]
    print(f"\n我们能做的（填充音 + 重复）{doable} 个，占 {doable / total:.1%}")
    print(
        f"已明确不做的（改口）      {real['改口（说了一半重来）']} 个，"
        f"占 {real['改口（说了一半重来）'] / total:.1%}"
    )


if __name__ == "__main__":
    main()
