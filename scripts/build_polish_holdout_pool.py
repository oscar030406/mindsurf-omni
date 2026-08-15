"""Sentences the polisher has never trained on, for a held-out set big enough to read.

The four acceptance numbers are read on 156 sentences, and 156 sentences cannot
resolve the differences the configuration table ranks by: content retention
carries a 95% interval half-width of ±0.009 there, while the table separates
arms by 0.003 to 0.006. Every arm added at that size compares noise.

The cheap fix is that the corpus already holds sentences the training pool never
took. Collected here, they become a held-out set no existing checkpoint has
seen, so the arms already on disk can be re-read on it without retraining any
of them -- which is the whole point, because retraining to grow the split would
also change the thing being measured.

Deduplicated against the training pool by the sentence itself, not by id: the
same sentence reached several probe files under different ids, and one that is
in the pool under any of them is not held out.

    python scripts/build_polish_holdout_pool.py --pool artifacts/polish_train/pool.jsonl \
        --output artifacts/polish_train/pool_holdout.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Where a sentence can hide in this repository's jsonl. Read as "any of these
# keys, if it holds a string" rather than per-file, because the probe files and
# the reply archives disagree on the name and both are worth harvesting.
TEXT_KEYS = ("text", "prompt", "reply", "reference_text", "answer", "output")

# Same bounds the training pool used: under 4 characters is not a sentence, and
# over 160 is not something a person says in one breath -- and the Thinker's 512
# tokens have to hold source and target together.
SHORTEST = 4
LONGEST = 160

SOURCES = (
    "configs/chat_refs_external_v1.jsonl",
    "configs/chat_refs_short_a_v1.jsonl",
    "configs/chat_refs_short_b_v1.jsonl",
    "configs/talker_clauses_zh_v1.jsonl",
    "configs/talker_texts_zh_v1.jsonl",
    "artifacts/asr_probe_scored.jsonl",
    "artifacts/homefield_sft_graft_768_up.jsonl",
)


def harvest(path: Path) -> list[str]:
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        for key in TEXT_KEYS:
            value = row.get(key)
            if isinstance(value, str) and SHORTEST <= len(value.strip()) <= LONGEST:
                found.append(value.strip())
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path, help="the training pool to exclude")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", nargs="*", type=Path)
    args = parser.parse_args()

    trained_on = set()
    for line in args.pool.read_text(encoding="utf-8").splitlines():
        if line.strip():
            trained_on.add(json.loads(line)["text"].strip())

    rows, seen = [], set()
    for path in args.sources or [Path(name) for name in SOURCES]:
        if not path.is_file():
            print(f"  跳过（不在磁盘上）{path}")
            continue
        taken = 0
        for text in harvest(path):
            if text in trained_on or text in seen:
                continue
            seen.add(text)
            rows.append({"id": f"hold{len(rows):06d}", "text": text, "source": path.stem})
            taken += 1
        print(f"  {taken:>5} 条来自 {path}")

    if not rows:
        raise SystemExit("一条没有：语料全在训练池里了")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(rows)} 条写到 {args.output}（训练池 {len(trained_on)} 条，零重叠）")


if __name__ == "__main__":
    main()
