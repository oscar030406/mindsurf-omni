"""What humans delete, and whether our injector makes the same shapes.

Tonight's 那/就 was found by hand: the protection list held only the words the
injector knew, so the holdout set could never show what was missing. This asks
the same question exhaustively, against CS2W's human annotation -- first what
share of real deletions the vocabulary can reach at all, then, for the part it
cannot, whether our training pairs carry repetitions shaped like the real ones.

    python scripts/measure_repetition_shape.py
"""

from __future__ import annotations

import collections
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.service.polish import DOUBLE_DUTY  # noqa: E402

# The interjections both arms already treat as removable on sight. Kept beside
# DOUBLE_DUTY rather than imported from it: those are the words that are only
# sometimes filler, these are the ones that always are.
ALWAYS_FILLER = ("嗯", "呃", "啊", "哦", "呢", "吧", "哎", "唉", "诶", "呐", "嘛", "喔")

# How far apart the two halves of a repetition may sit and still be one: a
# comma, a breath, a filler. Humans repeat across small interruptions -- "他演的
# 他演的" is adjacent, "但是最后，但是最后" is not.
SLACK = 4

CS2W = Path("artifacts/polish_train/pairs_cs2w.jsonl")
OURS = Path("artifacts/polish_train/pairs_v3.jsonl")


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def deletions(row: dict):
    """Every contiguous run the annotator removed, with its surroundings."""
    source, target = row["source"], row["target"]
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        # A replacement counts: under the copy constraint an arm cannot
        # substitute, so what the matcher pairs as a replacement is a deletion.
        if tag in ("delete", "replace"):
            yield source, i1, i2


def reachable(span: str) -> bool:
    core = span.strip("，。！？、 ")
    if not core:
        return True
    return any(word in core for word in (*DOUBLE_DUTY, *ALWAYS_FILLER))


def repeated(source: str, i1: int, i2: int) -> str | None:
    """'adjacent', 'gapped', or None -- is the cut text there a second time?"""
    core = source[i1:i2].strip("，。！？、 ")
    if not core:
        return None
    if source[i2 : i2 + len(core)] == core:
        return "adjacent"
    right = source[i2 : i2 + len(core) + SLACK]
    left = source[max(0, i1 - len(core) - SLACK) : i1]
    return "gapped" if core in right or core in left else None


def coverage(pairs: list[dict]) -> dict:
    buckets: collections.Counter[str] = collections.Counter()
    for row in pairs:
        for source, i1, i2 in deletions(row):
            span = source[i1:i2]
            if reachable(span):
                buckets["词表够得着"] += 1
            elif not span.strip("，。！？、 "):
                buckets["只是标点"] += 1
            elif repeated(source, i1, i2):
                buckets["重复"] += 1
            elif len(span.strip("，。！？、 ")) == 1:
                buckets["单字非重复"] += 1
            else:
                buckets["其他多字"] += 1
    return dict(buckets)


def shape(pairs: list[dict]) -> dict:
    lengths: collections.Counter[int] = collections.Counter()
    where: collections.Counter[str] = collections.Counter()
    words: collections.Counter[str] = collections.Counter()
    for row in pairs:
        for source, i1, i2 in deletions(row):
            kind = repeated(source, i1, i2)
            if kind is None:
                continue
            core = source[i1:i2].strip("，。！？、 ")
            lengths[len(core)] += 1
            where[kind] += 1
            words[core] += 1
    total = sum(where.values()) or 1
    return {
        "total": sum(where.values()),
        "adjacent_share": round(where["adjacent"] / total, 3),
        "gapped_share": round(where["gapped"] / total, 3),
        "one_char_share": round(lengths[1] / total, 3),
        "two_char_share": round(lengths[2] / total, 3),
        "three_or_more_share": round(sum(n for c, n in lengths.items() if c >= 3) / total, 3),
        "most_common": words.most_common(8),
    }


def main() -> None:
    human = rows(CS2W)
    buckets = coverage(human)
    total = sum(buckets.values())
    print(f"CS2W {len(human)} 对，删除跨度 {total} 次")
    for name, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {n:6d}  {n / total:6.1%}  {name}")

    print("\n重复的形状——我们注入的对真人的：")
    for name, path in (("我们 (pairs_v3)", OURS), ("真人 (CS2W)", CS2W)):
        s = shape(rows(path))
        print(
            f"  {name}: {s['total']} 次，"
            f"1 字 {s['one_char_share']:.0%} / 2 字 {s['two_char_share']:.0%} / "
            f"3 字以上 {s['three_or_more_share']:.0%}，隔开的 {s['gapped_share']:.0%}"
        )
        print(f"    最常见 {', '.join(f'{w}×{n}' for w, n in s['most_common'])}")


if __name__ == "__main__":
    main()
