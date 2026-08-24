"""Does the polish stage ever delete a negation that was not a repetition.

Driving the service on real conversation turned up 人也没了 coming back as
人也了 within the first eight clips. Character-level content retention cannot
see that -- one character out of eleven sits well inside the noise of every
reading we take -- and what it costs is the meaning, which inverts.

The ruler took three tries and the two failures are worth keeping:

* Counting negations in the transcript and in the output made 不好不好意思 →
  不好意思 a lost negation. That is a repetition removed correctly.
* Counting against the human's transcript instead did the same thing, because
  RAMC's transcript is verbatim and the human's repetitions are in it too.

So the test is positional and carries its own exemption: a negation counts as
lost when the character was deleted **and** the deleted span is not a repeat of
text beside it. The exemption is written here and nowhere else -- the last time
a patch and its ruler shared one, both were wrong in the same direction and
agreed with each other.

    python scripts/measure_negation_survival.py --eval artifacts/eval_arm.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

# Single characters only. 不 没 别 无 非 莫 勿 甭 flip a clause on their own;
# multi-character negations are built from these.
NEGATIONS = set("不没别无非莫勿甭")

# 没 negates in 没有 and 没关系. Where it does not is 沉没 / 淹没 / 埋没, so the
# character before it decides.
NOT_NEGATION_AFTER = set("沉淹埋出湮")

# 别 is the awkward one: it negates in 别去 and 别动, and does not in 特别,
# 分别, 差别, 个别, 区别, 级别, 性别, 告别, 送别, 派别. The first version of
# this script reported exactly one lost negation on the holdout and it was 特别.
BIE_IS_NOT_NEGATION_AFTER = set("特分差个区级性告送派类识")

# How far apart the two halves of a repetition may sit: a comma, a breath, a
# filler. 不再不再 is adjacent; 没睡，没睡 is not.
SLACK = 4


def negation_positions(text: str) -> list[int]:
    out = []
    for i, ch in enumerate(text):
        if ch not in NEGATIONS:
            continue
        if i and text[i - 1] in NOT_NEGATION_AFTER:
            continue
        if ch == "别" and i and text[i - 1] in BIE_IS_NOT_NEGATION_AFTER:
            continue
        out.append(i)
    return out


def deleted_spans(source: str, output: str) -> list[tuple[int, int]]:
    matcher = difflib.SequenceMatcher(None, source, output, autojunk=False)
    return [(i1, i2) for tag, i1, i2, _, _ in matcher.get_opcodes() if tag in ("delete", "replace")]


def is_repeat(source: str, i1: int, i2: int) -> bool:
    """Is the cut text sitting there a second time, just beside it?"""
    core = source[i1:i2].strip("，。！？、 ")
    if not core:
        return False
    if source[i2 : i2 + len(core)] == core:
        return True
    right = source[i2 : i2 + len(core) + SLACK]
    left = source[max(0, i1 - len(core) - SLACK) : i1]
    return core in right or core in left


def losses(source: str, output: str) -> list[int]:
    """Positions of negations the arm removed for no repetition reason."""
    out = []
    wanted = set(negation_positions(source))
    for i1, i2 in deleted_spans(source, output):
        if is_repeat(source, i1, i2):
            continue
        out.extend(i for i in range(i1, i2) if i in wanted)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = {}
    for path in args.eval:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        present = lost = affected = sentences = 0
        examples = []
        for row in rows:
            source = row.get("source", "")
            output = row.get("polished") or row.get("output") or ""
            if not source or not output:
                continue
            sentences += 1
            here = losses(source, output)
            present += len(negation_positions(source))
            lost += len(here)
            if here:
                affected += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "source": source[:90],
                            "output": output[:90],
                            "at": [source[max(0, i - 4) : i + 4] for i in here[:3]],
                        }
                    )
        rate = f"{1 - lost / present:.4f}" if present else "n/a"
        report[path.stem] = {
            "sentences": sentences,
            "negations": present,
            "deleted_and_not_a_repeat": lost,
            "survival": round(1 - lost / present, 5) if present else None,
            "sentences_affected": affected,
            "examples": examples,
        }
        print(
            f"{path.stem:26s} 否定词 {present:5d}  非重复地删掉 {lost:3d}  "
            f"存活 {rate}  涉及 {affected}/{sentences} 句"
        )

    for name, block in report.items():
        for eg in block["examples"][:3]:
            print(f"\n  [{name}] 删在 {eg['at']}")
            print(f"    转写 {eg['source']}")
            print(f"    输出 {eg['output']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
