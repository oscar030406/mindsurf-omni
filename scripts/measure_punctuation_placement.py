"""Punctuation restoration scored the way the task is normally scored.

The field's formulation is sequence labelling: every token carries the mark that
follows it or NONE, and the reading is per-class precision / recall / F1 over
those labels. That is what IWSLT's punctuation task and punctuator2 report, and
sklearn's classification_report is the reference implementation, so this does
not hand-roll one.

Two things are ours and are marked as ours:

* **Alignment.** The usual setup scores a model against the same tokens it was
  given. Here the hypothesis comes out of a recogniser that gets 18% of the
  characters wrong, so the two sides do not share a token sequence and have to
  be aligned before labels can be compared. Reference characters the recogniser
  lost are excluded -- a mark cannot be placed on a character nobody wrote.
* **A chance row.** On real speech the recogniser scores 0.5652 recall against
  0.3851 for its own marks thrown at random, so a bare recall number here is
  mostly the tolerance and the density. Every table gets the random row.

    python scripts/measure_punctuation_placement.py \
        --eval artifacts/eval_graded.jsonl --report artifacts/punc-placement.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
from pathlib import Path

from sklearn.metrics import classification_report

# Grouped the way the task groups them: what matters to a reader is whether the
# clause broke, the sentence ended, or it was a question -- not which of 、 and
# ，was used.
CLASSES = {
    "，": "COMMA",
    ",": "COMMA",
    "、": "COMMA",
    "；": "COMMA",
    ";": "COMMA",
    "：": "COMMA",
    ":": "COMMA",
    "。": "PERIOD",
    ".": "PERIOD",
    "？": "QUESTION",
    "?": "QUESTION",
    "！": "EXCLAM",
    "!": "EXCLAM",
}
NONE = "NONE"
LABELS = ["COMMA", "PERIOD", "QUESTION", "EXCLAM"]


def label_sequence(text: str) -> tuple[str, list[str]]:
    """Bare characters, and the label that follows each of them."""
    chars: list[str] = []
    labels: list[str] = []
    for ch in text:
        if ch in CLASSES:
            if labels:
                labels[-1] = CLASSES[ch]
        elif not ch.isspace():
            chars.append(ch)
            labels.append(NONE)
    return "".join(chars), labels


def aligned(reference: str, hypothesis: str) -> tuple[list[str], list[str]]:
    """Label pairs for the characters both sides actually wrote."""
    ref_chars, ref_labels = label_sequence(reference)
    hyp_chars, hyp_labels = label_sequence(hypothesis)
    true: list[str] = []
    pred: list[str] = []
    matcher = difflib.SequenceMatcher(None, ref_chars, hyp_chars, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            true.append(ref_labels[block.a + k])
            pred.append(hyp_labels[block.b + k])
    return true, pred


def scatter(text: str, rng: random.Random) -> str:
    """Same characters, same number of marks, positions drawn at random."""
    chars, labels = label_sequence(text)
    marks = [label for label in labels if label != NONE]
    if not chars or not marks:
        return chars
    rng.shuffle(marks)
    spots = sorted(rng.sample(range(len(chars)), min(len(marks), len(chars))))
    back = {"COMMA": "，", "PERIOD": "。", "QUESTION": "？", "EXCLAM": "！"}
    out = list(chars)
    for offset, (spot, label) in enumerate(zip(spots, marks, strict=False)):
        out.insert(spot + 1 + offset, back[label])
    return "".join(out)


def table(rows: list[dict], reference_key: str, hypothesis_key: str) -> dict:
    true: list[str] = []
    pred: list[str] = []
    for row in rows:
        t, p = aligned(row[reference_key], row[hypothesis_key])
        true.extend(t)
        pred.extend(p)
    report = classification_report(true, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }
        for label in LABELS
        if report[label]["support"]
    } | {
        "micro_f1": round(report["micro avg"]["f1-score"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "marks_written": sum(p != NONE for p in pred),
        "marks_wanted": sum(t != NONE for t in true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.eval.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row.setdefault("polished", row.get("output", ""))
    rows = [r for r in rows if r.get("target") and r.get("polished")]

    rng = random.Random(args.seed)
    for row in rows:
        row["_scattered"] = scatter(row["polished"], rng)

    out = {
        "n": len(rows),
        "臂": table(rows, "target", "polished"),
        "地板：照抄转写的标点": table(rows, "target", "source"),
        "随机：本臂自己的标点撒开": table(rows, "target", "_scattered"),
        "天花板：目标对自己": table(rows, "target", "target"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
