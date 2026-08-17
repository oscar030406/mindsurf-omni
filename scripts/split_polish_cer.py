"""What the polisher's CER is made of, and how much of it the model can move.

CER is one number covering four different things, and they have four different
owners. Deciding where to push on the criterion that is still short of its line
without splitting it first is picking a direction by guessing.

The split, on the aligned edit operations rather than on the totals so the parts
add back up to the arm's CER exactly:

``filler_left``
    A character in the output the target does not have, and it belongs to a
    vocabulary filler. The polisher was supposed to delete it and did not. This
    is the same failure ``filler_removed`` reports, priced in CER.

``misspelled_filler``
    Same shape, and the character is one of the spellings the recogniser uses
    for a filler it did not write literally -- 呃 arriving as 饿 恶 啊 遏, 嗯 as
    恩 温. The decoder's door matches the vocabulary literally, so it never
    opens on these. Split out from ``kept_other`` because the first version of
    this file guessed that class was "repetition and restarts mostly" and the
    examples said otherwise; the guess would have sent the next round at the
    tagger when the target is the vocabulary.

``kept_other``
    Same shape and neither of the above. Repetition, restarts, and whatever the
    recogniser inserted that nothing else explains.

``native_filler_deleted``
    A character the target has and the output does not, and it belongs to a
    vocabulary filler. **The target keeps the corpus's own filler** -- only the
    injected spans were stripped when the targets were built -- so deleting a
    native 那个 is charged here as an error. For a dictation product it is
    usually the right edit, which makes this the share of the gap that is not
    worth closing. Naming it is the point of the split.

``content_deleted``
    Same shape, not filler. Real over-deletion, the failure ``content_kept``
    exists for.

``substituted``
    The target says one character and the output says another. Under the copy
    constraint the polisher cannot substitute, so every one of these is the
    recogniser's: 慕田峪 heard as 牧田御. DECISIONS 16 gave these up on purpose,
    and no amount of polish training reaches them.

``numeral``
    Any of the above whose character is a digit, counted apart. cn2an folds
    四万 to 40000 and leaves 4万 alone, so the two sides can disagree on a
    number both of them got right -- the fold is a correction, not a bijection.
    These belong to the ruler rather than to the model, and reading them as
    over-deletion is how 40000 turns into evidence about the polisher.

Run it against the ceiling too. The ceiling is the best a deletion-only polisher
could do, so its share of each class is the floor of that class -- what is left
after subtracting it is what the model can actually move::

    python scripts/split_polish_cer.py \\
        --arm service=artifacts/polish_train/val_service_words.jsonl \\
        --arm ceiling=artifacts/polish_train/val_bigholdout_ceiling.jsonl \\
        --report artifacts/polish-cer-split-2026-08-16.json

Segmentation is by literal vocabulary match, which is a judgement and not ground
truth: 那个 inside 那个人 is content, and this counts the two characters as
filler. Same caveat class as jieba in ``measure_word_cuts`` -- a rate to compare
arms with, not a defect count to trust absolutely.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.evaluation.metrics import normalise_for_cer  # noqa: E402
from mindsurf_omni.service.polish import (  # noqa: E402
    _WORTH_A_LOOK,
    BRIDGING_FILLERS,
    LEADING_FILLERS,
    RECOGNISED_FILLERS,
)
from scripts.measure_polish import FOLD_NUMERALS  # noqa: E402

# What the decoder's door opens on, matched literally.
VOCABULARY = tuple(
    sorted((*LEADING_FILLERS, *BRIDGING_FILLERS, *RECOGNISED_FILLERS), key=len, reverse=True)
)
# The same words plus how the recogniser actually spells them. The service
# already keeps this list for deciding whether a piece is worth a model call;
# reused here so the two cannot disagree about what counts as a filler.
MISSPELLINGS = tuple(sorted(set(_WORTH_A_LOOK) - set(VOCABULARY), key=len, reverse=True))
CLASSES = (
    "filler_left",
    "misspelled_filler",
    "kept_other",
    "native_filler_deleted",
    "content_deleted",
    "substituted",
    "numeral",
)


def spans_of(text: str, words: tuple[str, ...]) -> list[bool]:
    """True where a character sits inside a literal occurrence of one of ``words``."""
    mask = [False] * len(text)
    for word in words:
        start = text.find(word)
        while start != -1:
            for index in range(start, start + len(word)):
                mask[index] = True
            start = text.find(word, start + 1)
    return mask


def filler_mask(text: str) -> list[bool]:
    """True where a character sits inside a literal vocabulary filler."""
    return spans_of(text, VOCABULARY)


def operations(reference: str, hypothesis: str) -> list[tuple[str, int, int]]:
    """Levenshtein's own edit script: (kind, reference index, hypothesis index).

    difflib would be faster and is not the same measurement -- it models a
    replacement as a delete plus an insert, so the counts would not add back up
    to the distance CER is computed from. The parts summing to the whole is the
    only reason this split can be trusted.
    """
    rows, columns = len(reference), len(hypothesis)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        distance[i][0] = i
    for j in range(columns + 1):
        distance[0][j] = j
    for i in range(1, rows + 1):
        row, previous = distance[i], distance[i - 1]
        source = reference[i - 1]
        for j in range(1, columns + 1):
            row[j] = (
                previous[j - 1]
                if source == hypothesis[j - 1]
                else 1 + min(previous[j - 1], previous[j], row[j - 1])
            )

    script: list[tuple[str, int, int]] = []
    i, j = rows, columns
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            script.append(("substitute", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            script.append(("delete", i - 1, j))
            i -= 1
        else:
            script.append(("insert", i, j - 1))
            j -= 1
    return script


def classify(target: str, polished: str) -> tuple[Counter[str], int, list[tuple[str, str]]]:
    """Class counts, the reference length they are charged against, and examples."""
    reference = normalise_for_cer(target, fold_numbers=FOLD_NUMERALS)
    hypothesis = normalise_for_cer(polished, fold_numbers=FOLD_NUMERALS)
    counts: Counter[str] = Counter()
    examples: list[tuple[str, str]] = []
    if not reference:
        return counts, 0, examples

    reference_filler = filler_mask(reference)
    hypothesis_filler = filler_mask(hypothesis)
    hypothesis_misspelled = spans_of(hypothesis, MISSPELLINGS)
    for kind, i, j in operations(reference, hypothesis):
        if kind == "substitute":
            character = reference[i]
            name = "substituted"
            shown = f"{reference[i]} -> {hypothesis[j]}"
        elif kind == "delete":
            character = reference[i]
            name = "native_filler_deleted" if reference_filler[i] else "content_deleted"
            shown = f"删掉 {reference[max(0, i - 3) : i + 4]}（{reference[i]}）"
        else:
            character = hypothesis[j]
            if hypothesis_filler[j]:
                name = "filler_left"
            elif hypothesis_misspelled[j]:
                name = "misspelled_filler"
            else:
                name = "kept_other"
            shown = f"留下 {hypothesis[max(0, j - 3) : j + 4]}（{hypothesis[j]}）"
        # Checked last so it wins: a digit disagreement is the fold's, whichever
        # shape the alignment gave it.
        if character.isdigit():
            name = "numeral"
        counts[name] += 1
        examples.append((name, shown))
    return counts, len(reference), examples


def measure(path: Path, sample: int) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    shares: dict[str, list[float]] = {name: [] for name in CLASSES}
    totals: Counter[str] = Counter()
    cers: list[float] = []
    examples: dict[str, list[str]] = {name: [] for name in CLASSES}
    for row in rows:
        counts, length, found = classify(row["target"], row["polished"])
        if not length:
            continue
        cers.append(sum(counts.values()) / length)
        for name in CLASSES:
            shares[name].append(counts[name] / length)
        totals.update(counts)
        for name, shown in found:
            if len(examples[name]) < sample:
                examples[name].append(shown)
    return {
        "n": len(cers),
        "cer": round(statistics.mean(cers), 4) if cers else 0.0,
        "share": {name: round(statistics.mean(values), 4) for name, values in shares.items()},
        "edits": {name: totals[name] for name in CLASSES},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--floor", default="ceiling", help="which arm is the unavoidable share")
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    report: dict[str, Any] = {}
    for entry in arguments.arm:
        name, _, path = entry.partition("=")
        report[name] = measure(Path(path), arguments.sample)
        row = report[name]
        print(f"\n===== {name}  n={row['n']}  CER {row['cer']:.4f} =====")
        for label in CLASSES:
            print(f"  {label:<24} {row['share'][label]:.4f}  ({row['edits'][label]} 处)")

    floor = report.get(arguments.floor)
    if floor is not None and len(report) > 1:
        print(f"\n===== 减去「{arguments.floor}」之后，模型真正能动的 =====")
        movable = {}
        for name, row in report.items():
            if name == arguments.floor:
                continue
            movable[name] = {
                label: round(row["share"][label] - floor["share"][label], 4) for label in CLASSES
            }
            print(f"\n  {name}  可动合计 {row['cer'] - floor['cer']:.4f}")
            for label in CLASSES:
                print(f"    {label:<24} {movable[name][label]:+.4f}")
        report["movable"] = movable

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {arguments.report}")


if __name__ == "__main__":
    main()
