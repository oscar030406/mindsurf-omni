"""Preference triples that rank short over long, from samples already on disk.

The product problem is that the median reply takes half a minute to say. A
system prompt could not move it -- 30.0 to 30.9 seconds, and it tripled
fabrication on the way (docs/experiments/2026-08-01-field-comparison.md section
8). What is left is training, and the cheapest training signal is already
sitting in the sampling this project did for the last DPO round: four seeds per
prompt, whose lengths differ by 74 characters at the median.

So no card time is spent here. For each prompt this takes the shortest and the
longest of the four and, when a judge confirms the short one still answers,
emits `chosen = short, rejected = long`.

**Only the short one is judged.** The direction to learn does not depend on how
good the long one is; the only risk is a short reply with nothing in it, and
that is exactly what the screen catches. Judging both would spend twice the
judge for no extra decision.

This shares `build_preference_pairs.py`'s output shape so `train_dpo.py` reads
it unchanged, and it keeps that file's invariant: every sample must come from
the same checkpoint, because ranking across checkpoints teaches imitation
rather than preference.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# Below this, the two samples are the same length in any way a listener would
# notice, and the pair teaches nothing about brevity while still spending an
# update on whatever else differs between them.
MINIMUM_GAP_CHARS = 20


def pick(
    samples: list[dict[str, Any]], answers: dict[int, bool] | None = None
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The shortest draft that still answers, against the longest.

    Taking the shortest outright was the first design and it was wrong: screened
    over 771 prompts, only 191 of the shortest drafts still answered. Short and
    good are anti-correlated in this model's output, so "prefer the shortest"
    would mostly have taught "prefer the broken one".

    ``answers`` maps a draft's rank by length to whether a judge said it answers.
    With it, chosen is the shortest draft that passed; without it, the shortest
    non-empty one -- the second form exists only for the tests and for callers
    that have already filtered.
    """
    usable = [s for s in samples if s.get("reply", "").strip()]
    if len(usable) < 2:
        return None
    ordered = sorted(usable, key=lambda s: len(s["reply"]))

    short = None
    for rank, candidate in enumerate(ordered):
        if answers is None or answers.get(rank):
            short = candidate
            break
    if short is None:
        return None

    long = ordered[-1]
    if long is short:
        return None
    if len(long["reply"]) - len(short["reply"]) < MINIMUM_GAP_CHARS:
        return None
    return short, long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--judged",
        type=Path,
        required=True,
        help='JSON mapping prompt id to "yes"/"no": did the SHORT reply still answer',
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.samples]
    checkpoints = {report.get("checkpoint") for report in reports}
    if len(checkpoints) != 1:
        raise SystemExit(
            f"samples come from more than one checkpoint {sorted(checkpoints)}; "
            "ranking across checkpoints teaches imitation, not preference"
        )

    by_id: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        for reply in report["replies"]:
            by_id.setdefault(reply["id"], []).append(reply)

    judged = json.loads(args.judged.read_text(encoding="utf-8"))
    triples = []
    dropped = {"too_close": 0, "short_does_not_answer": 0, "unjudged": 0}
    for prompt_id, samples in sorted(by_id.items()):
        chosen = pick(samples)
        if chosen is None:
            dropped["too_close"] += 1
            continue
        verdict = judged.get(prompt_id)
        if verdict is None:
            dropped["unjudged"] += 1
            continue
        if verdict != "yes":
            dropped["short_does_not_answer"] += 1
            continue
        short, long = chosen
        triples.append(
            {
                "id": prompt_id,
                "prompt": short["prompt"],
                "chosen": short["reply"],
                "rejected": long["reply"],
            }
        )

    if not triples:
        raise SystemExit("no triples survived; nothing to train on")

    saved = [len(t["rejected"]) - len(t["chosen"]) for t in triples]
    payload = {
        "checkpoint": next(iter(checkpoints)),
        "objective": "shorter reply preferred, conditional on it still answering",
        "triples": triples,
        "dropped": dropped,
        "median_chars_saved": statistics.median(saved),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(triples)} 个三元组，丢弃 {dropped}")
    print(f"  chosen 比 rejected 短 {statistics.median(saved):.0f} 字（中位）")
    print(f"写入 {args.output}")


if __name__ == "__main__":
    main()
