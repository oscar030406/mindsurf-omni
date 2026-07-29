"""Preference triples for DPO, from samples one checkpoint drew of itself.

The action guide put DPO third because its stated precondition -- an SFT model
good enough that ranking its outputs ranks something real -- was finally met.
This builds the data half.

**On-policy, deliberately.** The blind judgements already on file are 316 pairs
with reasons, and they are the wrong shape for this: each pair is one model's
reply against another model's reply, so what they rank is checkpoints, not
responses. Tuning a third model on which of two other models a judge preferred
teaches it to imitate the winner, not to prefer its own better draft. DPO wants
several drafts from the model being tuned, ranked among themselves, which is
what `--samples` produces.

**Ties are dropped rather than split.** A pair the judge could not separate
carries no gradient direction, and keeping it as a coin flip trains toward
noise. The count of dropped ties is reported, because a set that is mostly ties
means the sampler is not producing enough variety to rank.

**Degenerate drafts are dropped before judging, not after.** A reply that looped
or came back empty would win or lose for reasons that have nothing to do with
preference, and a judge asked to rank it will produce a confident answer
anyway. The same repetition screen the chat instrument uses runs first.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.measure_chat_loss import repetition  # noqa: E402

# Above this share of repeated 8-grams a draft is looping, and its rank would
# describe the loop rather than a preference.
LOOPING = 0.5


def usable(reply: str) -> tuple[bool, str]:
    if not reply.strip():
        return False, "empty"
    if repetition(reply) > LOOPING:
        return False, "looping"
    return True, ""


def draw_pairs(drafts: list[str], per_prompt: int, rng: random.Random) -> list[tuple[int, int]]:
    """Which drafts to put in front of a judge, without judging every pair.

    All-pairs grows quadratically and buys little: the ranking a judge produces
    on a handful of drafts is dominated by the extremes. A random sample of
    distinct pairs keeps the cost linear in the number of prompts.
    """
    if len(drafts) < 2:
        return []
    everything = [(i, j) for i in range(len(drafts)) for j in range(i + 1, len(drafts))]
    rng.shuffle(everything)
    return everything[:per_prompt]


def build(args: argparse.Namespace) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.samples]
    checkpoints = {report.get("checkpoint") for report in reports}
    if len(checkpoints) != 1:
        raise SystemExit(
            f"samples come from {sorted(checkpoints)}; preference pairs have to be drawn "
            "from one checkpoint or they rank checkpoints instead of responses"
        )

    drafts: dict[str, list[str]] = collections.defaultdict(list)
    prompts: dict[str, str] = {}
    for report in reports:
        for row in report.get("replies", []):
            drafts[row["id"]].append(row["reply"])
            prompts.setdefault(row["id"], row.get("prompt", ""))

    rng = random.Random(args.seed)
    items: list[dict[str, Any]] = []
    reasons: collections.Counter[str] = collections.Counter()
    for probe_id, all_drafts in sorted(drafts.items()):
        kept = []
        for draft in all_drafts:
            ok, why = usable(draft)
            if ok:
                kept.append(draft)
            else:
                reasons[why] += 1
        if len(kept) < 2:
            reasons["too few usable drafts"] += 1
            continue
        for nth, (left, right) in enumerate(draw_pairs(kept, args.pairs_per_prompt, rng)):
            # Sides are shuffled here, once, and the key is kept separately --
            # a judge shown the same model's drafts always in the same order
            # would be ranking position.
            flip = rng.random() < 0.5
            items.append(
                {
                    # Unique per pair, not per prompt. A prompt contributes
                    # several pairs, so keying by prompt id silently collapses
                    # them and attaches a judgement to a pair the judge never
                    # saw -- half the labels land on the wrong text.
                    "key": f"{probe_id}#{nth}",
                    "id": probe_id,
                    "prompt": prompts[probe_id],
                    "left": kept[right] if flip else kept[left],
                    "right": kept[left] if flip else kept[right],
                }
            )

    keys = [item["key"] for item in items]
    if len(set(keys)) != len(keys):
        raise SystemExit("pair keys are not unique; judgements would attach to the wrong pair")

    return {
        "checkpoint": next(iter(checkpoints)),
        "prompts": len(drafts),
        "pairs": items,
        "dropped_drafts": dict(reasons),
    }


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    """Turn judged pairs into (prompt, chosen, rejected), dropping ties."""
    payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    formed = payload["pairs"]
    pairs = {item["key"]: item for item in formed}
    if len(pairs) != len(formed):
        raise SystemExit(f"{len(formed)} pairs collapsed to {len(pairs)} keys")
    judged = json.loads(args.judged.read_text(encoding="utf-8"))

    triples: list[dict[str, str]] = []
    ties = 0
    unmatched = 0
    for row in judged:
        item = pairs.get(row.get("key", row.get("id")))
        if item is None:
            unmatched += 1
            continue
        if row["winner"] == "tie":
            ties += 1
            continue
        chosen = item[row["winner"]]
        rejected = item["right" if row["winner"] == "left" else "left"]
        if chosen == rejected:
            ties += 1
            continue
        triples.append({"prompt": item["prompt"], "chosen": chosen, "rejected": rejected})

    return {
        # Carried through, not dropped. The on-policy invariant is enforced two
        # files upstream and the trainer has no other way to re-assert it: with
        # this missing, pointing --checkpoint at a different model turns the run
        # into off-policy imitation and nothing anywhere notices.
        "checkpoint": payload.get("checkpoint"),
        "triples": triples,
        "dropped_ties": ties,
        "unmatched_judgements": unmatched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pair = sub.add_parser("pairs", help="form blind pairs from repeated sampling")
    pair.add_argument(
        "--samples",
        type=Path,
        nargs="+",
        required=True,
        help="measure_chat_loss --generate reports, all from ONE checkpoint, one per sampling seed",
    )
    pair.add_argument("--pairs-per-prompt", type=int, default=2)
    pair.add_argument("--seed", type=int, default=17)
    pair.add_argument("--output", type=Path, required=True)

    done = sub.add_parser("resolve", help="turn judgements into DPO triples")
    done.add_argument("--pairs", type=Path, required=True)
    done.add_argument("--judged", type=Path, required=True, help="[{id, winner}]")
    done.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    report = build(args) if args.command == "pairs" else resolve(args)

    if args.command == "pairs":
        print(f"{report['prompts']} 条提问 -> {len(report['pairs'])} 个待判对")
        if report["dropped_drafts"]:
            print(f"  丢弃的草稿: {report['dropped_drafts']}")
    else:
        print(f"{len(report['triples'])} 个三元组，丢弃平局 {report['dropped_ties']}")
        if report["unmatched_judgements"]:
            print(f"  对不上的判决 {report['unmatched_judgements']} 条")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
