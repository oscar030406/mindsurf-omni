"""Does the reply answer the question? A conversation guardrail that ignores length.

`chat_nll` cannot judge a length intervention. It scores how closely a
checkpoint's distribution matches one author's way of writing, and a way of
writing includes how long it is, so an intervention aimed at length changes the
number in a direction the reference set picks rather than the model does
(docs/experiments/2026-08-01-length-dpo.md section 6.2.2). Stratifying by
reference length was tried and falsified in section 7.3, and replacing the
guardrail with blind preference collapses it into criterion B.

So this asks a different question, one length cannot answer for: did the reply
address what was asked. The wording comes from the filter that built the
preference data, which already had to be neutral about length for the same
reason and states so in the prompt.

**The criteria below are registered before the instrument is run.** They decide
whether it may gate at all, not whether any checkpoint passes. PROJECT_RULES
section 6 requires invariance, discrimination, noise floor and sample size
before a measurement is allowed to judge, and this project has paid for skipping
that step more than once.

    1. Invariance: judging the same replies twice agrees on at least 95% of
       probes. Below that the instrument's own flicker is larger than the effect
       it would be reading.
    2. Discrimination: a shuffled control -- every prompt paired with a
       different probe's reply -- must score at most 0.30 in absolute terms and
       must sit resolvably below the intact arm. A judge that waves through
       answers to other questions is not reading.
    3. Resolution: the paired comparison's floor must satisfy 3 x noise <= 0.05.
       Five points is the effect of interest, the same order as every other
       declared line here, and section 2.1 measured differences of that size
       between length ranks.
    4. Any of the three failing means this may be reported and may not gate. It
       is allowed to fail. Writing a criterion that can only pass is how the
       last two attempts at fixing C wasted a round.

    python scripts/answerable_rate.py --validate base=artifacts/chat-...-ext.json
    python scripts/answerable_rate.py --arm base=... --arm length=... --output ...
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.judge import Judge  # noqa: E402
from mindsurf_omni.evaluation.metrics import compare_paired  # noqa: E402

# Same wording as the filter that built the preference triples, and neutral
# about length for the same reason: a guardrail that rewards long answers would
# fail a length intervention for being a length intervention.
PROMPT = """判断下面这个回答有没有**答上**这个问题。

只看实质：有没有回应被问的东西、信息对不对、能不能用。
**简短不是缺点，冗长也不是优点。** 一句话答到点上就是答上了，
长篇大论绕开问题就是没答上。承认「我查不到」也算答上——那是诚实的回应。

问题：{prompt}

回答：{reply}

只回一个字：是 或 否。"""

EFFECT_OF_INTEREST = 0.05
INVARIANCE_FLOOR = 0.95
SHUFFLED_CEILING = 0.30


def load_replies(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    replies = blob.get("replies")
    if not replies:
        raise SystemExit(f"{path} 没有 replies 字段")
    return sorted(replies, key=lambda row: row["id"])


def answered(answer: str) -> bool:
    """A one-character verdict, defaulting to "no" on anything unexpected.

    Defaulting to yes would let a malformed reply inflate the rate, which is the
    direction that flatters whatever is being measured.
    """
    return answer.strip().startswith("是")


def judge_rows(judge: Judge, rows: list[dict[str, Any]], label: str) -> list[bool]:
    answers = judge.run(rows, lambda row: PROMPT.format(**row), label=label)
    return [answered(answer) for answer in answers]


def shuffled(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Every prompt with someone else's reply, and nobody with their own."""
    rng = random.Random(seed)
    offsets = list(range(1, len(rows)))
    shift = rng.choice(offsets)
    return [
        {**row, "reply": rows[(index + shift) % len(rows)]["reply"]}
        for index, row in enumerate(rows)
    ]


def binomial_noise(n: int) -> float:
    return 1.96 * math.sqrt(0.25 / n) if n else float("inf")


def validate(judge: Judge, label: str, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    intact = judge_rows(judge, rows, f"{label} 完整")
    again = judge_rows(judge, rows, f"{label} 重判")
    control = judge_rows(judge, shuffled(rows, seed), f"{label} 打乱")

    agreement = sum(1 for a, b in zip(intact, again, strict=True) if a == b) / len(rows)
    intact_rate = sum(intact) / len(rows)
    control_rate = sum(control) / len(rows)
    separation = intact_rate - control_rate
    noise = binomial_noise(len(rows))

    # The floor that matters for gating is the paired one, and the pair that
    # will be compared is two arms on the same probes. Re-judging the same arm
    # is the tightest available stand-in: whatever it cannot resolve between two
    # readings of one arm, it cannot resolve between two arms either.
    deltas = [float(b) - float(a) for a, b in zip(intact, again, strict=True)]
    paired = compare_paired(
        "answerable_rate(重判自比)", deltas, lower_is_better=False,
        effect_of_interest=EFFECT_OF_INTEREST,
    )
    resolves = float(paired.split("±")[1].split(",")[0]) if "±" in paired else float("inf")

    checks = {
        "invariance": agreement >= INVARIANCE_FLOOR,
        "discrimination": control_rate <= SHUFFLED_CEILING and separation > 3 * noise,
        "resolution": resolves <= EFFECT_OF_INTEREST,
    }
    return {
        "arm": label,
        "n": len(rows),
        "intact_rate": intact_rate,
        "rejudge_agreement": agreement,
        "shuffled_rate": control_rate,
        "separation": separation,
        "binomial_noise": noise,
        "paired_self_comparison": paired,
        "resolves": resolves,
        "checks": checks,
        "gating_eligible": all(checks.values()),
        "registered": {
            "invariance_floor": INVARIANCE_FLOOR,
            "shuffled_ceiling": SHUFFLED_CEILING,
            "effect_of_interest": EFFECT_OF_INTEREST,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="LABEL=PATH")
    parser.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    judge = Judge(credentials=args.credentials, workers=args.workers)
    print(f"判官 {judge.model} @ {judge.base_url}")

    def rows_of(spec: str) -> tuple[str, list[dict[str, Any]]]:
        label, _, path = spec.partition("=")
        rows = load_replies(Path(path))
        return label, rows[: args.limit] if args.limit else rows

    if args.validate:
        label, rows = rows_of(args.validate)
        report = validate(judge, label, rows, args.seed)
        report["judge"] = judge.provenance(PROMPT, seed=args.seed)
        for key in ("n", "intact_rate", "rejudge_agreement", "shuffled_rate", "separation"):
            print(f"  {key}: {report[key]}")
        print(f"  {report['paired_self_comparison']}")
        for name, passed in report["checks"].items():
            print(f"  {name}: {'过' if passed else '不过'}")
        print("有门控资格" if report["gating_eligible"] else "**仅报告**——不得用于判定 C")
        if args.output:
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return 0

    if len(args.arm) != 2:
        raise SystemExit("要正好两个 --arm，或者用 --validate")
    (base_label, base_rows), (arm_label, arm_rows) = (rows_of(spec) for spec in args.arm)
    shared = sorted({row["id"] for row in base_rows} & {row["id"] for row in arm_rows})
    base_by_id = {row["id"]: row for row in base_rows}
    arm_by_id = {row["id"]: row for row in arm_rows}

    base = judge_rows(judge, [base_by_id[i] for i in shared], base_label)
    arm = judge_rows(judge, [arm_by_id[i] for i in shared], arm_label)
    deltas = [float(b) - float(a) for a, b in zip(base, arm, strict=True)]
    verdict = compare_paired(
        f"answerable_rate[{arm_label} vs {base_label}]",
        deltas,
        lower_is_better=False,
        effect_of_interest=EFFECT_OF_INTEREST,
    )
    report = {
        "n": len(shared),
        f"{base_label}_rate": sum(base) / len(base),
        f"{arm_label}_rate": sum(arm) / len(arm),
        "verdict": verdict,
        "judge": judge.provenance(PROMPT, seed=args.seed),
    }
    for key in ("n", f"{base_label}_rate", f"{arm_label}_rate"):
        print(f"  {key}: {report[key]}")
    print(f"  {verdict}")
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
