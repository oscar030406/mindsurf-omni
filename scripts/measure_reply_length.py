"""Did an intervention change how long the replies are, on shared probes.

Three rounds gated this on the median reply length of each arm, converted to
seconds at 4.67 characters per second and compared against a 1.5-second line.
That ruler has no resolution: the line is 7.0 characters wide and the bootstrap
3-sigma interval on one arm's median alone spans 8. It nonetheless decided two
rounds, passing one by 0.001 seconds and failing the next by 0.106, while the
two rounds differed by 0.02 characters when measured properly.

Properly means paired. Both arms answer the same probes, so subtracting per
probe cancels the part of the spread that is the probe rather than the model --
which is most of it. That is the same argument ``compare_paired`` is built on,
and this reuses it rather than restating it, so a length verdict comes out in
the same three words as every other axis in this project.

It reports the old median-seconds reading too. Not because it is worth
anything, but because three rounds are on record in those units and a new ruler
that cannot be lined up against the old numbers starts its own incompatible
history.

    python scripts/measure_reply_length.py \
        --candidate artifacts/dpo4/blind-dpo4-608.json \
        --reference artifacts/blind608/chat-sft_merge-608.json \
        --output artifacts/dpo4/length-dpo4.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from mindsurf_omni.evaluation.metrics import compare_paired

# The rate three rounds of reports are denominated in. Kept so the old reading
# stays comparable, not because it is defensible: it carries its own estimation
# error and the handover already says to stop converting.
CHARS_PER_SECOND = 4.67


def replies(path: Path) -> dict[str, str]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = blob.get("replies")
    if not rows:
        raise SystemExit(f"{path} 没有 replies 字段，它记的不是某个臂自己的生成")
    return {row["id"]: row["reply"] for row in rows}


def paired_deltas(candidate: dict[str, str], reference: dict[str, str]) -> list[float]:
    """Character change per shared probe, candidate minus reference."""
    shared = sorted(set(candidate) & set(reference))
    return [float(len(candidate[i]) - len(reference[i])) for i in shared]


def median_seconds_reading(
    candidate: dict[str, str], reference: dict[str, str], seed: int = 0
) -> dict[str, Any]:
    """The retired ruler, reported so the older rounds stay comparable.

    The bootstrap interval is what disqualifies it: when the noise on one arm's
    median is wider than the line the difference has to clear, the verdict is
    reading the sampler, not the model.
    """
    import random

    rng = random.Random(seed)

    def boot(values: list[int]) -> tuple[float, float]:
        draws = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(4000))
        return draws[int(4000 * 0.0015)], draws[int(4000 * 0.9985)]

    cand = [len(v) for v in candidate.values()]
    ref = [len(v) for v in reference.values()]
    cand_med, ref_med = statistics.median(cand), statistics.median(ref)
    line_chars = 1.5 * CHARS_PER_SECOND
    lo, hi = boot(ref)
    return {
        "candidate_median_chars": cand_med,
        "reference_median_chars": ref_med,
        "candidate_seconds": cand_med / CHARS_PER_SECOND,
        "reference_seconds": ref_med / CHARS_PER_SECOND,
        "moved_seconds": abs(cand_med - ref_med) / CHARS_PER_SECOND,
        "old_line_seconds": 1.5,
        "old_verdict": "过" if abs(cand_med - ref_med) / CHARS_PER_SECOND <= 1.5 else "不过",
        "eligibility": {
            "line_chars": line_chars,
            "reference_median_bootstrap_3sigma_chars": [lo, hi],
            "gating_eligible": (hi - lo) < line_chars,
            "note": "参照臂中位数自身的 3σ 宽于判据线时，判的是采样不是模型",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cand, ref = replies(args.candidate), replies(args.reference)
    deltas = paired_deltas(cand, ref)
    if not deltas:
        raise SystemExit("两份读数没有共同的探针 id，配不上对")

    mean = statistics.fmean(deltas)
    # The line this round registered: three binomial-free sigmas of the paired
    # mean. Reported beside compare_paired's bootstrap floor because the two are
    # different estimators of the same thing and a verdict that depends on which
    # one is used should say so rather than pick.
    sigma3 = 3 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    report = {
        "paired": {
            "n": len(deltas),
            "mean_char_change": mean,
            "median_char_change": statistics.median(deltas),
            "registered_line_3sigma_chars": sigma3,
            "registered_verdict": "indistinguishable" if abs(mean) <= sigma3 else "moved",
            "project_instrument": compare_paired("reply_chars", deltas, seed=args.seed),
        },
        "retired_median_seconds": median_seconds_reading(cand, ref, seed=args.seed),
        "arms": {"candidate": str(args.candidate), "reference": str(args.reference)},
    }
    p = report["paired"]
    print(
        f"  配对 n={p['n']}  每条平均 {p['mean_char_change']:+.2f} 字"
        f"  注册线 3σ ±{p['registered_line_3sigma_chars']:.2f}"
        f"  -> {p['registered_verdict']}"
    )
    print(f"  项目比较器: {p['project_instrument']}")
    old = report["retired_median_seconds"]
    print(
        f"  作废的旧读法: {old['candidate_seconds']:.3f} 秒，移动 {old['moved_seconds']:.3f}"
        f" -> {old['old_verdict']}（有判定资格: {old['eligibility']['gating_eligible']}）"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
