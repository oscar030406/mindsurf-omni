"""Compare two checkpoints on chat likelihood, across both neutral authors at once.

This comparison has decided three DPO rounds and it was run by hand every time:
pair one author, pair the other, read the two verdicts. That worked until an
intervention made the two authors disagree in direction, and only then did
anyone ask whether they had ever agreed. They had not -- the round whose
aggregates matched agreed on 59% of individual probes, barely over a coin.

So the check is in the tool now. It reports both authors, and before either
verdict is allowed to gate it asks whether they rank the same probes the same
way beyond chance. When they do not, both numbers are printed and neither
decides, which is what `assess` already does one axis over for resolution.

Why they can disagree: `chat_nll` measures how likely a checkpoint finds one
author's exact replies, and that includes how long those replies are. Within one
author, the length-tuned arm's per-probe delta tracks the reference's own length
at r = +0.62. Any change to the model's verbosity therefore moves the score in
whichever direction that particular reference sits -- so the metric cannot
cleanly judge an intervention aimed at length.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import (  # noqa: E402
    compare_paired,
    cross_reference_agreement,
)


def deltas(candidate: Path, reference: Path) -> tuple[list[str], list[float], dict[str, Any]]:
    """Per-probe nll difference on the probes both reports scored."""
    new = json.loads(candidate.read_text(encoding="utf-8"))
    old = json.loads(reference.read_text(encoding="utf-8"))
    a = {row["id"]: row for row in new["samples"]}
    b = {row["id"]: row for row in old["samples"]}
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(f"{candidate.name} and {reference.name} share no probes")
    return (
        shared,
        [a[i]["nll"] - b[i]["nll"] for i in shared],
        {"candidate": new, "reference": old},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, nargs=2, required=True, metavar=("UP", "EXT"))
    parser.add_argument("--reference", type=Path, nargs=2, required=True, metavar=("UP", "EXT"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {"authors": {}}
    collected: dict[str, tuple[list[str], list[float]]] = {}
    for author, candidate, reference in zip(
        ("up", "ext"), args.candidate, args.reference, strict=True
    ):
        shared, difference, payload = deltas(candidate, reference)
        collected[author] = (shared, difference)
        verdict = compare_paired(f"chat_nll[{author}]", difference)
        report["authors"][author] = {
            "candidate": payload["candidate"].get("checkpoint"),
            "reference": payload["reference"].get("checkpoint"),
            "candidate_chat_nll": payload["candidate"].get("chat_nll"),
            "reference_chat_nll": payload["reference"].get("chat_nll"),
            "candidate_entropy": payload["candidate"].get("entropy"),
            "reference_entropy": payload["reference"].get("entropy"),
            "verdict": verdict,
            "n": len(shared),
        }
        print(f"{author}: {verdict}")

    # Both authors scored the same probe ids, so the two delta lists line up by
    # position only after intersecting them -- an author that dropped a probe
    # would otherwise shift every pair after it.
    common = sorted(set(collected["up"][0]) & set(collected["ext"][0]))
    index_up = {probe: i for i, probe in enumerate(collected["up"][0])}
    index_ext = {probe: i for i, probe in enumerate(collected["ext"][0])}
    agreement = cross_reference_agreement(
        [collected["up"][1][index_up[p]] for p in common],
        [collected["ext"][1][index_ext[p]] for p in common],
    )
    report["cross_reference_agreement"] = agreement

    share = agreement["agreement"]
    print(
        f"\n两个作者逐条同号 {share:.1%}（n={agreement['n']}，噪声 ±{agreement['noise']:.1%}）"
        if share is not None
        else "\n作者间一致性：样本不足"
    )
    if agreement["gating_eligible"]:
        print("  两个作者排序一致到可以判定——上面的判定作数")
    else:
        print("  ⚠ **两个作者逐条不一致，超不出随机**——上面两条都只作报告，谁都不判定")
        print("  这把尺子对参考回复的长度敏感，改长度的干预它判不了")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n报告 {args.output}")


if __name__ == "__main__":
    main()
