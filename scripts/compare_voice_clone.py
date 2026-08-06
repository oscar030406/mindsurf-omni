"""Did an intervention move voice cloning, against a checkpoint it should match.

``measure_voice_clone.py`` scores one checkpoint. Comparing two of them was
done by hand every round it mattered, which is why the numbers quoted in three
write-ups cannot be recomputed from anything in this repository. This is that
comparison, written down.

The statistic is not obvious and the wrong one changes the verdict. Twenty
clips of one voice do not carry twenty voices' worth of evidence: they move
together, and a per-row noise floor counts that correlation as independent
support. ``compare_paired_clustered`` resamples voices instead of clips, and
its own docstring records the case where the two disagreed -- the
identification margin reads ``improved`` per row and ``indistinguishable``
per voice. So this pairs by clip id, groups the differences by voice, and
takes the floor from the groups.

    python scripts/compare_voice_clone.py \
        --candidate artifacts/dpo5/clone-dpo5.json \
        --reference artifacts/merge/clone-merge.json \
        --output artifacts/dpo5/D2-dpo5-vs-merge.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mindsurf_omni.evaluation.metrics import compare_paired_clustered

# The clone effect this project cares about, carried over from
# measure_voice_clone.py's own identification effect_of_interest.
SIMILARITY_EFFECT = 0.05
HIT_EFFECT = 0.10


def samples(path: Path) -> dict[str, dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = blob.get("samples")
    if not rows:
        raise SystemExit(f"{path} 没有 samples 字段，它记的不是逐条读数")
    return {row["id"]: row for row in rows}


def voice_of(sample_id: str) -> str:
    """``arthur/zh000`` -> ``arthur``. The clip id carries its own cluster."""
    return sample_id.split("/", 1)[0]


def grouped_deltas(
    candidate: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]], field: str
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for clip in sorted(set(candidate) & set(reference)):
        groups[voice_of(clip)].append(float(candidate[clip][field]) - float(reference[clip][field]))
    return dict(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cand, ref = samples(args.candidate), samples(args.reference)
    shared = sorted(set(cand) & set(ref))
    if not shared:
        raise SystemExit("两份读数没有共同的片段 id，配不上对")

    similarity = grouped_deltas(cand, ref, "similarity")
    hits = {
        voice: [
            float(bool(cand[c]["hit"])) - float(bool(ref[c]["hit"]))
            for c in sorted(set(cand) & set(ref))
            if voice_of(c) == voice
        ]
        for voice in sorted({voice_of(c) for c in shared})
    }

    report = {
        "n_clips": len(shared),
        "n_voices": len(similarity),
        # Higher similarity and more hits are better, so lower_is_better is off.
        "similarity": compare_paired_clustered(
            "clone_similarity",
            similarity,
            lower_is_better=False,
            seed=args.seed,
            effect_of_interest=SIMILARITY_EFFECT,
        ),
        "rank1_hit": compare_paired_clustered(
            "clone_rank1_hit",
            hits,
            lower_is_better=False,
            seed=args.seed,
            effect_of_interest=HIT_EFFECT,
        ),
        "per_voice_similarity_mean": {
            voice: sum(v) / len(v) for voice, v in sorted(similarity.items())
        },
        "per_voice_hit_change": {voice: sum(v) for voice, v in sorted(hits.items())},
        "arms": {"candidate": str(args.candidate), "reference": str(args.reference)},
        "note": "按音色聚类算噪声底。逐行底会把二十条同音色的片段当成二十份独立证据。",
    }

    print(f"  片段 {report['n_clips']}，音色 {report['n_voices']}")
    print(f"  {report['similarity']}")
    print(f"  {report['rank1_hit']}")
    worst = sorted(report["per_voice_similarity_mean"].items(), key=lambda kv: kv[1])[:3]
    print("  相似度掉得最多的三个音色: " + ", ".join(f"{v} {d:+.4f}" for v, d in worst))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
