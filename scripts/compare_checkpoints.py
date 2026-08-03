"""Check that the half a run froze really did not move.

``FREEZE=all`` trains the Talker and ``audio_proj`` and freezes the Thinker, and
nothing in the pipeline reports whether that held. A wrong flag, a checkpoint
loaded at the wrong shape, or an optimiser that kept a stale parameter group all
produce a model that trains and evaluates normally while the half that was
supposed to be constant has drifted. Acceptance for those runs is written as
"Thinker bit-for-bit unchanged", so this is the thing that decides it.

Equality is exact, not ``allclose``. The claim is that no gradient reached these
tensors, and a frozen weight that moved by one ulp did receive one. A tolerance
here would convert the claim being tested into a different, weaker claim.

Tensors are compared by group because the two halves have opposite expectations
in the same file: the frozen group must be identical and the trained group must
not be. A run that froze the wrong half passes a whole-file comparison in the
one direction anybody bothers to check, so both directions are reported.

    python scripts/compare_checkpoints.py out/t2a_graft_768.pth out/sft_emo_768.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Same split as scripts/graft_talker.py. audio_proj sits with the Thinker there
# because it is ours, but FREEZE=all trains it, so it belongs with the Talker
# for this question.
FROZEN_PREFIXES = ("model.", "lm_head.")
TRAINED_PREFIXES = ("talker.", "audio_proj.", "vision_proj.")


def group_of(name: str, frozen: tuple[str, ...], trained: tuple[str, ...]) -> str:
    if name.startswith(frozen):
        return "frozen"
    if name.startswith(trained):
        return "trained"
    return "unclassified"


def compare(
    before: dict[str, Any],
    after: dict[str, Any],
    frozen: tuple[str, ...] = FROZEN_PREFIXES,
    trained: tuple[str, ...] = TRAINED_PREFIXES,
) -> dict[str, Any]:
    """Per group: how many tensors are bit-for-bit equal, and the largest move.

    Keys present in only one checkpoint are reported rather than skipped. A
    missing tensor is how a shape mismatch shows up after the loader has already
    dropped it, and that is the failure this project has paid for twice.
    """
    report: dict[str, Any] = {
        "only_in_before": sorted(set(before) - set(after)),
        "only_in_after": sorted(set(after) - set(before)),
        "groups": {},
    }
    for name in sorted(set(before) & set(after)):
        group = report["groups"].setdefault(
            group_of(name, frozen, trained),
            {"total": 0, "identical": 0, "max_abs_diff": 0.0, "moved": []},
        )
        group["total"] += 1
        left, right = before[name], after[name]
        if left.shape != right.shape:
            group["moved"].append(f"{name} (shape {tuple(left.shape)} -> {tuple(right.shape)})")
            continue
        if left.equal(right):
            group["identical"] += 1
            continue
        diff = (left.float() - right.float()).abs().max().item()
        group["max_abs_diff"] = max(group["max_abs_diff"], diff)
        group["moved"].append(name)
    return report


def load(path: Path) -> dict[str, Any]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=True)
    # Trainers save either the state dict itself or a resume bundle around it.
    for key in ("state_dict", "model", "weights"):
        if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
            return blob[key]
    return blob


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--frozen-prefix", action="append", default=None)
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()

    frozen = tuple(args.frozen_prefix) if args.frozen_prefix else FROZEN_PREFIXES
    report = compare(load(args.before), load(args.after), frozen=frozen)

    for name, group in sorted(report["groups"].items()):
        verdict = ""
        if name == "frozen":
            verdict = " <- must be all" if group["identical"] == group["total"] else " <- MOVED"
        print(
            f"{name:>13}: {group['identical']}/{group['total']} identical, "
            f"max |diff| {group['max_abs_diff']:.6g}{verdict}"
        )
        for moved in group["moved"][:5]:
            print(f"               moved: {moved}")
    for side in ("only_in_before", "only_in_after"):
        if report[side]:
            print(f"{side}: {len(report[side])} -- {report[side][:5]}")

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    frozen_group = report["groups"].get("frozen", {"total": 0, "identical": 0})
    if frozen_group["total"] == 0:
        print("no tensor matched the frozen prefixes -- nothing was checked")
        return 2
    ok = frozen_group["identical"] == frozen_group["total"] and not report["only_in_after"]
    print("frozen half unchanged" if ok else "frozen half MOVED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
