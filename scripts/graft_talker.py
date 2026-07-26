"""Put upstream's Talker on our Thinker, and say exactly what came from where.

Two rounds of training have now failed to teach our Talker to speak, while
upstream's released Talker does it at CER 0.0763 -- with fewer parameters than
ours on both halves. The interface between the two halves is narrow: the Talker
reads the Thinker's hidden states through ``talker.embed_proj`` and nothing
else. So the halves can be swapped, and the question "is our problem in the
Thinker's hidden states or in the Talker" becomes a measurement instead of an
argument.

What does not line up is the Talker's own four transformer layers: ours were
built with the base's shape (FFN 3584, 8 KV heads) and upstream's with the
library defaults (FFN 2432, 4 KV heads). That is why the shape has to be chosen
per half rather than globally -- the Thinker takes our overrides, the Talker
takes upstream's, and every tensor at the boundary already matches.

The merged checkpoint records its provenance in a sidecar JSON, because a
checkpoint that cannot say which half came from whom is not evidence about
either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Everything the Talker owns, including the bridge it reads the Thinker with.
TALKER_PREFIXES = ("talker.",)
# Ours: the language half, and the audio encoder's projection into it.
THINKER_PREFIXES = ("model.", "lm_head.", "audio_proj.", "vision_proj.")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def merge(ours: dict[str, Any], upstream: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Our half by prefix, theirs by prefix, and nothing silently dropped."""
    merged: dict[str, Any] = {}
    counts = {"thinker_from_ours": 0, "talker_from_upstream": 0, "unclaimed": 0}

    for name, tensor in ours.items():
        if name.startswith(THINKER_PREFIXES):
            merged[name] = tensor
            counts["thinker_from_ours"] += 1
    for name, tensor in upstream.items():
        if name.startswith(TALKER_PREFIXES):
            merged[name] = tensor
            counts["talker_from_upstream"] += 1

    # A tensor in neither list is a part of the model this script does not know
    # how to attribute; loading would then silently take a default. Counted so
    # the caller sees it rather than discovers it in a bad sample.
    for name in set(ours) | set(upstream):
        if not name.startswith(THINKER_PREFIXES + TALKER_PREFIXES):
            counts["unclaimed"] += 1
    return merged, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path, help="checkpoint supplying the Thinker")
    parser.add_argument(
        "--upstream", required=True, type=Path, help="checkpoint supplying the Talker"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    ours = torch.load(str(args.ours), map_location="cpu", weights_only=True)
    upstream = torch.load(str(args.upstream), map_location="cpu", weights_only=True)
    merged, counts = merge(ours, upstream)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(args.output))

    provenance = {
        "thinker_from": {"file": args.ours.name, "sha256": digest(args.ours)},
        "talker_from": {"file": args.upstream.name, "sha256": digest(args.upstream)},
        "tensors": counts,
        "parameters": {
            "thinker_side": sum(
                t.numel() for k, t in merged.items() if k.startswith(THINKER_PREFIXES)
            ),
            "talker_side": sum(
                t.numel() for k, t in merged.items() if k.startswith(TALKER_PREFIXES)
            ),
        },
        "note": (
            "the Thinker keeps the base's shape and the Talker keeps upstream's; "
            "load with a config that sets them separately, or the four Talker "
            "layers will not fit"
        ),
    }
    sidecar = args.output.with_suffix(".provenance.json")
    sidecar.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"thinker 取自 {args.ours.name}: {counts['thinker_from_ours']} 个张量")
    print(f"talker  取自 {args.upstream.name}: {counts['talker_from_upstream']} 个张量")
    if counts["unclaimed"]:
        print(f"未归属 {counts['unclaimed']} 个张量——它们会以初始化值留在模型里，先查清楚再用")
    print(f"参数量 thinker 侧 {provenance['parameters']['thinker_side']:,}")
    print(f"      talker  侧 {provenance['parameters']['talker_side']:,}")
    print(f"写入 {args.output}\n出处 {sidecar}")


if __name__ == "__main__":
    main()
