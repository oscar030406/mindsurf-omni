"""Build the emotional voice pack: each voice's own vector, moved along one direction.

Signal processing was ruled out twice -- a reworked reference clip loses
intelligibility, and the reference code strip turned out to carry no prosody at
all, so a perfect vocoder would still have nothing to ride on. What does reach
the model is the speaker vector, and prosody rides it.

So the deliverable is arithmetic, not audio::

    spk_emb = the voice's own + alpha * (excited - calm),   ref_codes = its own

where the two references are one speaker read in two prosodies. That direction
transplants: extracted from a synthesised pair, it still raises F0 on a
built-in voice that had nothing to do with it.

This lived as a one-off on the training host while it was a question. It is in
the tree now because it builds a deliverable, and a deliverable whose builder is
a file on one machine is a deliverable nobody else can rebuild.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def shift(
    voices: dict[str, Any], delta: Any, alpha: float, only: list[str] | None = None
) -> dict[str, Any]:
    """Same voices, same reference codes, speaker vector moved by ``alpha`` * delta.

    The codes are copied through untouched on purpose. They are the other input
    the model conditions on, and the cross-arm measurement said they carry no
    prosody -- swapping them would change identity without buying emotion.
    """
    wanted = set(only) if only else set(voices)
    missing = wanted - set(voices)
    if missing:
        raise SystemExit(f"音色包里没有 {sorted(missing)}")
    return {
        name: {"ref_codes": entry["ref_codes"], "spk_emb": entry["spk_emb"].float() + alpha * delta}
        for name, entry in voices.items()
        if name in wanted
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", type=Path, required=True, help="pack of the original vectors")
    parser.add_argument(
        "--delta-pack",
        type=Path,
        required=True,
        help="pack holding the two prosodies of one speaker",
    )
    parser.add_argument("--from-voice", default="calm", help="the delta's origin")
    parser.add_argument("--to-voice", default="excited", help="the delta's destination")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--only", nargs="*", help="limit to these voices")
    parser.add_argument("--output", type=Path, required=True, help="directory; writes voices.pt")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch

    voices = torch.load(str(args.voices), map_location="cpu")
    pair = torch.load(str(args.delta_pack), map_location="cpu")
    for name in (args.from_voice, args.to_voice):
        if name not in pair:
            raise SystemExit(f"{args.delta_pack} 里没有 {name!r}，有的是 {sorted(pair)}")

    delta = pair[args.to_voice]["spk_emb"].float() - pair[args.from_voice]["spk_emb"].float()
    shifted = shift(voices, delta, args.alpha, args.only)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(shifted, str(args.output / "voices.pt"))
    print(f"α={args.alpha} 写了 {len(shifted)} 个音色到 {args.output / 'voices.pt'}")

    # The cosine to the voice's own original vector is the price, before any
    # audio exists. It is not the criterion -- the criterion scores generated
    # audio -- but a pack whose vectors have already left the neighbourhood is
    # not worth spending a card on.
    from torch.nn.functional import cosine_similarity

    moved = {
        name: float(cosine_similarity(entry["spk_emb"], voices[name]["spk_emb"].float(), dim=0))
        for name, entry in shifted.items()
    }
    for name, value in sorted(moved.items(), key=lambda item: item[1]):
        print(f"  {name:<10} 对自己原向量 {value:.4f}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "alpha": args.alpha,
                    "delta": f"{args.to_voice} - {args.from_voice}",
                    "delta_norm": float(delta.norm()),
                    "vector_cosine_to_original": moved,
                    "note": "vector-space only; the criterion scores generated audio",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
