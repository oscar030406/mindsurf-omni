"""Measure strict holdout loss on a checkpoint, for the text-regression check.

The number this produces has to be comparable to the 1.7268 the base was
recorded at, or it is worse than useless: a value that looks comparable and is
not silently invents a regression or hides one.

Comparability is the whole job here, and it has three parts, all pinned and
printed with the result:

* the holdout, by digest -- a fresh split is a different number;
* the sequence length, 384 -- a different length is a different number;
* the loss recipe. This is the part an earlier version got wrong. The baseline
  is a token-weighted loss over contiguously packed blocks, the way the
  pretraining eval computes it. Padding each text to max_length and averaging
  per-example means -- which is what this script used to do, and it counted the
  padding tokens as targets on top of that -- is a different number that looked
  like the same one. So the packing here matches the baseline exactly: texts
  are concatenated with an end-of-text token between them, cut into blocks of
  max_length+1, and the loss is the total NLL divided by the total tokens.

One caveat this cannot enforce, only state: comparing a base checkpoint to one
that has been SFT'd is not a like-for-like regression. An instruction- or
audio-tuned model reallocates probability toward its new format and scores
worse on raw pretraining prose whatever its language ability -- so a large rise
from base to an SFT checkpoint is expected in direction and does not by itself
mean the language ability broke. The 0.05-nat threshold is for two runs of the
same objective, not for base vs SFT. Read the number, then read that sentence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# What the base was measured with. A run that differs is not comparable to the
# baseline, however similar the number looks.
BASELINE_MAX_LENGTH = 384
BASELINE_STRICT_VAL = 1.7268
# Our base's shape; MiniMind's defaults would build a different, smaller model.
THINKER_SHAPE = {
    "hidden_size": 768,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 3584,
    "vocab_size": 6400,
    "max_position_embeddings": 32768,
    "use_moe": False,
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def packed_blocks(texts: list[str], tokenizer: Any, max_length: int) -> list[list[int]]:
    """Concatenate with an EOS between texts and cut into max_length+1 blocks.

    The same packing the baseline used: contiguous, so a block spans document
    boundaries, and the trailing partial buffer is dropped. Per-document padding
    would change the number and is what made the old version incomparable.
    """
    block_size = max_length + 1
    eos = tokenizer.eos_token_id
    if eos is None:
        raise SystemExit("the tokenizer defines no eos_token_id; the packing needs one")
    buffer: list[int] = []
    blocks: list[list[int]] = []
    for text in texts:
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        buffer.append(eos)
        while len(buffer) >= block_size:
            blocks.append(buffer[:block_size])
            del buffer[:block_size]
    return blocks


def strict_loss(
    checkpoint: Path,
    minimind_root: Path,
    tokenizer: Any,
    blocks: list[list[int]],
    device: str,
    batch_size: int,
) -> float:
    """Token-weighted NLL of the Thinker over the packed blocks."""
    import sys as _sys

    import torch

    if str(minimind_root) not in _sys.path:
        _sys.path.insert(0, str(minimind_root))
    from model.model_minimind import (  # type: ignore[import-not-found]
        MiniMindConfig,
        MiniMindForCausalLM,
    )

    model = MiniMindForCausalLM(MiniMindConfig(**THINKER_SHAPE))
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    weights = {key: value for key, value in state.items() if key.startswith(("model.", "lm_head"))}
    if sum(1 for _ in weights) < 50:
        raise SystemExit(f"only {len(weights)} language weights in {checkpoint}; not the Thinker")
    model.load_state_dict(weights, strict=False)
    model.to(device).eval()

    nll_sum = 0.0
    tokens = 0
    with torch.inference_mode():
        for start in range(0, len(blocks), batch_size):
            batch = torch.tensor(
                blocks[start : start + batch_size], dtype=torch.long, device=device
            )
            logits = model(input_ids=batch[:, :-1]).logits
            labels = batch[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
            )
            nll_sum += float(loss)
            tokens += labels.numel()
    if not tokens:
        raise SystemExit("no evaluation tokens; is the holdout empty?")
    return nll_sum / tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        help="run this one too and report the delta; the honest comparison for "
        "'did SFT erode the base' is base against candidate on the same holdout",
    )
    parser.add_argument(
        "--holdout", required=True, type=Path, help="JSONL, one {'text': ...} per line"
    )
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument(
        "--minimind-root",
        type=Path,
        default=Path.home() / "omni" / "minimind-o",
        help="MiniMind-O checkout, for its model class",
    )
    parser.add_argument("--max-length", type=int, default=BASELINE_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_length != BASELINE_MAX_LENGTH:
        print(
            f"注意: max_length {args.max_length} 与基线的 {BASELINE_MAX_LENGTH} 不同，"
            "这个数字与 1.7268 不可比",
            file=sys.stderr,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    texts = [
        json.loads(line)["text"]
        for line in args.holdout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blocks = packed_blocks(texts, tokenizer, args.max_length)
    if not blocks:
        raise SystemExit(f"{args.holdout} produced no full blocks at length {args.max_length}")

    value = strict_loss(
        args.checkpoint, args.minimind_root, tokenizer, blocks, args.device, args.batch_size
    )
    print(f"strict_val {value:.4f} over {len(blocks)} blocks")
    print(f"  holdout    {args.holdout.name} sha256 {digest(args.holdout)[:16]}")
    print(f"  max_length {args.max_length}  packing token-weighted, contiguous")

    payload: dict[str, Any] = {
        "strict_val": value,
        "blocks": len(blocks),
        "max_length": args.max_length,
        "holdout_sha256": digest(args.holdout),
    }

    if args.baseline_checkpoint is not None:
        base = strict_loss(
            args.baseline_checkpoint,
            args.minimind_root,
            tokenizer,
            blocks,
            args.device,
            args.batch_size,
        )
        payload["baseline_checkpoint_strict_val"] = base
        payload["delta_over_baseline_checkpoint"] = value - base
        print(f"  baseline ckpt {base:.4f}  candidate - baseline {value - base:+.4f}")
        print(
            "  note: base vs SFT on pretraining prose conflates expected instruction-tuning "
            "distribution shift with genuine forgetting; a large rise is not by itself damage"
        )
    else:
        print(f"  recorded 1.7268  difference {value - BASELINE_STRICT_VAL:+.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  output     {args.output}")


if __name__ == "__main__":
    main()
