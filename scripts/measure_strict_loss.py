"""Measure strict holdout loss on a checkpoint, for the text-regression check.

The holdout is the one the base was released against, so the number is
comparable to the 1.7268 recorded then. Anything else -- a fresh split, a
different max length, a different batching -- produces a number that looks
comparable and is not.

So the three things that would silently shift it are pinned here and printed
with the result: the holdout's digest, the sequence length, and the number of
batches. A reader can tell at a glance whether two runs are comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# What the base was measured with. A run that differs is not comparable to the
# baseline, however similar the number looks.
BASELINE_MAX_LENGTH = 384
BASELINE_STRICT_VAL = 1.7268


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path, help="JSONL, one text per line")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=BASELINE_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=250)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.max_length != BASELINE_MAX_LENGTH:
        print(
            f"注意: max_length {args.max_length} 与基线的 {BASELINE_MAX_LENGTH} 不同，"
            "这个数字与 1.7268 不可比",
            file=sys.stderr,
        )

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # The checkpoint may be a bare state dict or a training checkpoint; both
    # appear in this project and confusing them yields a model of random
    # weights that still produces a plausible-looking loss.
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    if not isinstance(state, dict) or not state:
        raise SystemExit(f"{args.checkpoint} holds no weights this script recognises")

    sys.path.insert(0, str(Path.home() / "omni" / "minimind-o"))
    from model.model_minimind import (  # type: ignore[import-not-found]
        MiniMindConfig,
        MiniMindForCausalLM,
    )

    config = MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=3584,
        vocab_size=6400,
        max_position_embeddings=32768,
        use_moe=False,
    )
    model = MiniMindForCausalLM(config)
    missing, unexpected = model.load_state_dict(
        {key: value for key, value in state.items() if key.startswith(("model.", "lm_head"))},
        strict=False,
    )
    loaded = sum(1 for key in state if key.startswith(("model.", "lm_head")))
    if loaded < 50:
        raise SystemExit(
            f"only {loaded} language weights matched; this checkpoint is not the Thinker"
        )
    model.to(args.device).eval()

    texts = [
        json.loads(line)["text"]
        for line in args.holdout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    losses = []
    with torch.inference_mode():
        for start in range(0, min(len(texts), args.batches * args.batch_size), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )
            ids = encoded["input_ids"].to(args.device)
            output = model(input_ids=ids, labels=ids)
            losses.append(float(output.loss))

    mean = sum(losses) / len(losses)
    print(f"strict_val {mean:.4f} over {len(losses)} batches")
    print(f"  holdout   {args.holdout.name} sha256 {digest(args.holdout)[:16]}")
    print(f"  max_length {args.max_length}  batch_size {args.batch_size}")
    print(f"  baseline  {BASELINE_STRICT_VAL:.4f}  difference {mean - BASELINE_STRICT_VAL:+.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(losses) + "\n", encoding="utf-8")
        print(f"  losses    {args.output}")


if __name__ == "__main__":
    main()
