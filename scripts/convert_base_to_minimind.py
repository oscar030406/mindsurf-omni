"""Convert the released text base into the checkpoint MiniMind-O trains from.

MiniMind-O's Thinker is a MiniMind transformer, and our base is the same
architecture under different parameter names -- q_proj, then a per-head
RMSNorm, then RoPE, SwiGLU feed-forward, tied embeddings. So this is a rename
plus a config, not a reimplementation.

Two config values differ from MiniMind's defaults and must be passed
explicitly, because the defaults would silently build a smaller model that
refuses our weights:

    num_key_value_heads  8     (MiniMind defaults to 4; our base is MHA)
    intermediate_size    3584  (MiniMind derives 1536 from hidden_size)

A rename fails quietly -- swap two feed-forward matrices and the model still
runs, just worse -- so nothing is written until the converted model reproduces
the reference logits recorded when the base was released.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO / "artifacts" / "reference-logits-seed20260721.npz"


def load_minimind_model_module(minimind_root: Path) -> Any:
    """Import MiniMind's own model definition, so the check is against theirs."""
    module_path = minimind_root / "model" / "model_minimind.py"
    if not module_path.is_file():
        raise SystemExit(f"MiniMind model definition not found: {module_path}")
    sys.path.insert(0, str(module_path.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("model_minimind", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rename_from_hf(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Our released HF export already uses MiniMind's parameter names.

    Both are the standard decoder layout: model.embed_tokens, model.layers.N
    .self_attn.{q,k,v,o}_proj and .{q,k}_norm, .mlp.{gate,up,down}_proj,
    .{input,post_attention}_layernorm, model.norm, lm_head. Passing the state
    dict through unchanged is therefore correct, and the equivalence check
    below is what proves it rather than this comment.
    """
    return dict(state)


def build_config(module: Any, config_json: dict[str, Any]) -> Any:
    return module.MiniMindConfig(
        hidden_size=config_json["hidden_size"],
        num_hidden_layers=config_json["num_hidden_layers"],
        num_attention_heads=config_json["num_attention_heads"],
        num_key_value_heads=config_json["num_key_value_heads"],
        intermediate_size=config_json["intermediate_size"],
        vocab_size=config_json["vocab_size"],
        rms_norm_eps=config_json["rms_norm_eps"],
        rope_theta=config_json.get("rope_theta", 1e6),
        tie_word_embeddings=config_json["tie_word_embeddings"],
        max_position_embeddings=config_json["max_position_embeddings"],
        use_moe=False,
    )


def max_logit_difference(model: Any, fixture_path: Path) -> float:
    fixture = np.load(fixture_path)
    input_ids = torch.tensor(fixture["input_ids"], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    return float(np.abs(logits.float().numpy() - fixture["logits"]).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="released HF model directory (config.json + model.safetensors)",
    )
    parser.add_argument(
        "--minimind-root",
        required=True,
        type=Path,
        help="checkout of minimind or minimind-o, for its model definition",
    )
    parser.add_argument("--output", required=True, type=Path, help="checkpoint to write")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="largest tolerated logit difference; the default demands equality",
    )
    args = parser.parse_args()

    from safetensors.torch import load_file

    module = load_minimind_model_module(args.minimind_root)
    config_json = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    config = build_config(module, config_json)

    model = module.MiniMindForCausalLM(config)
    state = rename_from_hf(load_file(str(args.source / "model.safetensors")))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # lm_head is tied to the embedding, so loading the embedding sets both and
    # lm_head reports as missing. Anything else missing is a real hole.
    missing = [key for key in missing if key != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"parameter mismatch: missing={missing} unexpected={unexpected}")
    model.eval().float()

    difference = max_logit_difference(model, args.fixture)
    if difference > args.tolerance:
        raise SystemExit(
            f"conversion rejected: logits differ by {difference:.6e}, "
            f"above the tolerated {args.tolerance:.6e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "max_logit_difference": difference,
                "parameters": sum(p.numel() for p in model.parameters()),
                "config": {
                    "hidden_size": config.hidden_size,
                    "num_hidden_layers": config.num_hidden_layers,
                    "num_attention_heads": config.num_attention_heads,
                    "num_key_value_heads": config.num_key_value_heads,
                    "intermediate_size": config.intermediate_size,
                },
                # Talker reads the Thinker's middle layer, not the last one.
                "bridge_layer": config.num_hidden_layers // 2 - 1,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
