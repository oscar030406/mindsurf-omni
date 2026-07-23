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


def compare_to_fixture(model: Any, fixture_path: Path) -> dict[str, Any]:
    """Compare against the logits recorded when the base was released.

    Reports more than a maximum, because a maximum alone cannot tell the two
    failure modes apart:

    * a wrong rename *concentrates* -- one projection goes wrong and the error
      compounds through the stack, so a few positions are far worse than the
      rest, and the argmax moves;
    * float32 accumulating differently *smears* -- every position drifts by
      roughly the same amount, near the epsilon of the logit scale, and the
      argmax never moves.

    The spread ratio and the argmax agreement are what separate them.
    """
    fixture = np.load(fixture_path)
    reference = fixture["logits"]
    input_ids = torch.tensor(fixture["input_ids"], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits.float().numpy()

    difference = np.abs(logits - reference)
    scale = float(np.abs(reference).max())
    per_position = difference.max(axis=-1).ravel()
    median = float(np.percentile(per_position, 50))
    return {
        "max_absolute": float(difference.max()),
        "relative_to_scale": float(difference.max()) / scale,
        "float32_eps_at_scale": scale * float(np.finfo(np.float32).eps),
        # Near 1 means the error is spread evenly; large means it is piled on a
        # few positions, which is what a wrong mapping looks like.
        "spread_ratio": float(np.percentile(per_position, 99)) / max(median, 1e-30),
        "argmax_agrees": bool((logits.argmax(-1) == reference.argmax(-1)).all()),
    }


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
        "--eps-multiple",
        type=float,
        default=16.0,
        help=(
            "tolerated logit difference, in multiples of float32 epsilon at the "
            "logit scale; expressed this way rather than as an absolute number "
            "so the bound stays meaningful if the model's logit range changes"
        ),
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=5.0,
        help=(
            "largest tolerated p99/p50 ratio of per-position error; a wrong "
            "mapping piles error on a few positions, arithmetic noise does not"
        ),
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

    comparison = compare_to_fixture(model, args.fixture)
    allowed = comparison["float32_eps_at_scale"] * args.eps_multiple
    failures = []
    if comparison["max_absolute"] > allowed:
        failures.append(
            f"logits differ by {comparison['max_absolute']:.3e}, above the tolerated "
            f"{allowed:.3e} ({args.eps_multiple:g} x float32 eps at this logit scale)"
        )
    if comparison["spread_ratio"] > args.max_spread:
        failures.append(
            f"error is concentrated rather than spread: p99/p50 ratio "
            f"{comparison['spread_ratio']:.1f}, above {args.max_spread:g} -- this is what a "
            f"wrong parameter mapping looks like, not arithmetic noise"
        )
    if not comparison["argmax_agrees"]:
        failures.append("the model's predictions changed, so the conversion is not faithful")
    if failures:
        raise SystemExit("conversion rejected: " + "; ".join(failures))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "equivalence": comparison,
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
