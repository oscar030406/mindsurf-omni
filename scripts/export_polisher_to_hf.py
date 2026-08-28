"""Export the polisher to a HuggingFace layout vLLM loads natively.

MiniMind's tensor names are Llama's, plus per-head q_norm/k_norm on the
attention -- which is exactly Qwen3, not Llama. So this needs no custom
architecture registered with vLLM: written out as Qwen3ForCausalLM it is a
model vLLM already knows how to serve.

Everything the config declares is read off the checkpoint rather than typed in,
because a config that disagrees with the weights loads without complaint and
answers nonsense.
"""

import json
import re
import shutil
import sys
from pathlib import Path

import torch

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/sft_polish6_768.pth")  # noqa: E501
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("<导出目录>")
TOK = Path("assets/tokenizer")

sd = torch.load(str(SRC), map_location="cpu", weights_only=True)
keep = {
    k: v
    for k, v in sd.items()
    if k.startswith(("model.", "lm_head.")) and not k.startswith("model.audio")
}
layers = 1 + max(int(m.group(1)) for k in keep if (m := re.match(r"model\.layers\.(\d+)\.", k)))

vocab, hidden = keep["model.embed_tokens.weight"].shape
intermediate = keep["model.layers.0.mlp.gate_proj.weight"].shape[0]
head_dim = keep["model.layers.0.self_attn.q_norm.weight"].shape[0]
heads = keep["model.layers.0.self_attn.q_proj.weight"].shape[0] // head_dim
kv_heads = keep["model.layers.0.self_attn.k_proj.weight"].shape[0] // head_dim
tied = "lm_head.weight" not in keep

print(f"层 {layers}  隐藏 {hidden}  词表 {vocab}  中间层 {intermediate}")
print(f"头 {heads}  KV头 {kv_heads}  head_dim {head_dim}  共享词嵌入 {tied}")
print(f"参数 {sum(v.numel() for v in keep.values()) / 1e6:.2f}M")

OUT.mkdir(parents=True, exist_ok=True)
config = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": hidden,
    "intermediate_size": intermediate,
    "num_hidden_layers": layers,
    "num_attention_heads": heads,
    "num_key_value_heads": kv_heads,
    "head_dim": head_dim,
    "vocab_size": vocab,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": tied,
    "torch_dtype": "float32",
    "bos_token_id": 1,
    "eos_token_id": 2,
}
(OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

from safetensors.torch import save_file  # noqa: E402

if tied:
    keep.pop("lm_head.weight", None)
save_file(
    {k: v.contiguous() for k, v in keep.items()},
    str(OUT / "model.safetensors"),
    metadata={"format": "pt"},
)

for name in (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
):  # noqa: E501
    src = TOK / name
    if src.is_file():
        shutil.copy(src, OUT / name)

print(f"\n写出 {OUT}")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")

# The config has to agree with the weights, so load it back through
# transformers before anyone tries to serve it.
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

model = AutoModelForCausalLM.from_pretrained(str(OUT), dtype=torch.float32)
tok = AutoTokenizer.from_pretrained(str(OUT))
print(f"\n装回来了：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 参数")
ids = tok("今天天气", return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=12, do_sample=False)
print("续写：", tok.decode(out[0], skip_special_tokens=True))
