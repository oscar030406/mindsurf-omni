"""Polish latency on a CPU. AISHELL is clean read speech, so the cheap
pre-check skipped the model on nearly every clip and the p50 read 0.0 ms.
These are the probe sentences, which all carry something to remove."""

import asyncio
import json
import os
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(8)
sys.path.insert(0, os.path.expanduser("~/omni/mindsurf-omni/src"))
from pathlib import Path  # noqa: E402

from mindsurf_omni.service.polish import Polisher  # noqa: E402

rows = [
    json.loads(line)
    for line in open(os.path.expanduser("~/omni/probes_zh.jsonl"), encoding="utf-8")  # noqa: SIM115
    if line.strip()
]
texts = [r.get("source") or r.get("transcript") or r.get("input") for r in rows]
texts = [t for t in texts if t]
print(f"{len(texts)} probes, median {statistics.median(len(t) for t in texts)} chars", flush=True)

root = Path(os.path.expanduser("~/omni/minimind-o"))
pol = Polisher(
    checkpoint=root / "out/sft_polish6_768.pth",
    tokenizer_dir=Path("assets/tokenizer"),
    minimind_root=root,
    device="cpu",
    tagger=root / "out/polish_tagger_unionchar.pt",
    tagger_backbone=root / "out/polish_tagger_unionchar_backbone.pth",
    tagger_threshold=0.4,
)
pol.load()


async def run():
    ms, moved = [], 0
    for i, text in enumerate(texts):
        t = time.perf_counter()
        out = await pol.polish(text)
        ms.append((time.perf_counter() - t) * 1000)
        moved += out != text
        if i == 0:
            print(f"  first (warm-up included): {ms[0]:.0f} ms", flush=True)
    return ms, moved


ms, moved = asyncio.run(run())
warm = ms[1:]
print(
    json.dumps(
        {
            "stage": "polish_cpu_probes",
            "n": len(warm),
            "changed": moved,
            "p50_ms": round(statistics.median(warm), 1),
            "p95_ms": round(sorted(warm)[int(len(warm) * 0.95)], 1),
            "mean_ms": round(statistics.mean(warm), 1),
            "max_ms": round(max(warm), 1),
        },
        ensure_ascii=False,
    ),
    flush=True,
)
