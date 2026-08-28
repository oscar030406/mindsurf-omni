"""Latency of both stages on a CPU, because "local and small" has so far only
ever been measured on a 4090."""

import glob
import json
import os
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)

clips = sorted(
    glob.glob(os.path.expanduser("~/omni/corpora/data_aishell/data_aishell/wav/test/*/*.wav"))
)[:40]
print(f"{len(clips)} clips, {torch.get_num_threads()} threads", flush=True)

import soundfile as sf  # noqa: E402

seconds = [sf.info(c).duration for c in clips]
print(f"audio: total {sum(seconds):.1f}s, median {statistics.median(seconds):.2f}s", flush=True)

from funasr import AutoModel  # noqa: E402

model = AutoModel(
    model=os.path.expanduser("~/omni/minimind-o/model/SenseVoiceSmall"),
    device="cpu",
    disable_update=True,
    disable_pbar=True,
    disable_log=True,
)

asr_ms, texts = [], []
for i, clip in enumerate(clips):
    t = time.perf_counter()
    out = model.generate(input=clip, cache={}, language="zh", use_itn=True, batch_size=1)
    asr_ms.append((time.perf_counter() - t) * 1000)
    texts.append(out[0]["text"])
    if i == 0:
        print(f"  first (warm-up included): {asr_ms[0]:.0f} ms", flush=True)

warm = asr_ms[1:]
rtf = sum(warm) / 1000 / sum(seconds[1:])
print(
    json.dumps(
        {
            "stage": "asr_cpu",
            "n": len(warm),
            "p50_ms": round(statistics.median(warm), 1),
            "p95_ms": round(sorted(warm)[int(len(warm) * 0.95)], 1),
            "mean_ms": round(statistics.mean(warm), 1),
            "rtf": round(rtf, 3),
        },
        ensure_ascii=False,
    ),
    flush=True,
)

sys.path.insert(0, os.path.expanduser("~/omni/minimind-o/src"))
import asyncio  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

from mindsurf_omni.service.polish import Polisher  # noqa: E402

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

bodies = [b for b in (re.sub(r"<\|[^|]*\|>", "", t).strip() for t in texts) if b]


async def run():
    ms = []
    for i, body in enumerate(bodies):
        t = time.perf_counter()
        await pol.polish(body)
        ms.append((time.perf_counter() - t) * 1000)
        if i == 0:
            print(f"  first polish (warm-up included): {ms[0]:.0f} ms", flush=True)
    return ms


pol_ms = asyncio.run(run())
warm = pol_ms[1:]
print(
    json.dumps(
        {
            "stage": "polish_cpu",
            "n": len(warm),
            "chars_median": statistics.median(len(b) for b in bodies[1:]),
            "p50_ms": round(statistics.median(warm), 1),
            "p95_ms": round(sorted(warm)[int(len(warm) * 0.95)], 1),
            "mean_ms": round(statistics.mean(warm), 1),
        },
        ensure_ascii=False,
    ),
    flush=True,
)
