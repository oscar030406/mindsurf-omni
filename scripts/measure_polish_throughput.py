"""Pure polish throughput, which we have never measured on its own.

The 10.14 requests/second on record is a whole dictation turn -- transcribe and
polish together -- so putting it beside a number for free LLM generation was
comparing two different things. This measures the polish stage alone, at the
same concurrencies, so there is something to compare against when the vLLM
question gets answered.

Also reports tokens/second, because that is the unit the other side is using.
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "<仓库>/src")
from mindsurf_omni.service.polish import Polisher, build_prompt  # noqa: E402

ROOT = Path("<minimind 检出>")
POOL = Path("<仓库>/artifacts/polish_train")
rows = [
    json.loads(x)
    for x in (POOL / "pairs_holdout.jsonl").read_text(encoding="utf-8").splitlines()
    if x.strip()
]  # noqa: E501
texts = [r["source"] for r in rows if r["source"].strip()][:256]
CONCURRENCY = [1, 4, 8, 16, 32]

pol = Polisher(
    checkpoint=ROOT / "out/sft_polish6_768.pth",
    tokenizer_dir=Path("assets/tokenizer"),
    minimind_root=ROOT,
    device="cuda",
    tagger=ROOT / "out/polish_tagger_unionchar.pt",
    tagger_backbone=ROOT / "out/polish_tagger_unionchar_backbone.pth",
    tagger_threshold=0.4,
)
pol.load()
tok = pol._tokeniser

prompt_tokens = statistics.median(
    len(tok(build_prompt(t), add_special_tokens=False).input_ids) for t in texts
)  # noqa: E501
print(f"{len(texts)} 条，prompt 中位 {prompt_tokens:.0f} token", flush=True)


async def main():
    await pol.polish(texts[0])  # warm
    out = []
    for n in CONCURRENCY:
        batch = (texts * 4)[: max(n * 8, 64)]
        started = time.perf_counter()
        done, waits = [], []

        async def one(text, waits=waits, done=done):
            t = time.perf_counter()
            answer = await pol.polish(text)
            waits.append((time.perf_counter() - t) * 1000)
            done.append(answer)

        pending = set()
        for text in batch:
            pending.add(asyncio.create_task(one(text)))
            if len(pending) >= n:
                finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if pending:
            await asyncio.wait(pending)

        elapsed = time.perf_counter() - started
        output_tokens = sum(len(tok(a, add_special_tokens=False).input_ids) for a in done)
        waits.sort()
        row = {
            "in_flight": n,
            "requests": len(done),
            "seconds": round(elapsed, 2),
            "requests_per_second": round(len(done) / elapsed, 2),
            "p50_ms": round(waits[len(waits) // 2]),
            "p95_ms": round(waits[int(len(waits) * 0.95) - 1]),
            "output_tokens": output_tokens,
            "output_tokens_per_second": round(output_tokens / elapsed),
            "total_tokens_per_second": round((output_tokens + prompt_tokens * len(done)) / elapsed),
        }
        out.append(row)
        print(
            f"  在飞 {n:2d}: {row['requests_per_second']:6.2f} 请求/秒  "
            f"P50 {row['p50_ms']:5d} ms  P95 {row['p95_ms']:5d} ms  "
            f"输出 {row['output_tokens_per_second']:6d} tok/s  "
            f"含 prompt {row['total_tokens_per_second']:6d} tok/s",
            flush=True,
        )
    return out


rows_out = asyncio.run(main())
best = max(rows_out, key=lambda r: r["total_tokens_per_second"])
print(
    f"\n峰值 {best['total_tokens_per_second']} token/秒（含 prompt）= "
    f"{best['total_tokens_per_second'] * 60 / 1000:.0f}K TPM，在飞 {best['in_flight']}"
)
print(f"纯输出口径 {best['output_tokens_per_second'] * 60 / 1000:.0f}K TPM")
json.dump(rows_out, open("<工作目录>/polishthrough.json", "w"), ensure_ascii=False, indent=1)  # noqa: SIM115
