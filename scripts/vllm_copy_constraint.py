"""The copy constraint as a vLLM logits processor, and what it costs.

Free generation on vLLM runs 6.8x our in-process batching and holds its latency
flat as concurrency rises. None of that is claimable until the constraint comes
with it: the output has to stay a subsequence of the transcript, and enforcing
that means a mask rebuilt per row per step from that row's own decode history.

vLLM's per-request logits processor takes exactly the arguments the mask needs
-- the tokens generated so far -- so this is a straight port of what
`_polish_batch` already does, not a new algorithm.
"""

import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/models/src")

LOOKAHEAD = 6


def reachable(source, pointer, lookahead=LOOKAHEAD):
    """The tokens a single step may land on. Same rule as the service."""
    if not lookahead:
        return source[pointer:]
    return list(source[pointer : pointer + lookahead])


def pointer_after(source, produced):
    """How far into the transcript the output has consumed, matching greedily."""
    pointer = 0
    for token in produced:
        while pointer < len(source) and source[pointer] != token:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer


class CopyConstraint:
    """One per request, closed over that request's transcript tokens."""

    __slots__ = ("source", "stop")

    def __init__(self, source, stop):
        self.source = source
        self.stop = stop

    def __call__(self, produced, logits):
        allowed = reachable(self.source, pointer_after(self.source, produced))
        keep = torch.full_like(logits, float("-inf"))
        index = torch.tensor(sorted({*allowed, self.stop}), device=logits.device, dtype=torch.long)
        keep[index] = logits[index]
        return keep


from vllm.v1.sample.logits_processor import AdapterLogitsProcessor  # noqa: E402


class CopyConstraintAdapter(AdapterLogitsProcessor):
    """Per-request constraint, wired the way v0.26 wants it.

    Declared not argmax-invariant, which is the truth -- the mask is there
    precisely to change which token wins -- and which costs some of vLLM's
    fast path. That cost is what this benchmark exists to measure.
    """

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        extra = params.extra_args or {}
        source = extra.get("source")
        if not source:
            return None
        return CopyConstraint(list(source), int(extra["stop"]))


def main() -> None:
    from transformers import AutoTokenizer  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    tok = AutoTokenizer.from_pretrained("/models/polish6")
    chat = AutoTokenizer.from_pretrained("/models/tokenizer")

    INSTRUCTION = (
        "把下面这段语音转写整理成通顺的文字。只删掉口语词和重复，不要改写内容，不要添加任何东西。"  # noqa: E501
    )
    rows = [
        json.loads(x)
        for x in open("/models/pairs_holdout.jsonl", encoding="utf-8")  # noqa: SIM115
        if x.strip()  # noqa: SIM115
    ]  # noqa: E501, SIM115
    texts = [r["source"] for r in rows if r["source"].strip()][:256]

    prompts, sources = [], []
    for text in texts:
        msg = [{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]
        prompts.append(
            str(chat.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        )  # noqa: E501
        sources.append(tok(text, add_special_tokens=False).input_ids)

    stop = tok.eos_token_id or 2
    print(
        f"{len(prompts)} 条，转写中位 {statistics.median(len(s) for s in sources):.0f} token",
        flush=True,
    )  # noqa: E501

    llm = LLM(
        model="/models/polish6",
        dtype="float32",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        enforce_eager=False,
        logits_processors=[CopyConstraintAdapter],
    )

    out = {}
    for name, constrained in (("自由生成", False), ("带复制约束", True)):
        params = []
        for source in sources:
            kw = {"max_tokens": 160, "temperature": 0.0}
            if constrained:
                kw["extra_args"] = {"source": source, "stop": stop}
            params.append(SamplingParams(**kw))
        started = time.perf_counter()
        got = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        made = sum(len(g.outputs[0].token_ids) for g in got)
        asked = sum(len(g.prompt_token_ids) for g in got)
        out[name] = {
            "requests": len(got),
            "seconds": round(elapsed, 2),
            "requests_per_second": round(len(got) / elapsed, 2),
            "output_tokens_per_second": round(made / elapsed),
            "total_tokens_per_second": round((made + asked) / elapsed),
        }
        print(
            f"  {name:<12} {out[name]['requests_per_second']:7.2f} 请求/秒  "
            f"输出 {out[name]['output_tokens_per_second']:6d} tok/s  "
            f"含 prompt {out[name]['total_tokens_per_second']:6d} tok/s",
            flush=True,
        )
        if constrained:
            bad = 0
            for g, source in zip(got, sources):  # noqa: B905
                produced = list(g.outputs[0].token_ids)
                it = iter(source)
                if not all(t in it for t in produced if t != stop):
                    bad += 1
            print(
                f"     输出不是转写子序列的：{bad}/{len(got)}   ← 0 才算约束真的接上了", flush=True
            )  # noqa: E501

    free = out["自由生成"]["total_tokens_per_second"]
    tied = out["带复制约束"]["total_tokens_per_second"]
    print(f"\n约束的代价：{free} → {tied} tok/s，剩 {tied / free:.0%}")
    print(
        f"对照我们的进程内批处理 3678 tok/s（在飞 32）：带约束的 vLLM 是它的 {tied / 3678:.1f} 倍"
    )  # noqa: E501
    json.dump(out, open("/models/vllm_constrained.json", "w"), ensure_ascii=False, indent=1)  # noqa: SIM115


if __name__ == "__main__":
    main()
