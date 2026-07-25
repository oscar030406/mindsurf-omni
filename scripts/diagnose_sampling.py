"""Can the audio sampling knobs possibly matter for this checkpoint?

A sweep of seven settings found nothing that beat the inherited
temperature 0.2 / top-k 50 / penalty 1.05, and nothing provably worse either --
every comparison landed inside its noise floor. A null like that has two very
different explanations that lead opposite ways:

* the knobs cannot matter, because dividing logits by 0.2 leaves a distribution
  so peaked that every setting decodes the same codes;
* or they can, and n=24 was too small to see it.

This separates them without another sweep, by looking at the distribution the
sampler actually faces. If the drawn code is the argmax essentially always,
decoding is greedy in practice, temperature and top-k are inert, and no n will
find an effect that cannot exist. Well below 100% and the sampler is making
real choices, so the axis is worth a properly powered sweep.

Reported per codebook rather than pooled. The coarse layers carry semantics and
the fine ones detail; one can be saturated while the other is not, and the
average would hide exactly that. Measured on upstream's checkpoint it does:
codebook 0 agrees with greedy 86.8% of the time, codebook 7 only 70.8%.

The generation loop lives in evaluate_talker so there is one copy of it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scripts.evaluate_talker import (  # noqa: E402
    DEFAULT_SAMPLING,
    build_model,
    load_texts,
    speak_forced,
)

# Above this, sampling is greedy in practice and the knobs cannot bite.
INERT_ABOVE = 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--shape", default="mindsurf")
    parser.add_argument("--minimind-root", required=True, type=Path)
    parser.add_argument("--audio-encoder", required=True, type=Path)
    parser.add_argument("--texts", type=Path, default=Path("configs/talker_texts_zh_v1.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    model = build_model(
        args.checkpoint, args.minimind_root, args.audio_encoder, args.shape, args.device
    )
    if model.audio_encoder is not None:
        model.audio_encoder.to(args.device)

    pooled: dict[int, list[tuple[float, bool]]] = {layer: [] for layer in range(8)}

    def watch(layer: int, top1: float, was_argmax: bool) -> None:
        pooled[layer].append((top1, was_argmax))

    for index, row in enumerate(load_texts(args.texts)[: args.limit]):
        speak_forced(
            model,
            tokenizer,
            row["prompt"],
            row["text"],
            args.device,
            seed=20260725 + index,
            sampling=DEFAULT_SAMPLING,
            observer=watch,
        )
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{args.limit}", flush=True)

    print(
        f"\n采样锐度（继承设置 t{DEFAULT_SAMPLING.temperature} k{DEFAULT_SAMPLING.top_k} "
        f"rp{DEFAULT_SAMPLING.penalty}/win{DEFAULT_SAMPLING.penalty_window}）"
    )
    print(f"{'码本':>4} {'top-1 概率中位':>14} {'采样==argmax':>14} {'步数':>8}")
    report: dict[str, Any] = {"per_codebook": {}}
    agreements = []
    for layer in range(8):
        values = pooled[layer]
        if not values:
            continue
        top1 = statistics.median(probability for probability, _ in values)
        agree = sum(1 for _, matched in values if matched) / len(values)
        agreements.append(agree)
        report["per_codebook"][str(layer)] = {
            "top1_probability_median": top1,
            "argmax_agreement": agree,
            "steps": len(values),
        }
        print(f"{layer:>4} {top1:>14.3f} {agree:>13.1%} {len(values):>8}")

    overall = statistics.fmean(agreements)
    report["overall_argmax_agreement"] = overall
    print(f"\n合计 采样==argmax {overall:.1%}")
    if overall >= INERT_ABOVE:
        report["verdict"] = "inert"
        print(
            f"  ≥ {INERT_ABOVE:.0%}：解码事实上已是贪心，温度与 top-k 在这个 checkpoint 上惰性。\n"
            "  扫描全落在噪声里不是样本量不够，是这里没有效应可找。"
        )
    else:
        report["verdict"] = "live"
        print(
            f"  < {INERT_ABOVE:.0%}：采样在真的做选择，这条轴还活着——扫描的 null 是功效问题。\n"
            "  逐码本的差异（粗码本自信、细码本发飘）说明单一温度是个假设，不是给定。"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"输出 {args.output}")


if __name__ == "__main__":
    main()
