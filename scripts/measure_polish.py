"""Did the polisher clean the transcript, and did it keep the words.

Two numbers that move in opposite directions, which is why they are scored
apart. A model that deletes aggressively cleans well and loses content; one
that copies its input keeps everything and cleans nothing. Either alone reads
as success.

The lines below were written before the first training run, from the floor
measurement: the polish job on the production loop is 0.0715 CER against the
original, the recogniser's own error rate under it is 0.0063, and 89% of the
gap is inserted filler.

    python scripts/measure_polish.py --checkpoint out/sft_polish_768.pth \
        --pairs artifacts/polish_train/pairs.jsonl --split val \
        --minimind-root ~/omni/minimind-o --report artifacts/polish-eval-2026-08-15.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.evaluation.metrics import (  # noqa: E402
    assess,
    character_error_rate,
    normalise_for_cer,
)
from mindsurf_omni.service.polish import (  # noqa: E402  # noqa: E402
    BRIDGING_FILLERS,
    LEADING_FILLERS,
    project_onto,
    reachable,
)
from scripts.train_polish import build_prompt  # noqa: E402

# 判据，训练之前写死。
#
# 主判据：润色后的文本对原文的 CER。输入端是 0.0715（口语词臂对干净原文），
# 识别器自己的底是 0.0063，所以 0.02 是「把差距砍掉七成、且不假装比识别底更好」。
POLISHED_CER = 0.02
# 口语词清除率：进到输入里的口语词，输出里还剩多少。
FILLER_REMOVED = 0.90
# 内容词保留率：原文的字有多少还在输出里。**这条是防编造的那一条**，
# 它和上面一条必须同时过——只过一条的模型要么没干活、要么把内容删了。
CONTENT_KEPT = 0.98
# 编造：输出里原文没有的字，占原文长度的比例。
INVENTED = 0.02

# 四条线一个字没动。动的是尺子的一处归一化：数字写法两边都折。
#
# 识别器开着 ITN，念出来的数字写成阿拉伯数字（6点），语料原文写中文数字（六点），
# 于是每个数字算一次替换。量过：这一项占天花板的 25–27%——把它折掉，
# **只能删的润色器所能产出的最好结果**从 CER 0.0218 回到 0.0159，重新落进 0.02 线以内。
# 不折的话，判据量的是「识别器怎么写数字」而不是「模型润色得怎么样」。
#
# 逐臂算过：折了之后**只有天花板换判定，没有任何一条实测臂因此过线**
# （并集 t=0.9 从 0.0567 到 0.0513、polish6 从 0.0496 到 0.0438，都还是不过）。
# 所以这不是把球门挪到球前面，是把球门挪回场内。
#
# 产品侧的论据独立成立：听写里用户念「六点」、识别器写「6点」、文本框显示「6点」，
# 那是对的甚至更好，为它扣润色器的分，量的不是这个产品要的东西。
# 同一个道理 2026-07-26 那一轮已经论过一次（experiments/2026-07-26-numeral-fold.md），
# 只是当时为合成评测加的，默认关着，没人想到润色这条路上也适用。
FOLD_NUMERALS = True


def content_kept(target: str, output: str) -> float:
    """Share of the original's characters that survived into the output."""
    left, right = (
        normalise_for_cer(target, fold_numbers=FOLD_NUMERALS),
        normalise_for_cer(output, fold_numbers=FOLD_NUMERALS),
    )
    if not left:
        return 1.0
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(left)


def invented(target: str, output: str) -> float:
    """Characters in the output that the original did not have, over its length."""
    left, right = (
        normalise_for_cer(target, fold_numbers=FOLD_NUMERALS),
        normalise_for_cer(output, fold_numbers=FOLD_NUMERALS),
    )
    if not left:
        return 0.0
    matched = sum(
        block.size
        for block in difflib.SequenceMatcher(
            None, left, right, autojunk=False
        ).get_matching_blocks()
    )
    return max(0, len(right) - matched) / len(left)


def filler_removed(row: dict[str, Any], output: str) -> tuple[int, int]:
    """Injected filler that reached the transcript, and how much of it went away.

    Counted against the source rather than the injection record: filler the
    recogniser already dropped is not work the polisher did.
    """
    heard, clean, written = (
        normalise_for_cer(row["source"], fold_numbers=FOLD_NUMERALS),
        normalise_for_cer(row["target"], fold_numbers=FOLD_NUMERALS),
        normalise_for_cer(output, fold_numbers=FOLD_NUMERALS),
    )
    arrived = removed = 0
    for item in row.get("injections", []):
        token = normalise_for_cer(item["token"])
        if not token:
            continue
        into_source = max(0, heard.count(token) - clean.count(token))
        into_output = max(0, written.count(token) - clean.count(token))
        arrived += into_source
        removed += max(0, into_source - into_output)
    return arrived, removed


def subsequence_pointer(source: list[int], produced: list[int]) -> int:
    """How far into the source the output has consumed, matching greedily.

    Deletion-only output is a subsequence of the input, so the position is
    recoverable from what has been written rather than carried as state -- which
    keeps this correct when the sampler backtracks or a batch reorders.
    """
    pointer = 0
    for token in produced:
        while pointer < len(source) and source[pointer] != token:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer


def copy_only_generate(
    model: Any,
    prompt_ids: Any,
    source: list[int],
    stop: int,
    max_new_tokens: int,
    torch: Any,
    lookahead: int = 0,
    fillers: tuple[tuple[int, ...], ...] = (),
    protect_head: bool = False,
) -> list[int]:
    """Greedy decode restricted to a subsequence of the transcript.

    Written as its own loop rather than a ``LogitsProcessor``: MiniMind's model
    class overrides ``generate`` with its own signature, so a processor passed
    to it lands in ``**kwargs`` and is silently ignored -- the first run of this
    round produced numbers identical to the unconstrained arm to four decimals,
    which is what gave it away.

    No KV cache. The sequences here are short and a wrong cache is a subtler
    bug than a slow loop.
    """
    produced: list[int] = []
    for _ in range(max_new_tokens):
        ids = (
            prompt_ids
            if not produced
            else torch.cat([prompt_ids, torch.tensor([produced], device=prompt_ids.device)], dim=1)
        )
        logits = model(input_ids=ids).logits[0, -1].float()
        pointer = subsequence_pointer(source, produced)
        # A bounded skip, because an unbounded one is how deletion-only turns
        # into deleting a clause: with every later token allowed, the model can
        # jump the length of a sentence and the output stays a legal
        # subsequence. lookahead=0 keeps the unbounded behaviour.
        ahead = set(reachable(source, pointer, lookahead, fillers, protect_head))
        ahead.add(stop)
        keep = torch.tensor(sorted(ahead), device=logits.device, dtype=torch.long)
        masked = torch.full_like(logits, float("-inf"))
        masked[keep] = logits[keep]
        chosen = int(masked.argmax())
        if chosen == stop:
            break
        produced.append(chosen)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--variant", default="mindsurf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="the product's default. Worth an ablation on this task and not "
        "elsewhere: polishing is mostly copying, and this knob exists to "
        "discourage repeating what is already in the context",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="restrict the decode to a subsequence of the transcript. Deletion "
        "becomes the only edit the model can make, so invention is impossible "
        "by construction -- and substitution becomes impossible too",
    )
    parser.add_argument(
        "--copy-lookahead",
        type=int,
        default=0,
        help="with --copy-only: how far ahead a single step may skip. 0 is "
        "unbounded, which lets the model drop a whole clause and still produce "
        "a legal subsequence",
    )
    parser.add_argument(
        "--protect-head",
        action="store_true",
        help="with --copy-only: the first token may only be skipped as part of a "
        "known filler. Measured end to end, 你想看… came back as 想看…",
    )
    parser.add_argument(
        "--tagger",
        type=Path,
        help="a trained keep/delete head. One forward pass per turn instead of "
        "a decode loop, and the operating point is a threshold rather than a "
        "decoder configuration",
    )
    parser.add_argument(
        "--tagger-threshold",
        type=float,
        default=0.5,
        help="delete a token when the head is at least this sure. Sweeping it "
        "is the point: it draws the curve the decoder configurations could only "
        "sample two points of",
    )
    parser.add_argument(
        "--project",
        action="store_true",
        help="generate freely, then keep only what the transcript actually said. "
        "The other way to make invention impossible, and it does not cost the "
        "free decode's filler removal",
    )
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="generate and time, do not score. For the deployment card, whose "
        "evaluation venv is pinned and does not carry the scoring packages -- "
        "quality is judged where the weights were trained, timing where they "
        "will run",
    )
    parser.add_argument("--output", type=Path, help="per-row rows, for reading the failures")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch

    from mindsurf_omni.service.thinker import ThinkerGenerator

    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == args.split][: args.limit]
    if not rows:
        raise SystemExit(f"{args.pairs} 里没有 split={args.split} 的行")

    generator = ThinkerGenerator(
        checkpoint=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        minimind_root=args.minimind_root,
        device=args.device,
        variant=args.variant,
    )
    generator.load()
    model, tokeniser = generator._model, generator._tokenizer  # noqa: SLF001
    model.eval()

    # Tokenised once: the decode consults these at every step, and the tokeniser
    # is the slow part of doing it inline.
    filler_spans = tuple(
        tuple(tokeniser(word).input_ids)
        for word in (*LEADING_FILLERS, *BRIDGING_FILLERS)
        if args.copy_only
    )

    tagger = None
    if args.tagger:
        from scripts.train_polish_tagger import features as tag_features
        from scripts.train_polish_tagger import token_spans

        saved = torch.load(str(args.tagger), map_location=args.device, weights_only=False)
        tagger = torch.nn.Linear(saved["hidden"], 2).to(args.device)
        tagger.load_state_dict(saved["state_dict"])
        tagger.eval()
        tag_lookahead = saved["lookahead"]
        # Absent in heads trained before the repetition columns existed, and
        # zero there means the old width -- so an old head still loads.
        tag_repetition = saved.get("repetition", 0)

    written = []
    latencies = []
    for index, row in enumerate(rows, start=1):
        if tagger is not None:
            started = time.perf_counter()
            ids, spans = token_spans(tokeniser, row["source"])
            with torch.no_grad():
                matrix = tag_features(model, ids, torch, args.device, tag_lookahead, tag_repetition)
                probability = torch.softmax(tagger(matrix), dim=-1)[:, 1]
            drop = {
                position
                for (start, end), keep in zip(spans, probability.tolist(), strict=True)
                if keep >= args.tagger_threshold
                for position in range(start, min(end, len(row["source"])))
            }
            text = "".join(c for i, c in enumerate(row["source"]) if i not in drop)
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)
            scored: dict[str, Any] = {}
            if not args.latency_only:
                arrived, removed = filler_removed(row, text)
                scored = {
                    "cer_before": character_error_rate(row["target"], row["source"]),
                    "cer_after": character_error_rate(row["target"], text),
                    "content_kept": content_kept(row["target"], text),
                    "invented": invented(row["target"], text),
                    "filler_arrived": arrived,
                    "filler_removed": removed,
                }
            written.append({**row, "polished": text, **scored, "elapsed_ms": elapsed})
            if index % 20 == 0:
                print(f"  {index}/{len(rows)}", flush=True)
            continue
        messages = [{"role": "user", "content": build_prompt(row["source"])}]
        prompt = str(
            tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        ids = torch.tensor([tokeniser(prompt).input_ids], device=args.device)
        started = time.perf_counter()
        with torch.no_grad():
            if args.copy_only:
                tail = copy_only_generate(
                    model,
                    ids,
                    list(tokeniser(row["source"]).input_ids),
                    int(tokeniser.eos_token_id),
                    args.max_new_tokens,
                    torch,
                    args.copy_lookahead,
                    filler_spans,
                    args.protect_head,
                )
            else:
                # Greedy, because the product's answer to the same dictation
                # must not change between attempts -- the same default the
                # sampling round settled on.
                produced = model.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=args.repetition_penalty,
                    # Named rather than left to the default: the answer ends at
                    # the end-of-turn token, and a run that does not stop there
                    # scores whatever the model said afterwards as invention.
                    eos_token_id=tokeniser.eos_token_id,
                    pad_token_id=tokeniser.pad_token_id or 0,
                )
                tail = produced[0][ids.shape[1] :].tolist()
        elapsed = (time.perf_counter() - started) * 1000
        text = tokeniser.decode(tail, skip_special_tokens=True).strip()
        if args.project:
            text = project_onto(row["source"], text)
        latencies.append(elapsed)
        scored: dict[str, Any] = {}
        if not args.latency_only:
            arrived, removed = filler_removed(row, text)
            scored = {
                "cer_before": character_error_rate(row["target"], row["source"]),
                "cer_after": character_error_rate(row["target"], text),
                "content_kept": content_kept(row["target"], text),
                "invented": invented(row["target"], text),
                "filler_arrived": arrived,
                "filler_removed": removed,
            }
        written.append({**row, "polished": text, **scored, "elapsed_ms": elapsed})
        if index % 20 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    latency = {
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
    }
    if args.latency_only:
        print(
            f"n={len(written)}  整段延迟 中位 {latency['latency_ms_median']:.0f} ms / "
            f"P95 {latency['latency_ms_p95']:.0f} ms（只计时，不打分）"
        )
        _write(args, written, {"checkpoint": args.checkpoint.name, "n": len(written), **latency})
        return

    before = assess(
        "cer_before", [r["cer_before"] for r in written], effect_of_interest=POLISHED_CER
    )
    after = assess("cer_after", [r["cer_after"] for r in written], effect_of_interest=POLISHED_CER)
    arrived = sum(r["filler_arrived"] for r in written)
    removed = sum(r["filler_removed"] for r in written)
    kept = statistics.mean(r["content_kept"] for r in written)
    made_up = statistics.mean(r["invented"] for r in written)
    empty = sum(1 for r in written if not r["polished"].strip())

    verdicts = {
        "主判据 CER": (
            f"过（{after.value:.4f} ≤ {POLISHED_CER}，输入端 {before.value:.4f}）"
            if after.value <= POLISHED_CER
            else f"不过（{after.value:.4f} > {POLISHED_CER}，输入端 {before.value:.4f}）"
        ),
        "口语词清除": (
            f"过（{removed / arrived:.3f} ≥ {FILLER_REMOVED}）"
            if arrived and removed / arrived >= FILLER_REMOVED
            else (
                f"不过（{removed / max(1, arrived):.3f} < {FILLER_REMOVED}）"
                if arrived
                else "无法判定（没有口语词进到输入）"
            )
        ),
        "内容保留": (
            f"过（{kept:.4f} ≥ {CONTENT_KEPT}）"
            if kept >= CONTENT_KEPT
            else f"不过（{kept:.4f} < {CONTENT_KEPT}）——**模型在删内容**"
        ),
        "不编造": (
            f"过（多出来的字占原文 {made_up:.4f} ≤ {INVENTED}）"
            if made_up <= INVENTED
            else f"不过（{made_up:.4f} > {INVENTED}）"
        ),
        "不塌": f"空输出 {empty}/{len(written)}",
    }

    report = {
        "checkpoint": args.checkpoint.name,
        "split": args.split,
        "n": len(written),
        "cer_before": before.value,
        "cer_after": after.value,
        "cer_noise_floor": after.noise_floor,
        "filler_arrived": arrived,
        "filler_removed": removed,
        "filler_removed_rate": removed / arrived if arrived else None,
        "content_kept": kept,
        "invented": made_up,
        "empty": empty,
        **latency,
        "thresholds": {
            "polished_cer": POLISHED_CER,
            "filler_removed": FILLER_REMOVED,
            "content_kept": CONTENT_KEPT,
            "invented": INVENTED,
        },
        "verdicts": verdicts,
    }
    print(
        f"n={len(written)}  CER {before.value:.4f} → {after.value:.4f} ± {after.noise_floor:.4f}\n"
        f"  口语词 {removed}/{arrived} 清掉，内容保留 {kept:.4f}，多出来的字 {made_up:.4f}\n"
        f"  整段延迟 中位 {report['latency_ms_median']:.0f} ms / P95 "
        f"{report['latency_ms_p95']:.0f} ms（多的这一跳要提前告诉前端）"
    )
    for name, line in verdicts.items():
        print(f"判据 {name}：{line}")

    _write(args, written, report)


def _write(args: Any, written: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in written) + "\n",
            encoding="utf-8",
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
