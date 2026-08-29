"""Teach the Thinker to clean a transcript without rewriting it.

The floor run fixed what this is for. On the production loop the polish job is
**89% deletion** (762 inserted filler characters against 96 substitutions), the
recogniser already punctuates, and its own error rate is 0.0063. So the target
here is: drop the spoken filler, leave everything else alone.

The failure to design against is not "misses a 那个". It is **rewriting** --
this model has a measured tendency to invent (13% to 42% fabrication under a
system prompt, see the field-comparison round), and a task that is mostly
deletion trains that tendency directly if nothing holds it back. Two things do:
a share of the pairs have nothing to remove, and the acceptance criteria score
content retention separately from cleanliness.

Loss is on the reply span only, the same span the evaluation reads, so the
model is never scored for reproducing the instruction.

    python scripts/train_polish.py --checkpoint out/sft_merge_768.pth \
        --pairs artifacts/polish_train/pairs.jsonl --minimind-root ~/omni/minimind-o \
        --output out/sft_polish_768.pth --report artifacts/polish-train-2026-08-15.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import (  # noqa: E402
    INSTRUCTION,
    build_prompt,
    project_onto,
)
from mindsurf_omni.service.thinker import thinker_weights  # noqa: E402

# The four helpers below moved here from train_dpo.py and measure_chat_loss.py
# when those files left with the assistant line; the polish trainers are their
# only remaining callers.

# A saved update has to clear the fp16 grid by a margin, or the save rounds it
# away entirely. Four ULPs is arbitrary but the order of magnitude is not.
MINIMUM_ULPS = 4.0


def storage_resolution(scale: float) -> float:
    """The fp16 spacing at a given weight magnitude.

    Every checkpoint in this project is stored at fp16, so an update smaller
    than this is not merely small -- it is unrepresentable in the format the
    comparison will be made in, and reads downstream as no effect at all.
    """
    import torch

    step = torch.tensor(scale, dtype=torch.float16)
    above = torch.nextafter(step, torch.tensor(float("inf"), dtype=torch.float16))
    return float(above - step)


def load_thinker(checkpoint: Path, args: argparse.Namespace, trainable: bool) -> Any:
    from mindsurf_omni.service.thinker import ThinkerGenerator

    generator = ThinkerGenerator(
        checkpoint=checkpoint,
        tokenizer_dir=args.tokenizer,
        minimind_root=args.minimind_root,
        device=args.device,
        variant=args.variant,
    )
    generator.load()
    model = generator._model  # noqa: SLF001
    if not trainable:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model, generator._tokenizer  # noqa: SLF001


def reply_span(prefix_ids: list[int], full_ids: list[int]) -> slice | None:
    """Where the assistant's reply sits in the full sequence, or None.

    Tokenisers are free to merge across the boundary between the template and
    the reply, and when they do, `len(prefix)` is the wrong offset -- scoring
    there charges the model for a token it was given. Returning None on any
    disagreement makes that a dropped sample instead of a quiet error.
    """
    if len(full_ids) <= len(prefix_ids):
        return None
    if full_ids[: len(prefix_ids)] != prefix_ids:
        return None
    return slice(len(prefix_ids), len(full_ids))


def encode(tokeniser: Any, source: str, target: str) -> tuple[list[int], slice] | None:
    """Token ids for one pair, and the span the answer occupies.

    None when the prompt boundary does not survive tokenisation -- a dropped
    pair rather than a loss computed over the wrong positions.
    """
    messages = [{"role": "user", "content": build_prompt(source)}]
    prefix = str(
        tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    )
    prefix_ids = list(tokeniser(prefix).input_ids)
    # The end-of-turn token is part of the answer. Without it the model is
    # never shown where an answer stops, and at generation time it does not
    # stop: the first run of this script left it off and a fifth of the
    # outputs ran on -- the sentence repeated, or the model answered the
    # question it had been asked to tidy.
    full = list(tokeniser(prefix + target + str(tokeniser.eos_token)).input_ids)
    span = reply_span(prefix_ids, full)
    return (full, span) if span is not None else None


def span_loss(model: Any, ids: Any, span: slice) -> Any:
    """Cross-entropy over the answer tokens only."""
    import torch

    logits = model(input_ids=ids).logits.float()
    targets = ids[0, span]
    # Predicting position i from i-1, the same offset the evaluation uses.
    predicted = logits[0, span.start - 1 : span.stop - 1]
    return torch.nn.functional.cross_entropy(predicted, targets)


def evaluate(model: Any, examples: list[tuple[list[int], slice]], device: str) -> float:
    import torch

    model.eval()
    total = 0.0
    with torch.no_grad():
        for ids, span in examples:
            total += float(span_loss(model, torch.tensor([ids], device=device), span))
    model.train()
    return total / max(1, len(examples))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="omni checkpoint to start from"
    )
    parser.add_argument("--pairs", type=Path, required=True, help="from build_polish_pairs.py")
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--variant", default="mindsurf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--accumulate", type=int, default=8, help="pairs per optimiser step")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="pairs longer than this are dropped"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--project-targets",
        action="store_true",
        help="train on the deletion-only projection of the clean text rather "
        "than the clean text itself. The decoder can only delete, so a target "
        "carrying characters the transcript never said is a target it cannot "
        "reach -- and reaching for it is what makes it skip content",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    policy, tokeniser = load_thinker(args.checkpoint, args, trainable=True)
    policy.train()

    train: list[tuple[list[int], slice]] = []
    heldout: list[tuple[list[int], slice]] = []
    dropped = 0
    unreachable = 0
    for row in rows:
        target = row["target"]
        if args.project_targets:
            projected = project_onto(row["source"], target)
            unreachable += projected != target
            target = projected
        encoded = encode(tokeniser, row["source"], target)
        if encoded is None or len(encoded[0]) > args.max_tokens:
            dropped += 1
            continue
        (heldout if row.get("split") == "val" else train).append(encoded)
    if not train:
        raise SystemExit("训练集是空的——检查 --pairs 和 --max-tokens")
    print(
        f"训练 {len(train)} 对，留出 {len(heldout)} 对，丢弃 {dropped} 对"
        f"（切分不开或超过 {args.max_tokens} token）",
        flush=True,
    )
    if args.project_targets:
        print(
            f"  目标投影到转写上：{unreachable}/{len(rows)} 对的干净原文不是转写的子序列，"
            "那部分（识别器听错的字）从训练目标里去掉了",
            flush=True,
        )

    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    generator = random.Random(args.seed)
    history: list[dict[str, float]] = []
    best: tuple[float, int, dict] | None = None
    if heldout:
        before = evaluate(policy, heldout, args.device)
        print(f"训练前留出 loss {before:.4f}", flush=True)
        history.append({"epoch": 0, "heldout_loss": before})

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        generator.shuffle(order)
        total = 0.0
        for step, index in enumerate(order, start=1):
            ids, span = train[index]
            loss = span_loss(policy, torch.tensor([ids], device=args.device), span)
            (loss / args.accumulate).backward()
            total += float(loss.detach())
            if step % args.accumulate == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(order)} loss {total / step:.4f}", flush=True)
        record = {"epoch": epoch, "loss": total / len(order)}
        if heldout:
            record["heldout_loss"] = evaluate(policy, heldout, args.device)
            print(
                f"epoch {epoch} loss {record['loss']:.4f} 留出 {record['heldout_loss']:.4f}",
                flush=True,
            )
        history.append(record)
        # Kept by held-out loss, not by being last. All four runs on 2026-08-24
        # reached their minimum at epoch 4 and rose at epoch 5, and the script
        # wrote the epoch-5 weights: the loop was already measuring the thing it
        # then threw away. With no held-out pairs there is nothing to choose on,
        # so the last epoch stays the answer.
        if heldout and (best is None or record["heldout_loss"] < best[0]):
            best = (
                record["heldout_loss"],
                epoch,
                {key: value.detach().clone() for key, value in policy.state_dict().items()},
            )

    if best is not None and best[1] != args.epochs:
        print(f"留出最低在 epoch {best[1]}（{best[0]:.4f}），写它而不是最后一个", flush=True)
        policy.load_state_dict(best[2])

    # The parent's tensors with the Thinker replaced, so anything measured
    # afterwards belongs to the text half and the Talker is bit-identical to
    # what it was fitted to.
    parent = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    tuned = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    merged = dict(parent)
    wanted = thinker_weights(parent)
    replaced = 0
    displacement = 0.0
    for key in wanted:
        if key in tuned:
            merged[key] = tuned[key]
            displacement = max(displacement, float((tuned[key] - parent[key].float()).abs().max()))
            replaced += 1
    if replaced != len(wanted):
        raise SystemExit(
            f"only {replaced} of {len(wanted)} Thinker tensors were replaced; "
            "writing a half-tuned checkpoint would look exactly like a tuned one"
        )
    scale = float(torch.cat([parent[key].float().flatten() for key in wanted]).pow(2).mean().sqrt())
    ulp = storage_resolution(scale)
    print(f"最大权重位移 {displacement:.3e}（fp16 间距的 {displacement / ulp:.1f} 倍）")
    if displacement < MINIMUM_ULPS * ulp:
        print(
            "⚠️ 位移低于 fp16 存储分辨率的门槛——下游任何比较都会读成「没有效应」，"
            "而那不是关于润色的结论。"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(args.output))
    print(f"替换 {replaced} 个 Thinker 张量，写入 {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint.name,
                    "pairs_trained": len(train),
                    "pairs_heldout": len(heldout),
                    "pairs_dropped": dropped,
                    "instruction": INSTRUCTION,
                    "project_targets": args.project_targets,
                    "targets_projected": unreachable if args.project_targets else 0,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "history": history,
                    "max_weight_displacement": displacement,
                    "fp16_ulp_at_weight_scale": ulp,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
