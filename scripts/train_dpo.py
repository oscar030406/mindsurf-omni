"""Direct preference optimisation on the Thinker, written because nothing had it.

Neither repository has a DPO implementation. minimind-o's trainer does SFT
only; the sister project carries `configs/training/dpo.yaml` complete with a
beta and no loss to go with it. So this is written rather than ported, and the
decisions that are easy to get quietly wrong are stated here.

**Only the Thinker moves.** The checkpoint is an omni model, but preference is
about what the model says, and the Talker is fitted to a specific Thinker's
hidden states -- moving both at once would mean two changes and one number. The
Talker and audio_proj tensors are copied through untouched, so the product
differs from its parent in the text half and nowhere else. That the Talker's
fit is invalidated by any Thinker change at all is a real problem and is not
solved here; it is measured, by the guardrails in the note.

**Only the reply is scored.** The prompt is identical in both branches of a
pair, so including it adds the same constant to both log-probabilities and
dilutes the difference the loss is built on. The boundary comes from
`measure_chat_loss.reply_span`, which is the same function the evaluation uses
and already has tests for the case where the tokeniser merges across the join.

**Log-probabilities are summed, not averaged.** DPO's derivation is over
sequence likelihood. Averaging per token silently reweights by length and
turns the objective into a preference for short replies -- which a judge that
rewards completeness would then have to fight.

**The reference model is frozen and detached.** It is a copy of the starting
checkpoint, in eval mode, with gradients off. A reference that drifts is not a
reference, and the failure is quiet: training still runs and the KL term stops
constraining anything.
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

from scripts.measure_chat_loss import reply_span  # noqa: E402

# DPO's temperature. Larger keeps the policy nearer the reference; the usual
# starting point, and small data is the reason to start conservative.
BETA = 0.1


def encode_pair(
    tokeniser: Any, prompt: str, chosen: str, rejected: str
) -> tuple[list[int], slice, list[int], slice] | None:
    """Both branches as token ids, each with the span its reply occupies.

    Returns None when either branch's prompt boundary does not survive
    tokenisation, which is a sample dropped rather than a span scored at the
    wrong offset.
    """
    messages = [{"role": "user", "content": prompt}]
    prefix = str(
        tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    )
    prefix_ids = list(tokeniser(prefix).input_ids)

    spans = []
    for reply in (chosen, rejected):
        full = list(tokeniser(prefix + reply).input_ids)
        span = reply_span(prefix_ids, full)
        if span is None:
            return None
        spans.append((full, span))
    return spans[0][0], spans[0][1], spans[1][0], spans[1][1]


def sequence_logprob(model: Any, ids: Any, span: slice) -> Any:
    """Summed log-probability of the reply tokens under this model."""
    import torch

    logits = model(input_ids=ids).logits.float()
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    # Predicting position i from i-1, so the reply's first token is read off
    # the last prompt token -- the same offset the evaluation uses.
    targets = ids[0, span]
    chosen = logp[0, span.start - 1 : span.stop - 1].gather(-1, targets.unsqueeze(-1))
    return chosen.sum()


def dpo_loss(
    policy_chosen: Any, policy_rejected: Any, reference_chosen: Any, reference_rejected: Any
) -> Any:
    """The standard objective, with the reference differences detached.

    Signs are the place this goes wrong silently: a flipped one trains the model
    to prefer the rejected branch and the loss still descends, because it is
    descending toward the wrong optimum.
    """
    import torch

    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -torch.nn.functional.logsigmoid(BETA * margin)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--triples", type=Path, required=True, help="from build_preference_pairs")
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--variant", default="mindsurf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--accumulate", type=int, default=8, help="pairs per optimiser step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="omni checkpoint to write")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch

    from mindsurf_omni.service.thinker import thinker_weights

    triples = json.loads(args.triples.read_text(encoding="utf-8"))["triples"]
    if not triples:
        raise SystemExit(f"no triples in {args.triples}")

    policy, tokeniser = load_thinker(args.checkpoint, args, trainable=True)
    reference, _ = load_thinker(args.checkpoint, args, trainable=False)

    encoded = []
    dropped = 0
    for triple in triples:
        pair = encode_pair(tokeniser, triple["prompt"], triple["chosen"], triple["rejected"])
        if pair is None:
            dropped += 1
            continue
        encoded.append(pair)
    if not encoded:
        raise SystemExit("every pair failed to encode; the prompt boundary never lined up")
    print(f"{len(encoded)} 对可用，丢弃 {dropped} 对（前缀边界对不齐）")

    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        optimiser.zero_grad(set_to_none=True)
        wins = 0
        total = 0.0
        for step, index in enumerate(order, start=1):
            chosen_ids, chosen_span, rejected_ids, rejected_span = encoded[index]
            chosen = torch.tensor([chosen_ids], device=args.device)
            rejected = torch.tensor([rejected_ids], device=args.device)

            with torch.no_grad():
                reference_chosen = sequence_logprob(reference, chosen, chosen_span)
                reference_rejected = sequence_logprob(reference, rejected, rejected_span)
            policy_chosen = sequence_logprob(policy, chosen, chosen_span)
            policy_rejected = sequence_logprob(policy, rejected, rejected_span)

            loss = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
            (loss / args.accumulate).backward()
            total += float(loss)
            # The share of pairs the policy already orders correctly. It is the
            # only progress signal that means anything here: the loss falls
            # even when the ordering does not change.
            wins += int(float(policy_chosen - policy_rejected) > 0)

            if step % args.accumulate == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(order)} loss {total / step:.4f} "
                    f"ordered {wins / step:.1%}"
                )
        history.append({"epoch": epoch, "loss": total / len(order), "ordered": wins / len(order)})

    # Write an omni checkpoint: the parent's tensors with the Thinker replaced,
    # so the Talker and audio_proj are bit-identical to what they were fitted
    # to and any measured change belongs to the text half.
    parent = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    tuned = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    merged = dict(parent)
    replaced = 0
    for key in thinker_weights(parent):
        if key in tuned:
            merged[key] = tuned[key].half()
            replaced += 1
    if replaced != len(thinker_weights(parent)):
        raise SystemExit(
            f"only {replaced} of {len(thinker_weights(parent))} Thinker tensors were replaced; "
            "writing a half-tuned checkpoint would look exactly like a tuned one"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(args.output))
    print(f"替换 {replaced} 个 Thinker 张量，写入 {args.output}")

    report = {
        "checkpoint": args.checkpoint.name,
        "pairs": len(encoded),
        "dropped_pairs": dropped,
        "beta": BETA,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "history": history,
        "thinker_tensors_replaced": replaced,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
