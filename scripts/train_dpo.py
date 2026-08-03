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
import math
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

# An update smaller than a few times the fp16 spacing at the checkpoint's own
# weight scale cannot be told from no update by anything downstream -- every
# other checkpoint here is stored at that precision. Four is arbitrary but the
# order of magnitude is not: at one ULP a save would round it away entirely.
MINIMUM_ULPS = 4.0


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
) -> tuple[Any, Any]:
    """The standard objective and its margin, with the reference detached.

    Signs are the place this goes wrong silently: a flipped one trains the model
    to prefer the rejected branch and the loss still descends, because it is
    descending toward the wrong optimum.

    The margin comes back because it is the only honest progress signal. The
    raw difference of summed log-probabilities is not: two replies of different
    lengths have systematically different sums, so a counter built on it tracks
    which branch is shorter. The margin subtracts the reference, and the
    reference has the same length bias, so it cancels.
    """
    import torch

    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -torch.nn.functional.logsigmoid(BETA * margin), margin


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


def behavioural_evidence(history: list[dict[str, float]], heldout_pairs: int) -> str | None:
    """Did the run change behaviour on data it never trained on?

    The displacement guard below is a proxy: it asks whether the weights moved
    further than the format the other checkpoints are stored in can resolve. A
    proxy is what you use when the thing itself is not measurable, and here it
    is. ``heldout_ordered`` counts pairs whose preference margin *relative to
    the reference model* came out positive, on pairs never backpropagated
    through, so above chance means this checkpoint orders preferences
    differently from the one it started from. That is the claim the proxy was
    standing in for, measured directly.

    It matters because the proxy has been wrong in the expensive direction three
    times. The generalisation peak in every DPO round here lands at epoch 0,
    where displacement is smallest -- 2.83 times fp16 spacing with held-out
    ordering at 96.8%, the best of that run. The guard refused it and what
    shipped was the three-epoch checkpoint at 93.5%. A proxy may not overrule
    the quantity it proxies for.

    Chance is 0.5 and the floor is binomial, the same one
    ``cross_reference_agreement`` uses. Returns None when there is no held-out
    set or the ordering is not resolvably above chance -- then the proxy is all
    there is, and it decides.
    """
    if not heldout_pairs or not history:
        return None
    ordered = history[-1].get("heldout_ordered")
    if ordered is None:
        return None
    noise = 1.96 * math.sqrt(0.25 / heldout_pairs)
    if ordered - 0.5 <= noise:
        return None
    return f"留出排序 {ordered:.1%} 对随机的 50%（噪声 ±{noise:.1%}，n={heldout_pairs}）"


def evaluate(
    policy: Any, reference: Any, pairs: list[Any], args: argparse.Namespace
) -> dict[str, float]:
    """Loss and ordering on pairs never backpropagated through.

    With a few hundred pairs the in-sample and held-out numbers separate fast,
    and that separation is the overfitting the groundwork note predicts. An
    in-sample-only curve cannot show it.
    """
    import torch

    total = 0.0
    wins = 0
    with torch.no_grad():
        for chosen_ids, chosen_span, rejected_ids, rejected_span in pairs:
            chosen = torch.tensor([chosen_ids], device=args.device)
            rejected = torch.tensor([rejected_ids], device=args.device)
            loss, margin = dpo_loss(
                sequence_logprob(policy, chosen, chosen_span),
                sequence_logprob(policy, rejected, rejected_span),
                sequence_logprob(reference, chosen, chosen_span),
                sequence_logprob(reference, rejected, rejected_span),
            )
            total += float(loss.detach())
            wins += int(float(margin.detach()) > 0)
    return {"heldout_loss": total / len(pairs), "heldout_ordered": wins / len(pairs)}


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
    # 5e-7 was the first guess and it is below the noise: at ~50 optimiser
    # steps AdamW's displacement is bounded by lr x steps = 2.5e-5, against a
    # relative-L2 of 0.031 that this project has already measured as inert
    # (2026-07-26-weight-scale.md). The guard at write-out enforces the floor;
    # this default aims an order of magnitude above it.
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--accumulate", type=int, default=4, help="pairs per optimiser step")
    parser.add_argument(
        "--heldout",
        type=float,
        default=0.1,
        help="share of pairs never trained on; with a few hundred pairs the "
        "in-sample and held-out curves separate fast and that is the signal",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-tiny-update",
        action="store_true",
        help="write the checkpoint even when the update is below the storage resolution; "
        "for smoke runs, where producing a null is the point",
    )
    parser.add_argument("--output", type=Path, required=True, help="omni checkpoint to write")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch

    from mindsurf_omni.service.thinker import thinker_weights

    payload = json.loads(args.triples.read_text(encoding="utf-8"))
    triples = payload["triples"]
    if not triples:
        raise SystemExit(f"no triples in {args.triples}")
    # The on-policy invariant, re-asserted here because it is enforced two files
    # upstream and nothing in between carries it. Off-policy pairs make this a
    # well-behaved imitation run: the loss descends, the margin grows, and the
    # thing being learnt is not preference.
    source = payload.get("checkpoint")
    if source and source != args.checkpoint.name:
        raise SystemExit(
            f"the pairs were drawn from {source} and --checkpoint is {args.checkpoint.name}; "
            "DPO on another model's drafts trains imitation of that model, not preference"
        )

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
    rng = random.Random(args.seed)
    rng.shuffle(encoded)
    cut = int(len(encoded) * args.heldout)
    heldout, encoded = encoded[:cut], encoded[cut:]
    print(f"{len(encoded)} 对训练 + {len(heldout)} 对留出，丢弃 {dropped} 对（前缀边界对不齐）")
    if not encoded:
        raise SystemExit("held-out split left nothing to train on")

    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        optimiser.zero_grad(set_to_none=True)
        wins = 0
        total = 0.0
        chosen_logp = 0.0
        rejected_logp = 0.0
        for step, index in enumerate(order, start=1):
            chosen_ids, chosen_span, rejected_ids, rejected_span = encoded[index]
            chosen = torch.tensor([chosen_ids], device=args.device)
            rejected = torch.tensor([rejected_ids], device=args.device)

            with torch.no_grad():
                reference_chosen = sequence_logprob(reference, chosen, chosen_span)
                reference_rejected = sequence_logprob(reference, rejected, rejected_span)
            policy_chosen = sequence_logprob(policy, chosen, chosen_span)
            policy_rejected = sequence_logprob(policy, rejected, rejected_span)

            loss, margin = dpo_loss(
                policy_chosen, policy_rejected, reference_chosen, reference_rejected
            )
            (loss / args.accumulate).backward()
            total += float(loss.detach())
            # Ordering by margin, not by the raw log-probability difference --
            # see dpo_loss. At step zero this is 0% by construction, because
            # policy and reference are the same weights; anything else on the
            # first screen means they are not, which is worth knowing early.
            wins += int(float(margin.detach()) > 0)
            # DPO's characteristic collapse pushes BOTH branches down while the
            # margin still grows. Tracking only the margin cannot see it.
            chosen_logp += float(policy_chosen.detach())
            rejected_logp += float(policy_rejected.detach())

            if step % args.accumulate == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(order)} loss {total / step:.4f} "
                    f"ordered {wins / step:.1%} "
                    f"logp chosen {chosen_logp / step:.1f} rejected {rejected_logp / step:.1f}"
                )
        record = {
            "epoch": epoch,
            "loss": total / len(order),
            "ordered": wins / len(order),
            "mean_chosen_logp": chosen_logp / len(order),
            "mean_rejected_logp": rejected_logp / len(order),
        }
        if heldout:
            record.update(evaluate(policy, reference, heldout, args))
            print(
                f"epoch {epoch} 留出: loss {record['heldout_loss']:.4f} "
                f"ordered {record['heldout_ordered']:.1%}"
            )
        history.append(record)

    # Write an omni checkpoint: the parent's tensors with the Thinker replaced,
    # so the Talker and audio_proj are bit-identical to what they were fitted
    # to and any measured change belongs to the text half.
    parent = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    tuned = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
    merged = dict(parent)
    wanted = thinker_weights(parent)
    replaced = 0
    displacement = 0.0
    for key in wanted:
        if key in tuned:
            # Written at full precision, not cast back to the parent's fp16.
            # The whole run's budget is lr x steps; at 5e-6 x a few hundred
            # steps that is ~1e-3, while fp16's spacing at this weight scale
            # (per-parameter RMS 0.1366) is 6.1e-5 to 1.2e-4 -- close enough
            # that a cast eats a large share of it, and at smaller budgets all
            # of it. Both readers build an fp32 model and load_state_dict casts,
            # so a mixed-dtype checkpoint loads fine.
            merged[key] = tuned[key]
            displacement = max(displacement, float((tuned[key] - parent[key].float()).abs().max()))
            replaced += 1
    if replaced != len(wanted):
        raise SystemExit(
            f"only {replaced} of {len(wanted)} Thinker tensors were replaced; "
            "writing a half-tuned checkpoint would look exactly like a tuned one"
        )
    # Counting tensors proves nothing, and counting *changed* tensors proves
    # nearly as little once the write-out is fp32: any nonzero gradient makes
    # every tensor differ by something. What matters is whether the update is
    # larger than the resolution the rest of this project's checkpoints are
    # stored at -- an update that would not survive being written as fp16 is at
    # the noise floor of everything it will be compared against.
    moved = sum(1 for key in wanted if not torch.equal(merged[key], parent[key].float()))
    scale = float(torch.cat([parent[key].float().flatten() for key in wanted]).pow(2).mean().sqrt())
    ulp = storage_resolution(scale)
    behaviour = behavioural_evidence(history, len(heldout))
    if displacement < MINIMUM_ULPS * ulp and not args.allow_tiny_update:
        if behaviour is None:
            raise SystemExit(
                f"max |dw| is {displacement:.2e}, under {MINIMUM_ULPS} times the fp16 spacing "
                f"({ulp:.2e}) at this checkpoint's weight scale ({scale:.4f}). An update this "
                "small is indistinguishable from no update once stored, and every downstream "
                "comparison would read it as 'no measurable effect' for a reason that is not "
                "about preference. Raise --learning-rate or --epochs, pass --heldout to "
                "get behavioural evidence that can overrule this, or pass --allow-tiny-update "
                "if a null run is what you meant to produce."
            )
        print(
            f"位移 {displacement / ulp:.2f}× fp16 间距，低于 {MINIMUM_ULPS}× 的门槛，"
            f"但留出集有行为证据：{behaviour}。按证据落盘。"
        )
    print(f"最大权重位移 {displacement:.3e}（fp16 间距的 {displacement / ulp:.1f} 倍）")
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
        "thinker_tensors_changed": moved,
        "max_weight_displacement": displacement,
        "fp16_ulp_at_weight_scale": ulp,
        "heldout_pairs": len(heldout),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
