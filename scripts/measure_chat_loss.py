"""Chat-format holdout loss: the instrument the text-ability question was missing.

The project could say what audio training did to prose -- `measure_strict_loss`
answers that, and A2A costs 1.52 nat of it. What it could not say is whether the
model answers worse, because prose continuation is not conversation, and the
only chat-format corpus on either machine looked like the training set.

It was not. `configs/talker_texts_zh_v1.jsonl` is 160 (prompt, reply) pairs
already used as the fixed-text CER benchmark, and checked against both training
parquets it overlaps by one prompt and one reply, with no pair sharing a row.
158 of 160 are clean. The holdout was sitting in configs the whole time; what
was missing was noticing that a benchmark's texts are also a holdout.

Two decisions make the number mean what it says.

**Only the reply is scored.** The prompt is identical in both arms, and letting
it into the average dilutes every difference toward zero -- an instrument that
reports "indistinguishable" because most of what it measured was shared input.
The prefix is tokenised separately and required to be a prefix of the whole,
byte for byte; a sample whose boundary does not line up is dropped and counted
rather than scored at the wrong offset.

**Per-sample values are kept, so two checkpoints can be paired.** Item
difficulty is most of the spread here -- some replies are simply more
predictable -- and on shared items it cancels. That is the same argument the
fixed-text CER protocol rests on, and it is what buys the resolution to see
0.05 nat.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import compare_paired  # noqa: E402

# The project's threshold for two comparable runs, in nats per token.
EFFECT_OF_INTEREST = 0.05


# Measured, not assumed: edge-tts over the 160 fixed texts reads Chinese at
# this rate (median of len(text) / audio_seconds, n=160, from
# artifacts/tts_edge/manifest.json). Recompute it if the synthesiser changes.
SPOKEN_CHARS_PER_SECOND = 4.67


def repetition(text: str, window: int = 8) -> float:
    """Share of `window`-grams that have appeared before, 0 when the text is short.

    A screen, not a score. This project has already shipped a repetition metric
    that gave a seven-character answer full marks, so this one is never
    compared between arms as a quality judgement -- it is here to catch a model
    that has started looping, which is the failure a likelihood number cannot
    see because a loop is highly probable.
    """
    if len(text) <= window:
        return 0.0
    grams = [text[i : i + window] for i in range(len(text) - window + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def generation_settings(args: Any) -> dict[str, Any] | None:
    """What produced the replies, or None when nothing was sampled.

    Recorded because it cannot be recovered afterwards. Extending this probe set
    had to answer "were the new replies sampled the way the old ones were", and
    the existing reports could not say: temperature, top-p, seed and the token
    cap all defaulted silently and none were written down. The cost is never
    picking a wrong value, it is being unable to tell afterwards which was
    picked.

    None rather than the defaults when --generate did not run, because writing
    a temperature for a pass that sampled nothing claims something untrue.
    """
    if not getattr(args, "generate", False):
        return None
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "system_prompt": str(args.system_prompt) if args.system_prompt else None,
    }


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


def score(checkpoint: Path, probes: list[dict[str, str]], args: argparse.Namespace) -> Any:
    import torch

    from mindsurf_omni.service.thinker import ThinkerGenerator

    generator = ThinkerGenerator(
        checkpoint=checkpoint,
        tokenizer_dir=args.tokenizer,
        minimind_root=args.minimind_root,
        device=args.device,
        variant=args.variant,
    )
    generator.load()
    model, tokeniser = generator._model, generator._tokenizer  # noqa: SLF001

    rows: list[dict[str, Any]] = []
    dropped: list[str] = []
    total_nll = 0.0
    total_entropy = 0.0
    total_tokens = 0

    for probe in probes:
        # A prompt set with no reference replies is a legitimate input -- it is
        # what preference sampling uses, where the model's own drafts are the
        # point and there is nothing to score them against yet. Such a set can
        # be generated from and not scored, rather than refused.
        if not probe.get("text"):
            continue
        messages = [{"role": "user", "content": probe["prompt"]}]
        prefix = str(
            tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        prefix_ids = list(tokeniser(prefix).input_ids)
        full_ids = list(tokeniser(prefix + probe["text"]).input_ids)
        span = reply_span(prefix_ids, full_ids)
        if span is None:
            dropped.append(probe["id"])
            continue

        ids = torch.tensor([full_ids], device=args.device)
        with torch.inference_mode():
            logits = model(input_ids=ids).logits.float()
        # Predicting position i from i-1: the reply's first token is scored
        # from the last prompt token, which the model was given.
        targets = ids[0, span]
        predictions = logits[0, span.start - 1 : span.stop - 1]
        nll = torch.nn.functional.cross_entropy(predictions, targets, reduction="sum")
        # The model's own entropy at the same positions, which is the control
        # for the reading this number invites. A checkpoint whose distribution
        # has sharpened pays more for every token it did not itself favour, so
        # it scores worse on ANY foreign text while being better on its own --
        # which looks exactly like "it got worse at conversation" and is not
        # the same claim. Reported beside the loss so the two can be told apart.
        logp = torch.nn.functional.log_softmax(predictions, dim=-1)
        entropy = float(-(logp.exp() * logp).sum(dim=-1).sum())

        count = int(targets.numel())
        total_nll += float(nll)
        total_entropy += entropy
        total_tokens += count
        rows.append(
            {
                "id": probe["id"],
                "nll": float(nll) / count,
                "entropy": entropy / count,
                "tokens": count,
            }
        )

    # Only the generation half takes it. Scoring compares likelihood against a
    # reference set that was written without any system prompt, so prepending
    # one there would change the conditioning on one side of a paired
    # comparison and the difference would read as a model change.
    system_prompt = (
        args.system_prompt.read_text(encoding="utf-8").strip() if args.system_prompt else ""
    )

    if args.generate:
        replies: list[dict[str, Any]] = []
        for probe in probes:
            # Generation needs only the prompt, so this loop covers every probe
            # including the ones the scoring loop above skipped.
            messages = [{"role": "user", "content": probe["prompt"]}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            torch.manual_seed(args.seed)
            text = ""
            import asyncio

            async def collect(messages: Any = messages) -> str:
                out = ""
                async for delta in generator.generate(messages, _settings(args)):
                    out += delta
                return out

            text = asyncio.run(collect())
            replies.append(
                {
                    "id": probe["id"],
                    "prompt": probe["prompt"],
                    "reply": text,
                    "chars": len(text),
                    "repetition": repetition(text),
                    "empty": not text.strip(),
                }
            )
        lengths = sorted(int(r["chars"]) for r in replies)
        screen = {
            "empty": sum(1 for r in replies if r["empty"]),
            "looping": sum(1 for r in replies if r["repetition"] > 0.5),
            "median_chars": statistics.median(lengths),
            "at_token_cap": sum(1 for r in replies if int(r["chars"]) >= args.max_tokens),
            # Characters are the wrong unit for a product nobody can skim. A
            # listener takes the reply at the synthesiser's pace, and 140
            # characters is half a minute of talking -- which no chat metric
            # here would have shown. The divisor is measured, not assumed:
            # edge-tts over the 160 fixed texts reads 4.67 characters a second
            # (artifacts/tts_edge/manifest.json). It is a rate for this
            # synthesiser, so treat the seconds as an estimate that moves if
            # the synthesiser does -- and remember characters only proxy
            # duration, since syllable count and prosody both move it.
            "median_spoken_seconds": statistics.median(lengths) / SPOKEN_CHARS_PER_SECOND,
            "p90_spoken_seconds": lengths[int(0.9 * len(lengths))] / SPOKEN_CHARS_PER_SECOND,
        }
        print(
            f"  生成筛查: 空 {screen['empty']}  疑似循环 {screen['looping']}  "
            f"中位长度 {screen['median_chars']:.0f} 字"
            f"（念出来约 {screen['median_spoken_seconds']:.0f} s，"
            f"P90 {screen['p90_spoken_seconds']:.0f} s）"
        )
    else:
        replies, screen = [], {}

    # What produced the replies, recorded because it cannot be recovered later.
    # The first attempt to extend this probe set had to answer "were the new
    # replies sampled the same way as the old ones", and the reports could not
    # say -- temperature, top-p, seed and the token cap all defaulted silently
    # and none of them were written down. The cost is not picking a wrong value,
    # it is being unable to tell afterwards which value was picked. The same
    # applies to training hyperparameters, one script over. Only present when
    # --generate ran; a likelihood
    # pass samples nothing.
    return {
        "checkpoint": checkpoint.name,
        "probes": str(args.probes),
        "generation": generation_settings(args),
        "scored": len(rows),
        "dropped_misaligned": dropped,
        # Token-weighted, the way strict_val is computed: long replies should
        # not count the same as short ones in the headline.
        "scored_of": len(probes),
        "chat_nll": total_nll / total_tokens if total_tokens else float("nan"),
        "entropy": total_entropy / total_tokens if total_tokens else float("nan"),
        "mean_of_samples": statistics.fmean(row["nll"] for row in rows) if rows else float("nan"),
        "samples": rows,
        "screen": screen,
        "replies": replies,
    }


def _settings(args: argparse.Namespace) -> Any:
    from mindsurf_omni.service.engine import GenerationSettings

    return GenerationSettings(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=Path("configs/talker_texts_zh_v1.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--variant",
        default="mindsurf",
        help="thinker shape+parameter pair; 'upstream-default' reads the released weights",
    )
    parser.add_argument("--only", type=Path, help="JSON list of ids to keep, e.g. the clean 158")
    parser.add_argument("--reference", type=Path, help="a previous report, to pair against")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="also answer each prompt and screen for gross defects -- a likelihood "
        "cannot see a loop, because a loop is probable",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        help="file whose text is prepended as a system turn, generation only. "
        "The speech-friendly prompt is configs/system_prompt_speech_zh.txt",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    probes = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.only:
        keep = set(json.loads(args.only.read_text(encoding="utf-8")))
        probes = [probe for probe in probes if probe["id"] in keep]
    if not probes:
        raise SystemExit("no probes left after filtering")

    report = score(args.checkpoint, probes, args)
    print(
        f"chat_nll {report['chat_nll']:.4f} entropy {report['entropy']:.4f} "
        f"over {report['scored']} replies"
    )
    if report["dropped_misaligned"]:
        dropped = report["dropped_misaligned"]
        print(f"  dropped {len(dropped)} for a prefix that did not line up: {dropped[:5]}")

    if args.reference:
        other = json.loads(args.reference.read_text(encoding="utf-8"))
        theirs = {row["id"]: row["nll"] for row in other["samples"]}
        deltas = [
            row["nll"] - theirs[row["id"]] for row in report["samples"] if row["id"] in theirs
        ]
        print(f"  reference  {other['checkpoint']}  chat_nll {other['chat_nll']:.4f}")
        print("  " + compare_paired("chat_nll", deltas, effect_of_interest=EFFECT_OF_INTEREST))
        print(f"  效应关心 {EFFECT_OF_INTEREST} nat/token")
        report["paired_against"] = other["checkpoint"]
        report["paired_verdict"] = compare_paired(
            "chat_nll", deltas, effect_of_interest=EFFECT_OF_INTEREST
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
