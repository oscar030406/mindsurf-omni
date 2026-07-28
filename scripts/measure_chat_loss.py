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
    )
    generator.load()
    model, tokeniser = generator._model, generator._tokenizer  # noqa: SLF001

    rows: list[dict[str, Any]] = []
    dropped: list[str] = []
    total_nll = 0.0
    total_tokens = 0

    for probe in probes:
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

        count = int(targets.numel())
        total_nll += float(nll)
        total_tokens += count
        rows.append({"id": probe["id"], "nll": float(nll) / count, "tokens": count})

    return {
        "checkpoint": checkpoint.name,
        "probes": str(args.probes),
        "scored": len(rows),
        "dropped_misaligned": dropped,
        # Token-weighted, the way strict_val is computed: long replies should
        # not count the same as short ones in the headline.
        "chat_nll": total_nll / total_tokens if total_tokens else float("nan"),
        "mean_of_samples": statistics.fmean(row["nll"] for row in rows) if rows else float("nan"),
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=Path("configs/talker_texts_zh_v1.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", type=Path, help="JSON list of ids to keep, e.g. the clean 158")
    parser.add_argument("--reference", type=Path, help="a previous report, to pair against")
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
    print(f"chat_nll {report['chat_nll']:.4f} over {report['scored']} replies")
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
        print("  " + compare_paired("chat_nll", deltas))
        print(f"  效应关心 {EFFECT_OF_INTEREST} nat/token")
        report["paired_against"] = other["checkpoint"]
        report["paired_verdict"] = compare_paired("chat_nll", deltas)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
