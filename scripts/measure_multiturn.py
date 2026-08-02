"""Run a conversation past turn one, which this project has never done.

Every quality number here was measured on a single exchange: the 158 chat
probes, the 160 fixed texts, all 1676 preference pairs. The contract carries
conversation history and the OKR worries about context length exploding, but
nobody has checked what the model does on turn 2 -- so "it holds a conversation"
is an assumption, not a finding.

This feeds the model its own replies back and asks the next question, which is
what a real session does. The probes are built so turns 2 and 3 cannot be
answered without the earlier ones: they use a pronoun, an ellipsis, or a
comparative whose referent exists only in the history. A model that silently
drops history therefore cannot score well by accident -- it will answer some
other question, and that is visible.

What comes out is per-turn: length, gross defects, and the rows a judge needs
to say whether the reply actually used the history. The judging is separate on
purpose; this half must be reproducible without a judge.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scripts.measure_chat_loss import (  # noqa: E402
    SPOKEN_CHARS_PER_SECOND,
    _settings,
    repetition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--variant", default="mindsurf")
    parser.add_argument("--probes", type=Path, default=Path("configs/multiturn_probes_zh_v1.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import asyncio

    import torch

    from mindsurf_omni.service.thinker import ThinkerGenerator

    probes = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    if not probes:
        raise SystemExit(f"no probes in {args.probes}")

    generator = ThinkerGenerator(
        checkpoint=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        minimind_root=args.minimind_root,
        device=args.device,
        variant=args.variant,
    )
    generator.load()

    conversations: list[dict[str, Any]] = []
    for index, probe in enumerate(probes, start=1):
        # The model's own replies go back in, not reference answers. Feeding
        # curated history would measure a conversation the model never had.
        messages: list[dict[str, str]] = []
        turns: list[dict[str, Any]] = []
        for position, question in enumerate(probe["turns"]):
            messages.append({"role": "user", "content": question})
            torch.manual_seed(args.seed + position)

            # Snapshot rather than the live list: the loop appends the reply to
            # `messages` right after, and a closure over the live list would
            # send a history that includes the answer being generated.
            history = list(messages)

            async def collect(history: list[dict[str, str]] = history) -> str:
                out = ""
                async for delta in generator.generate(history, _settings(args)):
                    out += delta
                return out

            reply = asyncio.run(collect())
            messages.append({"role": "assistant", "content": reply})
            turns.append(
                {
                    "turn": position + 1,
                    "question": question,
                    "reply": reply,
                    "chars": len(reply),
                    "spoken_seconds": len(reply) / SPOKEN_CHARS_PER_SECOND,
                    "repetition": repetition(reply),
                    "empty": not reply.strip(),
                }
            )
        conversations.append({"id": probe["id"], "turns": turns})
        if index % 10 == 0:
            print(f"  {index}/{len(probes)}", flush=True)

    by_turn: dict[int, list[dict[str, Any]]] = {}
    for conversation in conversations:
        for turn in conversation["turns"]:
            by_turn.setdefault(turn["turn"], []).append(turn)

    summary = {}
    print(f"\n{'轮':<4}{'条数':>6}{'中位字数':>10}{'朗读秒':>9}{'空':>5}{'循环':>6}")
    for position in sorted(by_turn):
        rows = by_turn[position]
        chars = [row["chars"] for row in rows]
        entry = {
            "n": len(rows),
            "median_chars": statistics.median(chars),
            "median_spoken_seconds": statistics.median(chars) / SPOKEN_CHARS_PER_SECOND,
            "empty": sum(1 for row in rows if row["empty"]),
            "looping": sum(1 for row in rows if row["repetition"] > 0.5),
        }
        summary[str(position)] = entry
        print(
            f"{position:<4}{entry['n']:>6}{entry['median_chars']:>10.0f}"
            f"{entry['median_spoken_seconds']:>9.1f}{entry['empty']:>5}{entry['looping']:>6}"
        )

    payload = {
        "checkpoint": args.checkpoint.name,
        "probes": str(args.probes),
        "by_turn": summary,
        "conversations": conversations,
        "note": (
            "the model's own replies are fed back as history; turns 2 and 3 are "
            "unanswerable without them, so whether history was used is visible in "
            "the text and is judged separately"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
