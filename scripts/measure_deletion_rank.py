"""At the moment it has to choose, does the model prefer to keep or to skip?

Recall on the unambiguous half is measurable and short of a human editor's. Two
causes read identically and want opposite fixes: the model knows and greedy
throws it away, in which case a bias at the decode recovers it and nothing is
retrained; or the model does not know, in which case the bias buys nothing.

**The first version of this script was wrong and its numbers were discarded.**
It built a reference target, walked character indices, and used those directly
as token indices -- this tokeniser runs about 0.84 tokens per character on
Chinese, so the probe drifted further off the intended step the deeper into the
text it went. It read argmax accuracy on fillers as 0.4275 while the assembled
stage recalls 0.778 of them, and a decode cannot beat its own argmax. The
contradiction is what caught it.

This version asks the question where the decode actually asks it. Under the copy
constraint each step chooses among source tokens at or after the pointer, so the
whole decision at a filler is between two of them:

* **keep** -- the token the filler starts with, which advances the pointer by one
  and writes the filler out;
* **skip** -- the token the text after the filler starts with, which advances the
  pointer past it and deletes it.

Build the prefix the decode would be holding (the source with every earlier
wanted deletion already applied), read the next-token distribution, restrict it
to what the constraint allows, and compare those two. No index arithmetic, and
the comparison is between two real tokens rather than against a reconstructed
position.

    python scripts/measure_deletion_rank.py --data <pairs> --split val \\
        --checkpoint out/sft_polish6_768.pth --minimind-root ~/omni/minimind-o
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.measure_group_length import (  # noqa: E402
    filler_occurrences,
    repetition_occurrences,
)
from mindsurf_omni.service.polish import build_prompt  # noqa: E402
from mindsurf_omni.service.thinker import ThinkerGenerator  # noqa: E402


def wanted(source: str) -> list[tuple[int, int, str]]:
    """(start, stop, kind) for every span the detector is confident about."""
    out = [(span.start, span.stop, "filler") for span in filler_occurrences(source)]
    for span in repetition_occurrences(source):
        half = len(span) // 2
        # Either copy is the same text; the second is the one a deletion takes.
        out.append((span.start + half, span.stop, "repeat"))
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--minimind-root", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=_ROOT / "assets/tokenizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch

    generator = ThinkerGenerator(
        checkpoint=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        minimind_root=args.minimind_root,
        device=args.device,
    )
    generator.load()
    model, tokeniser = generator._model, generator._tokenizer  # noqa: SLF001
    model.eval()

    rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("split", args.split) == args.split][: args.limit]
    print(f"{len(rows)} 段", flush=True)

    gaps: dict[str, list[float]] = {"filler": [], "repeat": []}
    wins: dict[str, int] = {"filler": 0, "repeat": 0}
    seen: dict[str, int] = {"filler": 0, "repeat": 0}
    for index, row in enumerate(rows):
        source = row["source"]
        spans = wanted(source)
        if not spans:
            continue
        messages = [{"role": "user", "content": build_prompt(source)}]
        prompt = str(
            tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        prompt_ids = list(tokeniser(prompt).input_ids)

        taken: set[int] = set()
        for start, stop, kind in spans:
            keep_text = source[start:]
            skip_text = source[stop:]
            if not skip_text:
                continue
            keep_ids = tokeniser(keep_text).input_ids
            skip_ids = tokeniser(skip_text).input_ids
            if not keep_ids or not skip_ids or keep_ids[0] == skip_ids[0]:
                # Same first token means the two choices are indistinguishable
                # at this step; the decision happens later and is not this one.
                continue
            written = "".join(c for i, c in enumerate(source[:start]) if i not in taken)
            ids = prompt_ids + list(tokeniser(written).input_ids) if written else prompt_ids
            with torch.no_grad():
                logits = model(torch.tensor([ids], device=args.device)).logits[0, -1]
            keep = float(logits[keep_ids[0]])
            skip = float(logits[skip_ids[0]])
            seen[kind] += 1
            gaps[kind].append(skip - keep)
            if skip > keep:
                wins[kind] += 1
            taken |= set(range(start, stop))
        if (index + 1) % 30 == 0:
            print(f"  {index + 1}/{len(rows)}，量了 {sum(seen.values())} 处", flush=True)

    report = {}
    print(f"\n{'':>8}{'决策点':>8}{'跳过赢':>9}{'差值中位':>10}{'差值P25':>10}{'差值P75':>10}")
    for kind in ("filler", "repeat"):
        got = gaps[kind]
        if not got:
            continue
        got_sorted = sorted(got)
        quarter = got_sorted[len(got_sorted) // 4]
        three = got_sorted[3 * len(got_sorted) // 4]
        share = wins[kind] / seen[kind]
        report[kind] = {
            "decisions": seen[kind],
            "skip_wins": share,
            "median_gap": statistics.median(got),
            "p25_gap": quarter,
            "p75_gap": three,
        }
        print(
            f"{kind:>8}{seen[kind]:>8}{share:>9.4f}"
            f"{statistics.median(got):>10.3f}{quarter:>10.3f}{three:>10.3f}"
        )

    print()
    print("怎么读：「跳过赢」是模型在这一步本来就想删的比例，贪心解码拿到的就是它。")
    print("差值是 skip 减 keep 的 logit 差——中位数只差一点点的话，一个小偏置就能")
    print("把大半个分布推过去；差得远，说明模型是真觉得该留，偏置只会伤到别处。")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
