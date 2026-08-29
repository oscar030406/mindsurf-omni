"""What the polisher does to Cantonese and Japanese that hold a repetition.

The multilingual claim in the guide was that yue/ja/ko pass through untouched:
`worth_polishing` was said to be constantly false for them. It is not. The gate
counts a repetition only where the halves hold a CJK character, and Cantonese is
written in them while Japanese uses them alongside kana -- so 我哋我哋 and
会議会議 reach the model, and only Hangul is genuinely out of reach.

The 44-probe multilingual set could not see this: its five yue/ja sentences all
open with a filler and none of them repeats, so all five took the pass-through
branch and the reading said nothing about the branch that does run.

This measures the branch that does run. Every probe carries the answer a native
speaker would give, and the two kinds fail in opposite directions:

- ``stutter`` -- the speaker said a word twice. Removing one copy is right;
  leaving both is a miss, and the cost is a transcript that reads badly.
- ``legit_reduplication`` -- the doubling is the word (Cantonese 啱啱 is "just
  now") or a coincidence of two words meeting (Japanese compound boundaries).
  Removing a copy destroys the sentence, and the Chinese stage guards these with
  a vocabulary that holds no Cantonese and no Japanese.

Both are reported. A stage that scores well on one by sacrificing the other has
not passed.

    python scripts/measure_multilingual_repeats.py \\
        --probes configs/polish_probes_repetition_ml_v1.jsonl \\
        --checkpoint out/sft_polish6_768.pth --minimind-root ../minimind-o \\
        --tagger out/polish_tagger_unionchar.pt \\
        --tagger-backbone out/polish_tagger_unionchar_backbone.pth \\
        --report artifacts/multilingual-repeats.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import (  # noqa: E402
    PUNCTUATION,
    Polisher,
    worth_polishing,
)


def bare(text: str) -> str:
    """The text without punctuation, which is what the gate tests."""
    return "".join(char for char in text if char not in PUNCTUATION)


def verdict(probe: dict[str, Any], got: str) -> str:
    """How this one came out, in the terms the probe was written in.

    Four outcomes rather than right/wrong, because a stutter left alone and a
    real word destroyed are not the same failure and must not average together.
    """
    wanted = probe["wanted"]
    source = probe["source"]
    if probe["kind"] == "legit_reduplication" or probe["kind"] == "control":
        return "held" if got == wanted else "damaged"
    if got == wanted:
        return "repaired"
    return "missed" if got == source else "damaged"


async def run(
    polisher: Polisher, probes: list[dict[str, Any]], tell_language: bool
) -> list[dict[str, Any]]:
    """Every probe through the real stage, on one event loop.

    One loop for all of them: a fresh asyncio.run per probe leaves the polisher's
    worker bound to a closed loop and the next call hangs with the card idle,
    which reads as slowness rather than as a hang.

    ``tell_language`` is the arm switch. Withholding the language reproduces
    what the service did before it carried one -- the transcribe result had it
    and dropped it into `_` a line above the polish call -- so both arms come
    out of one run and the difference is attributable.
    """
    rows = []
    for index, probe in enumerate(probes, 1):
        started = time.perf_counter()
        got = await polisher.polish(probe["source"], probe["lang"] if tell_language else None)
        elapsed = (time.perf_counter() - started) * 1000
        rows.append(
            {
                **probe,
                "got": got,
                "verdict": verdict(probe, got),
                # Whether the gate let this one reach the model at all. A probe
                # that never reached it says nothing about the model, and the
                # old reading was built entirely out of those.
                "gate_opened": worth_polishing(probe["source"]),
                "characters_removed": len(bare(probe["source"])) - len(bare(got)),
                "ms": round(elapsed, 1),
            }
        )
        print(
            f"  {index}/{len(probes)} {rows[-1]['verdict']:>9}  {probe['source'][:34]}", flush=True
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for language in sorted({row["lang"] for row in rows}):
        here = [row for row in rows if row["lang"] == language]
        stutters = [row for row in here if row["kind"] == "stutter"]
        legit = [row for row in here if row["kind"] != "stutter"]
        report[language] = {
            "probes": len(here),
            "gate_opened": sum(1 for row in here if row["gate_opened"]),
            "stutter": {
                "n": len(stutters),
                "repaired": sum(1 for row in stutters if row["verdict"] == "repaired"),
                "missed": sum(1 for row in stutters if row["verdict"] == "missed"),
                "damaged": sum(1 for row in stutters if row["verdict"] == "damaged"),
            },
            "legit_or_control": {
                "n": len(legit),
                "held": sum(1 for row in legit if row["verdict"] == "held"),
                "damaged": sum(1 for row in legit if row["verdict"] == "damaged"),
            },
            # The number this whole exercise is about: a real word taken apart.
            "damaged_examples": [
                {"source": row["source"], "got": row["got"], "why": row.get("why", "")}
                for row in here
                if row["verdict"] == "damaged"
            ][:6],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tagger", type=Path)
    parser.add_argument("--tagger-backbone", type=Path)
    parser.add_argument("--tagger-threshold", type=float, default=0.4)
    parser.add_argument("--merge", default="veto")
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="only report which probes the gate lets through, without loading the model",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    probes = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.gate_only:
        opened = [probe for probe in probes if worth_polishing(probe["source"])]
        print(f"{len(opened)}/{len(probes)} 条会进模型")
        for probe in probes:
            mark = "进模型" if worth_polishing(probe["source"]) else "原样通过"
            print(f"  {probe['lang']:4s} {mark}  {probe['source']}")
        return

    if not (args.checkpoint and args.minimind_root):
        raise SystemExit("要跑模型就得给 --checkpoint 和 --minimind-root")

    polisher = Polisher(
        checkpoint=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        minimind_root=args.minimind_root,
        device=args.device,
        tagger=args.tagger,
        tagger_backbone=args.tagger_backbone,
        tagger_threshold=args.tagger_threshold,
        merge_mode=args.merge,
    )
    polisher.load()

    async def both_arms() -> dict[str, list[dict[str, Any]]]:
        # One loop for both arms, same reason it is one loop within an arm.
        out = {}
        for arm, tell in (("语种没传（旧行为）", False), ("语种传了（现在）", True)):
            print(f"\n跑 {arm}")
            polisher.skipped_language = 0
            out[arm] = await run(polisher, probes, tell)
        return out

    arms = asyncio.run(both_arms())
    report = {
        "probes": str(args.probes),
        "checkpoint": str(args.checkpoint),
        "tagger": str(args.tagger) if args.tagger else None,
        "merge": args.merge,
        "arms": {arm: {"by_language": summarise(rows), "rows": rows} for arm, rows in arms.items()},
    }

    print()
    for arm, rows in arms.items():
        broken = sum(1 for row in rows if row["verdict"] == "damaged")
        print(f"{arm}: {len(rows)} 条，弄坏 {broken} 条")
        for language, got in summarise(rows).items():
            print(
                f"    {language}: 口吃 修好 {got['stutter']['repaired']}/{got['stutter']['n']}"
                f" | 合法叠词 保住 {got['legit_or_control']['held']}/{got['legit_or_control']['n']}"
                f"，弄坏 {got['legit_or_control']['damaged']}"
            )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n写到 {args.report}")


if __name__ == "__main__":
    main()
