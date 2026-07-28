"""Does `chunk_frames` buy time to first audio? Measured in pairs, in process.

Two instruments already exist and neither can answer this.

`measure_latency.py` goes over HTTP, and `chunk_frames` is fixed when the
service is built -- sweeping it there means restarting between arms, which
puts minutes and a fresh CUDA context between the numbers being compared. The
2026-07-25 sweep ran in process and alternated, which is right, but it ran on
a contended laptop where load alone moved the reading by 3x, so a 42 ms
difference had nowhere to land.

The quiet card fixes the load. It does not fix the other half: time to first
audio tracks how much the model decides to say, and the model says something
different every turn. Against a whole-turn spread of that size a 20 ms knob
has nowhere to land.

So this script does to latency what the fixed-text protocol did to CER. Within
a round every arm is seeded identically, and `chunk_frames` touches only how
decoded frames are grouped -- never generation -- so all arms in a round emit
the same tokens in the same order and differ in exactly one thing. The
difference is then per-round paired, and the per-round difficulty that swamped
the earlier attempt subtracts out.

Running in process is not only convenience. `chunk_frames` is read from the
engine's config, so a service-side sweep costs a restart per arm; here it is
one assignment between rounds, with the model and its warm allocator held
still across all of them.

The pairing is checked, not assumed: a round whose arms disagree on the
generated text is dropped and counted. If the count is large the seeding did
not hold and the numbers mean nothing, which is a result worth printing rather
than a condition worth hiding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from mindsurf_omni.evaluation.metrics import assess, compare_paired
from mindsurf_omni.service.config import Settings
from mindsurf_omni.service.engine import GenerationSettings

DEFAULT_LEVELS = (1, 2, 4, 8)
BASELINE = 4


def load_stimulus(path: Path) -> tuple[bytes, int]:
    """One clip, reused for every round: the input is held still on purpose."""
    import soundfile

    data, rate = soundfile.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    return data.tobytes(), int(rate)


def seed_rng(seed: int) -> None:
    """Set the generator every arm starts from.

    Its own function so a test can stand in for it, and deliberately not
    wrapped in a try: if seeding does not happen the arms generate different
    replies and every paired number below is a subtraction of two unrelated
    turns. A missing torch here should stop the run, not soften it.
    """
    import torch

    torch.manual_seed(seed)


async def one_arm(
    engine: Any, pcm: bytes, rate: int, settings: GenerationSettings, seed: int
) -> tuple[float, str]:
    """Time to the first chunk carrying audio, plus the whole reply's text.

    The turn is read to the end even though the number wanted is at the front.
    Stopping early is now safe -- the engine stops the producer with its
    consumer, which is what this sweep found by accident and what the service
    was fixed to do -- but a stopped arm still ends somewhere in the shared
    torch generator, and drawing the next arm's seed from an unknown point is
    exactly what the pairing cannot survive. Draining costs a few hundred
    milliseconds per arm and buys arms that cannot contaminate each other.
    """
    seed_rng(seed)
    text = ""
    first_audio: float | None = None
    started = time.perf_counter()
    async for chunk in engine.respond(pcm, rate, settings):
        if chunk.text:
            text += chunk.text
        if chunk.pcm and first_audio is None:
            first_audio = (time.perf_counter() - started) * 1000
    return (first_audio if first_audio is not None else float("nan")), text


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from mindsurf_omni.service import factory

    settings = Settings.from_environment()
    engine = factory.build(settings)
    if engine is None:
        raise SystemExit("no engine: check MINDSURF_ENGINE and the model paths")

    pcm, rate = load_stimulus(args.stimulus)
    generation = GenerationSettings(temperature=args.temperature, top_p=args.top_p)
    levels = list(args.levels)

    readings: dict[int, list[float]] = {level: [] for level in levels}
    paired: dict[int, list[float]] = {level: [] for level in levels}
    dropped = 0

    for index in range(args.rounds):
        seed = args.seed + index
        # Rotate the order so a warm-up or thermal trend cannot sit on one arm.
        order = levels[index % len(levels) :] + levels[: index % len(levels)]
        round_ms: dict[int, float] = {}
        round_text: dict[int, str] = {}
        for level in order:
            engine._config.chunk_frames = level  # type: ignore[attr-defined]  # the knob
            elapsed, text = await one_arm(engine, pcm, rate, generation, seed)
            round_ms[level] = elapsed
            round_text[level] = text

        for level in levels:
            readings[level].append(round_ms[level])

        # The pairing claim, checked. Identical seeds should give identical
        # tokens; if they do not, this round is comparing two different replies
        # and its difference is not attributable to chunk_frames.
        if len({text for text in round_text.values()}) != 1:
            dropped += 1
        else:
            for level in levels:
                paired[level].append(round_ms[level] - round_ms[BASELINE])

        print(
            f"round {index + 1}/{args.rounds} "
            + "  ".join(f"{level}:{round_ms[level]:.0f}ms" for level in levels)
            + ("  [texts differ, dropped]" if len(set(round_text.values())) != 1 else "")
        )

    report: dict[str, Any] = {
        "rounds": args.rounds,
        "dropped_unpaired": dropped,
        "baseline_chunk_frames": BASELINE,
        "stimulus": str(args.stimulus),
        "levels": {},
        "paired_verdicts": [],
    }
    for level in levels:
        values = readings[level]
        report["levels"][str(level)] = {
            "mean_ms": statistics.fmean(values),
            "p50_ms": statistics.median(values),
            "note": assess(f"chunk_frames={level}", values, effect_of_interest=args.effect).note,
        }
    for level in levels:
        if level == BASELINE:
            continue
        report["paired_verdicts"].append(
            compare_paired(f"chunk_frames {level} vs {BASELINE}", paired[level])
        )

    print()
    for level in levels:
        entry = report["levels"][str(level)]
        print(
            f"chunk_frames={level:<2} mean {entry['mean_ms']:8.1f} ms"
            f"  p50 {entry['p50_ms']:8.1f} ms"
        )
    print()
    for line in report["paired_verdicts"]:
        print(" ", line)
    if dropped:
        print(f"\n{dropped}/{args.rounds} rounds dropped: arms generated different text")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimulus", type=Path, required=True, help="one 16 kHz wav")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--effect", type=float, default=200.0, help="ms worth acting on, per the OKR budget"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = asyncio.run(run(args))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
