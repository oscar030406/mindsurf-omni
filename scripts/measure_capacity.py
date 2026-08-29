"""How many turns one card carries, and what the queue depth costs.

The backend says it will "handle by queueing to a limit". Nobody has measured
what that limit is, so the number in that sentence is currently a guess. This
produces the curve it should be read off: throughput and latency against the
number of requests in flight, for the two stages that own the card.

The stages are measured apart on purpose. They fail differently -- recognition
is one forward pass over a fixed-length buffer, generation is a loop whose
length depends on what the model decides to say -- and a combined number hides
which one saturates first. Anything the synthesiser adds is a third party's
latency and is not in here at all.

Read against a budget rather than as a maximum: the interesting point is the
concurrency where P95 crosses the line, not the concurrency where the card
falls over.

    python scripts/measure_capacity.py --base http://127.0.0.1:8002 \
        --stage asr --concurrency 1 2 4 8 16 --requests 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
# And the repository root: run as a script, sys.path[0] is scripts/ rather than
# the working directory, so the sibling module is not importable by package name.
sys.path.insert(0, str(_ROOT))


def speech_like(seconds: float, rate: int = 16_000) -> bytes:
    """Not silence: a VAD would never mark silence as the end of a turn.

    Moved here from measure_latency.py when that file left with the assistant
    line; this is its only remaining caller.
    """
    import math

    samples = [
        int(0.3 * 32767 * math.sin(2 * math.pi * 220 * index / rate))
        for index in range(int(rate * seconds))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def nearest_rank(values: list[float], quantile: float) -> float:
    """A latency some request actually experienced, not one interpolated between two."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(1, int(quantile * len(ordered) + 0.999999)) - 1
    return ordered[min(index, len(ordered) - 1)]


async def one_request(
    client: httpx.AsyncClient, stage: str, payload: Any
) -> tuple[float, str, int]:
    """Milliseconds, the failure if there was one, and how much was produced.

    The size matters as much as the clock on the generation stage: a curve
    measured on a model that answered in eight characters is a curve about
    nothing, and reply length is not something the caller controls.
    """
    started = time.perf_counter()
    try:
        if stage == "asr":
            response = await client.post("/v1/audio/transcriptions", content=payload)
        else:
            response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        # Read the body before stopping the clock: a response whose bytes are
        # still on the wire has not been served.
        body = response.read()
    except Exception as error:  # noqa: BLE001 - a refused request is data
        return (time.perf_counter() - started) * 1000, f"{type(error).__name__}: {error}", 0
    elapsed = (time.perf_counter() - started) * 1000
    produced = 0
    try:
        parsed = json.loads(body)
        produced = len(
            parsed["choices"][0]["message"]["content"] if stage == "llm" else parsed["text"]
        )
    except Exception:  # noqa: BLE001 - the size is reported, not judged
        produced = 0
    return elapsed, "", produced


async def sweep_level(
    base: str, stage: str, payload: Any, level: int, requests: int, timeout: float
) -> dict[str, Any]:
    """One point on the curve: `requests` turns with `level` of them in flight."""
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(requests):
        queue.put_nowait(index)
    latencies: list[float] = []
    failures: list[str] = []
    produced: list[int] = []

    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        # Warm the path once outside the clock. The first request after a
        # restart pays for lazily built state, and charging it to concurrency 1
        # would tilt the whole curve.
        await one_request(client, stage, payload)

        async def worker() -> None:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                elapsed, failure, size = await one_request(client, stage, payload)
                latencies.append(elapsed)
                produced.append(size)
                if failure:
                    failures.append(failure)

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(level)))
        wall = time.perf_counter() - started

    served = len(latencies) - len(failures)
    return {
        "concurrency": level,
        "requests": len(latencies),
        "failed": len(failures),
        "failures": sorted(set(failures))[:3],
        "seconds": wall,
        # Completed work over wall clock: the number a queue depth is chosen
        # against. Latency alone cannot say whether the card is saturated.
        "throughput_per_second": served / wall if wall > 0 else None,
        "p50_ms": nearest_rank(latencies, 0.50),
        "p95_ms": nearest_rank(latencies, 0.95),
        "mean_ms": statistics.mean(latencies) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
        # Characters produced per request. On the generation stage this is what
        # the latency was spent on; a curve without it cannot be compared to
        # another run.
        "median_characters": statistics.median(produced) if produced else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8002")
    parser.add_argument("--stage", choices=("asr", "llm"), required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--requests",
        type=int,
        default=32,
        help="turns per level. The same count at every level, so throughput is "
        "comparable and a slow level simply takes longer",
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=3.0,
        help="asr only: how long the buffer is. Recognition cost scales with it",
    )
    parser.add_argument(
        "--audio-file",
        type=Path,
        help="asr only: a real recording to send instead of a tone. A tone "
        "decodes to nothing, so the tone curve is the encoder's cost without "
        "the decoder's -- it understates a real turn",
    )
    parser.add_argument(
        "--prompt",
        default="今天天气怎么样",
        help="llm only: the same prompt every turn, so reply-length variance "
        "does not become the thing being measured",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def audio() -> bytes:
        if args.audio_file is None:
            return speech_like(args.audio_seconds)
        import numpy
        import soundfile

        waveform, rate = soundfile.read(str(args.audio_file), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if rate != 16_000:
            import librosa

            waveform = librosa.resample(waveform.astype(float), orig_sr=rate, target_sr=16_000)
        return (numpy.clip(waveform, -1.0, 1.0) * 32767).astype("int16").tobytes()

    payload: Any = (
        audio()
        if args.stage == "asr"
        else {
            "model": "mindsurf-omni",
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "stream": False,
        }
    )

    points = []
    for level in args.concurrency:
        point = asyncio.run(
            sweep_level(args.base, args.stage, payload, level, args.requests, args.timeout)
        )
        points.append(point)
        print(
            f"并发 {level:>2}: 吞吐 {point['throughput_per_second']:.2f} 次/秒  "
            f"P50 {point['p50_ms']:.0f} ms  P95 {point['p95_ms']:.0f} ms  "
            f"失败 {point['failed']}/{point['requests']}",
            flush=True,
        )
        if point["failures"]:
            print(f"    {point['failures'][0]}", flush=True)

    report = {
        "stage": args.stage,
        "base": args.base,
        "requests_per_level": args.requests,
        "audio_seconds": (
            (len(payload) / 2 / 16_000) if args.stage == "asr" else None  # type: ignore[arg-type]
        ),
        "audio_file": str(args.audio_file) if args.audio_file else None,
        "prompt": args.prompt if args.stage == "llm" else None,
        "max_tokens": args.max_tokens if args.stage == "llm" else None,
        "points": points,
        "caveat": (
            "单卡、单进程、同一份权重；合成器不在链路里。"
            "并发 1 的读数不是延迟基准（那要独占卡 + measure_latency.py）"
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"留档 {args.output}")


if __name__ == "__main__":
    main()
