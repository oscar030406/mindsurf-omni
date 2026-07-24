"""Measure time-to-first-audio against a running service, stage by stage.

A single total tells you the budget was missed; a breakdown tells you which
stage to go and fix. The largest term is usually not the model but when
synthesis is allowed to begin -- waiting for a whole reply spends the entire
generation before the listener hears anything.

Percentiles are nearest-rank, so every reported number is a latency some turn
actually experienced. An interpolated P95 is a value no user ever saw, which is
the wrong thing to hold a budget against.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.latency import LatencyReport, TurnTimings  # noqa: E402
from mindsurf_omni.service.engine import split_first_utterance  # noqa: E402


def speech_like(seconds: float, rate: int = 16_000) -> bytes:
    """Not silence: a VAD would never mark silence as the end of a turn."""
    import math

    samples = [
        int(0.3 * 32767 * math.sin(2 * math.pi * 220 * index / rate))
        for index in range(int(rate * seconds))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


async def one_turn(client: httpx.AsyncClient, prompt: str, audio: bytes) -> TurnTimings:
    """One request through the HTTP path, timed at the stages it exposes."""
    timings = TurnTimings()

    started = time.perf_counter()
    transcription = await client.post("/v1/audio/transcriptions", content=audio)
    timings.stages["encode"] = (time.perf_counter() - started) * 1000
    if transcription.status_code != 200:
        raise RuntimeError(f"transcription returned {transcription.status_code}")

    started = time.perf_counter()
    first_token_at: float | None = None
    text = ""
    clause: str | None = None
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}], "stream": True},
    ) as stream:
        async for line in stream.aiter_lines():
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
                timings.stages["first_text_token"] = (first_token_at - started) * 1000
            delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
            text += delta
            # Stop at the first speakable clause, using the same function the
            # cascade cuts with. Reading the stream to the end and subtracting
            # measures the whole generation and files it under "first_clause" --
            # which inflates time-to-first-audio by the entire tail of the
            # reply and makes the model look like the dominant stage. That is
            # precisely the wrong conclusion this instrument exists to prevent:
            # nothing after this point can affect when the listener hears
            # something, because synthesis has already started.
            clause = split_first_utterance(text)
            if clause is not None:
                timings.stages["first_clause"] = (time.perf_counter() - first_token_at) * 1000
                break

    if clause is None:
        # The reply never reached a clause boundary. Whatever there is, is what
        # gets spoken -- the same fallback the cascade takes.
        clause = text.strip()
        if first_token_at is not None:
            timings.stages["first_clause"] = (time.perf_counter() - first_token_at) * 1000

    started = time.perf_counter()
    audio_reply = await client.post("/v1/audio/speech", json={"input": clause or "好的"})
    timings.stages["synthesis"] = (time.perf_counter() - started) * 1000
    if audio_reply.status_code != 200:
        raise RuntimeError(f"speech returned {audio_reply.status_code}")
    return timings


async def one_realtime_turn(url: str, pcm: bytes, timeout: float) -> TurnTimings:
    """One turn over the WebSocket, which is the native path's only shape.

    Two stages, not six. The cascade's breakdown exists because its stages are
    separate processes and one of them is worth going to fix; the native path
    runs encode, generation and audio off a single forward pass, so the only
    boundaries that exist are the first word and the first sound. The stages it
    cannot report stay absent rather than being recorded as zero, and the
    script names them.
    """
    import base64

    import websockets

    timings = TurnTimings()
    async with websockets.connect(url, max_size=None, open_timeout=timeout) as socket:
        await socket.recv()  # session.created
        await socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        started = time.perf_counter()
        await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))

        spoke_at = None
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=timeout))
            kind = event.get("type")
            if kind == "response.text.delta" and "first_text_token" not in timings.stages:
                spoke_at = time.perf_counter()
                timings.stages["first_text_token"] = (spoke_at - started) * 1000
            elif kind == "response.audio.delta":
                # From the first word to the first sound. On this path that is
                # the Talker filling a chunk, not a separate synthesiser.
                reference = spoke_at or started
                timings.stages["synthesis"] = (time.perf_counter() - reference) * 1000
                return timings
            elif kind == "error":
                raise RuntimeError(str(event.get("error", {}).get("message"))[:200])
            elif kind == "response.done":
                raise RuntimeError("the turn finished without producing audio")


async def measure_realtime(
    base: str, turns: int, timeout: float, stimulus: Path, probes: list[str]
) -> tuple[LatencyReport, list[str]]:
    import soundfile

    from mindsurf_omni.contract import INPUT_SAMPLE_RATE
    from mindsurf_omni.service.audio import resample

    parsed = urlparse(base)
    url = f"{'wss' if parsed.scheme == 'https' else 'ws'}://{parsed.netloc}/v1/realtime"

    report = LatencyReport()
    errors: list[str] = []
    for index in range(turns):
        audio_file = stimulus / f"{probes[index % len(probes)]}.wav"
        if not audio_file.is_file():
            errors.append(f"turn {index}: no stimulus at {audio_file.name}")
            continue
        heard, rate = soundfile.read(audio_file, dtype="int16")
        try:
            report.add(
                await one_realtime_turn(
                    url, resample(heard.tobytes(), rate, INPUT_SAMPLE_RATE), timeout
                )
            )
        except Exception as error:  # noqa: BLE001 - a failed turn is data
            errors.append(f"turn {index}: {type(error).__name__}: {error}")
    return report, errors


async def measure(base: str, turns: int, timeout: float) -> tuple[LatencyReport, list[str]]:
    report = LatencyReport()
    errors: list[str] = []
    audio = speech_like(1.5)

    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        for index in range(turns):
            try:
                report.add(await one_turn(client, f"第{index}个问题", audio))
            except Exception as error:  # noqa: BLE001 - a failed turn is data
                errors.append(f"turn {index}: {type(error).__name__}: {error}")
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--budget-ms", type=float, default=3000.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--via",
        choices=("http", "realtime"),
        default="http",
        help="'realtime' is the only way to time the native path: it makes words "
        "and sound in one pass and will not read text it did not write",
    )
    parser.add_argument("--stimulus", type=Path, help="spoken probes, for --via realtime")
    parser.add_argument("--probes", type=Path, default=Path("configs/speech_probes_zh_v1.jsonl"))
    args = parser.parse_args()

    print(f"测 {args.turns} 轮，对 {args.base}")
    if args.via == "realtime":
        if args.stimulus is None:
            raise SystemExit("--via realtime needs --stimulus pointing at spoken probes")
        ids = [
            json.loads(line)["id"]
            for line in args.probes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report, errors = asyncio.run(
            measure_realtime(args.base, args.turns, args.timeout, args.stimulus, ids)
        )
    else:
        report, errors = asyncio.run(measure(args.base, args.turns, args.timeout))

    if not report.turns:
        print("没有一轮成功：")
        for error in errors[:5]:
            print(f"  {error}")
        raise SystemExit(1)

    print(f"\n{report.budget_verdict(args.budget_ms)}")
    print(f"P50 {report.percentile(0.5):.0f} ms   P95 {report.percentile(0.95):.0f} ms")

    print("\n各阶段中位数:")
    for name, value in report.stage_medians().items():
        print(f"  {name:18} {value:7.0f} ms")

    # A stage nobody measured reads as a stage that costs nothing, and the
    # total is a sum -- so an omission makes the budget look met.
    absent = report.turns[0].missing_stages()
    if absent:
        print(f"\n未计入的阶段: {', '.join(absent)}——总和因此偏小，不是真的 TTF-Audio")

    dominant = report.dominant_stage()
    if dominant:
        print(f"\n最大的一项是 {dominant[0]}（{dominant[1]:.0f} ms）——先修这里")

    measurement = report.measurement()
    print(f"\n{measurement}")
    if not measurement.gating_eligible:
        print(f"  仅报告：{measurement.note}")

    # Failed turns are named rather than dropped: a run that only counts the
    # turns that worked reports the latency of a service that half works.
    if errors:
        print(f"\n{len(errors)} 轮失败，未计入统计:")
        for error in errors[:5]:
            print(f"  {error}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "turns": len(report.turns),
                    "failed": errors,
                    "p50_ms": report.percentile(0.5),
                    "p95_ms": report.percentile(0.95),
                    "stage_medians_ms": report.stage_medians(),
                    "gating_eligible": measurement.gating_eligible,
                    "note": measurement.note,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n输出 {args.output}")


if __name__ == "__main__":
    main()
