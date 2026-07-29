"""Latency measurement, including what it does with turns that fail."""

from __future__ import annotations

import asyncio
import json

import httpx
from scripts.measure_latency import measure, one_turn, speech_like


def test_the_probe_audio_is_not_silence() -> None:
    """A VAD would never mark silence as the end of a turn."""
    audio = speech_like(0.1, rate=16_000)

    assert len(audio) == 3200  # 1600 samples, 2 bytes each
    assert any(byte != 0 for byte in audio)


class _Service:
    def __init__(
        self,
        *,
        fail_every: int | None = None,
        delay: float = 0.0,
        deltas: list[str] | None = None,
    ) -> None:
        self.fail_every = fail_every
        self.delay = delay
        self.turn = 0
        self.deltas = deltas
        self.spoken: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/audio/transcriptions":
            self.turn += 1
            if self.fail_every and self.turn % self.fail_every == 0:
                return httpx.Response(500)
            return httpx.Response(200, json={"text": "听到的", "language": "zh"})
        if path == "/v1/chat/completions":
            pieces = self.deltas if self.deltas is not None else ["好", "的。"]
            body = "".join(
                "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
                for piece in pieces
            )
            body += "data: [DONE]\n\n"
            return httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            )
        if path == "/v1/audio/speech":
            self.spoken.append(json.loads(request.content)["input"])
            return httpx.Response(200, content=b"\x00\x01" * 100)
        return httpx.Response(404)


def _run(service: _Service, turns: int) -> tuple:
    transport = httpx.MockTransport(service.handler)
    original = httpx.AsyncClient

    class Patched(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: object) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)  # type: ignore[arg-type]

    httpx.AsyncClient = Patched  # type: ignore[misc]
    try:
        return asyncio.run(measure("http://test", turns, timeout=5.0))
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]


def test_every_stage_is_timed() -> None:
    """A missing stage reads as a fast stage."""
    report, errors = _run(_Service(), turns=3)

    assert errors == []
    assert len(report.turns) == 3
    for name in ("encode", "first_text_token", "first_clause", "synthesis"):
        assert name in report.turns[0].stages


def test_failed_turns_are_named_not_counted_as_fast() -> None:
    """Counting only the turns that worked reports a service that half works."""
    report, errors = _run(_Service(fail_every=2), turns=4)

    assert len(report.turns) == 2
    assert len(errors) == 2
    assert "transcription returned 500" in errors[0]


def test_a_service_that_never_answers_exits_rather_than_reporting_zero() -> None:
    report, errors = _run(_Service(fail_every=1), turns=3)

    assert report.turns == []
    assert len(errors) == 3


def test_too_few_turns_cannot_gate() -> None:
    """Three timed turns cannot resolve a 200 ms difference."""
    report, _ = _run(_Service(), turns=3)

    assert not report.measurement().gating_eligible


def test_the_dominant_stage_is_identified() -> None:
    report, _ = _run(_Service(), turns=5)

    dominant = report.dominant_stage()

    assert dominant is not None
    assert dominant[0] in {"encode", "first_text_token", "first_clause", "synthesis"}


def test_percentiles_are_values_some_turn_actually_had() -> None:
    report, _ = _run(_Service(), turns=5)
    totals = {turn.time_to_first_audio_ms for turn in report.turns}

    assert report.percentile(0.95) in totals
    assert report.percentile(0.5) in totals


def test_first_clause_stops_at_a_clause_rather_than_at_the_end_of_the_reply() -> None:
    """Reading the stream to the end and subtracting measures the whole generation.

    Filed under "first_clause" it inflates time-to-first-audio by the entire
    tail of the reply and makes the model look like the dominant stage -- the
    exact conclusion this instrument exists to prevent. Nothing after the first
    speakable clause can change when the listener hears something, because
    synthesis has already started by then.
    """
    tail = ["还有很多很多后面的内容。"] * 40
    service = _Service(deltas=["今天", "天气很好。", *tail])

    _, errors = _run(service, turns=1)

    assert not errors
    assert service.spoken == ["今天天气很好。"]  # the clause, not the clause plus the tail


def test_a_reply_with_no_clause_boundary_is_still_spoken() -> None:
    """Dropping it would report a turn that never made a sound as a fast turn."""
    service = _Service(deltas=["短句没有句号"])

    report, errors = _run(service, turns=1)

    assert not errors
    assert service.spoken == ["短句没有句号"]
    assert "first_clause" in report.turns[0].stages


def test_the_native_turn_reports_only_the_stages_it_can_see() -> None:
    """Two stages, not six, and the missing four are named rather than zeroed.

    The cascade's breakdown exists because its stages are separate processes
    and one of them is worth going to fix. The native path runs encode,
    generation and audio off a single forward pass, so those boundaries do not
    exist in it. Recording them as zero would make the total look like a
    complete time-to-first-audio when it is not.
    """
    from mindsurf_omni.evaluation.latency import TurnTimings

    timings = TurnTimings()
    timings.stages["first_text_token"] = 72.0
    timings.stages["synthesis"] = 224.0

    assert timings.time_to_first_audio_ms == 296.0
    assert timings.missing_stages() == ["vad_endpoint", "encode", "first_clause", "transport"]


def test_skipping_synthesis_stops_before_the_speech_call() -> None:
    """The speech stage belongs to whichever synthesiser an operator picked.

    Without this the two stages this project wrote cannot be measured at all
    where no synthesiser is configured: every turn fails on the speech call and
    the report says nothing, rather than saying what the part that ran cost.
    """
    service = _Service()
    transport = httpx.MockTransport(service.handler)

    async def run() -> object:
        async with httpx.AsyncClient(transport=transport, base_url="http://svc") as client:
            return await one_turn(client, "问", speech_like(0.05), skip_synthesis=True)

    timings = asyncio.run(run())

    assert "first_clause" in timings.stages
    assert "synthesis" not in timings.stages
    # The speech endpoint records what it was asked to say; nothing reached it.
    assert service.spoken == []
