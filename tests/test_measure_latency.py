"""Latency measurement, including what it does with turns that fail."""

from __future__ import annotations

import asyncio

import httpx

from scripts.measure_latency import measure, speech_like


def test_the_probe_audio_is_not_silence() -> None:
    """A VAD would never mark silence as the end of a turn."""
    audio = speech_like(0.1, rate=16_000)

    assert len(audio) == 3200  # 1600 samples, 2 bytes each
    assert any(byte != 0 for byte in audio)


class _Service:
    def __init__(self, *, fail_every: int | None = None, delay: float = 0.0) -> None:
        self.fail_every = fail_every
        self.delay = delay
        self.turn = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/audio/transcriptions":
            self.turn += 1
            if self.fail_every and self.turn % self.fail_every == 0:
                return httpx.Response(500)
            return httpx.Response(200, json={"text": "听到的", "language": "zh"})
        if path == "/v1/chat/completions":
            body = (
                'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"的。"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            )
        if path == "/v1/audio/speech":
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
