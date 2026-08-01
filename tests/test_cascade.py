"""The cascade's latency behaviour, tested without a model.

Stubs stand in for the three components, with controlled delays. That is the
point: what governs time-to-first-audio here is *when synthesis starts*, not
how fast any model is, and stubs make that visible in a way real models would
obscure.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from mindsurf_omni.contract import TokenSpec
from mindsurf_omni.service.cascade import CascadeEngine
from mindsurf_omni.service.engine import GenerationSettings

SPEC = TokenSpec(
    text_vocab_size=6400,
    audio_codebooks=8,
    audio_codebook_size=2048,
    audio_frame_rate_hz=12.5,
    special_tokens={"im_start": 1, "im_end": 2},
    audio_special_tokens={"pad": 2049, "stop": 2050},
)


def _engine(reply: str, token_delay: float = 0.0, synth_delay: float = 0.0) -> CascadeEngine:
    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "你好", "zh"

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        for character in reply:
            if token_delay:
                await asyncio.sleep(token_delay)
            yield character

    async def synthesise(text: str, settings: object) -> bytes:
        if synth_delay:
            await asyncio.sleep(synth_delay)
        return text.encode("utf-8")  # stand-in for PCM

    return CascadeEngine(transcribe, generate, synthesise, [], SPEC)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_speech_starts_at_the_first_clause_not_the_end() -> None:
    """The whole latency argument rests on this."""
    engine = _engine("今天天气真好。我们出去走走吧。")

    chunks = [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings())]

    assert [chunk.text for chunk in chunks if chunk.text] == [
        "今天天气真好。",
        "我们出去走走吧。",
    ]
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_first_audio_arrives_long_before_the_reply_finishes() -> None:
    """A 40-character reply must not cost 40 characters of latency."""
    engine = _engine("今天天气真好。" + "很" * 60 + "。", token_delay=0.002)

    first = None
    async for chunk in engine.respond(b"", 16_000, GenerationSettings()):
        first = chunk
        break

    assert first is not None and first.text == "今天天气真好。"
    # Roughly the seven characters of the first clause, not the sixty-eight of
    # the whole reply.
    assert engine.last_timings.time_to_first_audio_ms < 60


@pytest.mark.asyncio
async def test_a_reply_with_no_sentence_end_is_still_spoken() -> None:
    """Dropping the tail because it lacks punctuation would lose the answer."""
    engine = _engine("好的")

    chunks = [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings())]

    assert [chunk.text for chunk in chunks] == ["好的"]
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_timings_attribute_each_stage() -> None:
    """A single total cannot tell you which stage to fix."""
    engine = _engine("你好呀。", synth_delay=0.01)

    async for _ in engine.respond(b"", 16_000, GenerationSettings()):
        pass

    timings = engine.last_timings
    assert timings.first_clause_ms > 0
    # The injected 10 ms lands on the synthesis stage and nowhere else, and
    # that attribution is the claim. It is asserted against the other stages
    # rather than against 10 ms: the sleep is scheduled on the loop's monotonic
    # clock, whose granularity on Windows is ~15.6 ms, while the bracket around
    # it reads perf_counter, so the loop can wake "early" and the measurement
    # reads 9.7 -- or 7.97 on a busy machine, which is what an absolute bound
    # caught here. Under load every stage inflates together, so the comparison
    # survives what the constant does not, and a test that fails for the
    # platform's clock is measuring the clock.
    assert timings.first_synthesis_ms > timings.first_clause_ms
    assert timings.first_synthesis_ms > 1
    assert timings.time_to_first_audio_ms >= timings.first_clause_ms


@pytest.mark.asyncio
async def test_an_empty_reply_produces_no_audio() -> None:
    """Synthesising silence wastes a request and plays a click."""
    engine = _engine("")

    chunks = [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings())]

    assert chunks == []


@pytest.mark.asyncio
async def test_a_streaming_synthesiser_speaks_before_the_clause_is_finished() -> None:
    """This is the whole point of the change, so it is what the test pins.

    Measured on the deployment card, a clause takes 2133.9 ms to synthesise
    while recognition and the Thinker together take 126.0 ms. Waiting for the
    clause is 94% of the budget and the reason the cascade's P95 sits near
    4695 ms against a 3000 ms target. Time to first audio has to be stamped on
    the first piece, not the last.
    """
    pieces = [bytes([1, 0]) * 4, bytes([2, 0]) * 4, bytes([3, 0]) * 4]

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "你好", "zh"

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        for character in "今天天气很好。":
            yield character

    async def whole(text: str, settings: object) -> bytes:
        raise AssertionError("the streaming path was available and was not used")

    async def stream(text: str, settings: object) -> AsyncIterator[bytes]:
        for piece in pieces:
            yield piece

    engine = CascadeEngine(
        transcribe,
        generate,
        whole,
        [],
        SPEC,
        stream_synthesiser=stream,
    )

    chunks = [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings())]
    audio = [chunk for chunk in chunks if chunk.pcm]

    assert [chunk.pcm for chunk in audio] == pieces
    # The clause's text rides on the first piece only: repeated, a client
    # renders the sentence three times; dropped, it loses the alignment.
    assert audio[0].text == "今天天气很好。"
    assert [chunk.text for chunk in audio[1:]] == [None, None]
    assert engine.last_timings.time_to_first_audio_ms > 0


@pytest.mark.asyncio
async def test_without_a_streaming_synthesiser_nothing_changes() -> None:
    """A hosted synthesiser cannot divide its round trip, so it keeps the old path."""
    engine = _engine("今天天气很好。")

    chunks = [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings())]
    spoken = [chunk for chunk in chunks if chunk.pcm]

    assert len(spoken) == 1
    assert spoken[0].text == "今天天气很好。"
