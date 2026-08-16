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
from mindsurf_omni.service.engine import GenerationSettings, TooLongForModel

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

    # Text arrives as it is decided; the audio that speaks it follows.
    assert "".join(chunk.text for chunk in chunks if chunk.text) == "今天天气真好。我们出去走走吧。"
    assert len([chunk for chunk in chunks if chunk.pcm]) == 2
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_first_audio_arrives_long_before_the_reply_finishes() -> None:
    """A 40-character reply must not cost 40 characters of latency."""
    engine = _engine("今天天气真好。" + "很" * 60 + "。", token_delay=0.002)

    first_audio = None
    async for chunk in engine.respond(b"", 16_000, GenerationSettings()):
        if chunk.pcm:
            first_audio = chunk
            break

    assert first_audio is not None
    # Roughly the seven characters of the first clause, not the sixty-eight of
    # the whole reply.
    assert engine.last_timings.time_to_first_audio_ms < 60


@pytest.mark.asyncio
async def test_a_reply_with_no_sentence_end_is_still_spoken() -> None:
    """Dropping the tail because it lacks punctuation would lose the answer."""
    engine = _engine("好的")

    chunks = [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings())]

    assert "".join(chunk.text for chunk in chunks if chunk.text) == "好的"
    assert [chunk.pcm for chunk in chunks if chunk.pcm] == ["好的".encode()]
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

    assert [chunk for chunk in chunks if chunk.pcm] == []
    # The turn still happened, and the session needs to know what was said.
    assert [chunk.transcript for chunk in chunks if chunk.transcript is not None] == ["你好"]


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
    # Audio carries no text: it already went out with the deltas, and a client
    # that rendered both would show the sentence twice.
    assert [chunk.text for chunk in audio] == [None, None, None]
    assert "".join(chunk.text for chunk in chunks if chunk.text) == "今天天气很好。"
    assert engine.last_timings.time_to_first_audio_ms > 0


@pytest.mark.asyncio
async def test_without_a_streaming_synthesiser_nothing_changes() -> None:
    """A hosted synthesiser cannot divide its round trip, so it keeps the old path."""
    engine = _engine("今天天气很好。")

    chunks = [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings())]
    spoken = [chunk for chunk in chunks if chunk.pcm]

    assert len(spoken) == 1
    assert spoken[0].text is None
    assert "".join(chunk.text for chunk in chunks if chunk.text) == "今天天气很好。"


@pytest.mark.asyncio
async def test_history_reaches_the_prompt_and_this_turn_is_appended_to_it() -> None:
    """Without this the session's bookkeeping described a history nobody read.

    The service counted turns, trimmed them and reported dropped_turns, while
    every prompt was built from the current utterance alone -- so the model
    answered each turn as the first one and the context budget measured
    nothing.
    """
    seen: list[list[dict[str, str]]] = []

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "那它多少钱", "zh"

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        seen.append(list(messages))
        for character in "两百块。":
            yield character

    async def synthesise(text: str, settings: object) -> bytes:
        return text.encode("utf-8")

    engine = CascadeEngine(transcribe, generate, synthesise, [], SPEC)  # type: ignore[arg-type]
    history = [
        {"role": "user", "content": "这个杯子好看吗"},
        {"role": "assistant", "content": "挺好看的。"},
    ]

    chunks = [
        chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings(), history=history)
    ]

    assert seen == [[*history, {"role": "user", "content": "那它多少钱"}]]
    # The transcript comes back out, because the caller holding the history has
    # no other way to learn what the user said on this path.
    assert [chunk.transcript for chunk in chunks if chunk.transcript is not None] == ["那它多少钱"]


@pytest.mark.asyncio
async def test_a_streaming_synthesiser_is_used_by_speak_too() -> None:
    """speak() served the HTTP path and the backend's per-clause loop.

    It waited for the whole clause while respond() streamed, so the same
    synthesiser was fast on one entry point and slow on the other.
    """
    pieces = [b"one", b"two"]

    async def stream(text: str, settings: object) -> AsyncIterator[bytes]:
        for piece in pieces:
            yield piece

    async def synthesise(text: str, settings: object) -> bytes:
        raise AssertionError("the streaming synthesiser was available and unused")

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "你好", "zh"

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        yield "好"

    engine = CascadeEngine(
        transcribe,  # type: ignore[arg-type]
        generate,  # type: ignore[arg-type]
        synthesise,  # type: ignore[arg-type]
        [],
        SPEC,
        stream_synthesiser=stream,  # type: ignore[arg-type]
    )

    chunks = [chunk async for chunk in engine.speak("今天天气很好。", GenerationSettings())]

    assert [chunk.pcm for chunk in chunks if chunk.pcm] == pieces
    assert chunks[0].text == "今天天气很好。"
    assert chunks[-1].is_final


@pytest.mark.asyncio
async def test_text_reaches_the_caller_before_any_audio_does() -> None:
    """The reason the text is no longer bound to its audio.

    Synthesis is a network round trip: measured live, the first word appeared
    at 1748 ms and the first audio at 1752 ms, so a reader watched a blank
    screen for the whole trip while the words had been decided since 100 ms.
    """
    synthesis_started = asyncio.Event()

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "你好", "zh"

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        for character in "今天天气很好。":
            yield character

    async def synthesise(text: str, settings: object) -> bytes:
        synthesis_started.set()
        await asyncio.sleep(0.05)  # the round trip
        return text.encode("utf-8")

    engine = CascadeEngine(transcribe, generate, synthesise, [], SPEC)  # type: ignore[arg-type]

    order: list[str] = []
    async for chunk in engine.respond(b"", 16_000, GenerationSettings()):
        if chunk.text:
            order.append("text")
        if chunk.pcm:
            order.append("audio")

    assert order, "nothing came back"
    assert order[0] == "text"
    assert order.index("audio") > 0, "audio arrived before any text"


@pytest.mark.asyncio
async def test_silence_is_not_handed_to_the_model_as_an_empty_question() -> None:
    """Measured on the running service: two seconds of silence came back as
    "Sure, what's your question?", in English, synthesised and played. The
    recogniser already returns "" for silence; what was missing is that the
    empty transcript still went on to the generator as a user message."""
    asked: list[list[dict[str, str]]] = []

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "", None

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        asked.append(list(messages))
        yield "我在。"

    async def synthesise(text: str, settings: object) -> bytes:
        return text.encode("utf-8")

    engine = CascadeEngine(transcribe, generate, synthesise, [], SPEC)  # type: ignore[arg-type]

    chunks = [chunk async for chunk in engine.respond(b"\x00" * 64, 16_000, GenerationSettings())]

    assert asked == []
    assert not any(chunk.text for chunk in chunks)
    assert not any(chunk.pcm for chunk in chunks)


@pytest.mark.asyncio
async def test_a_transcript_of_only_whitespace_counts_as_silence() -> None:
    asked: list[list[dict[str, str]]] = []

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "   ", None

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        asked.append(list(messages))
        yield "我在。"

    async def synthesise(text: str, settings: object) -> bytes:
        return text.encode("utf-8")

    engine = CascadeEngine(transcribe, generate, synthesise, [], SPEC)  # type: ignore[arg-type]

    [chunk async for chunk in engine.respond(b"\x00" * 64, 16_000, GenerationSettings())]

    assert asked == []


def test_a_short_spoken_turn_is_one_message_exactly_as_before() -> None:
    from mindsurf_omni.service.cascade import spoken_turn

    assert spoken_turn("杭州有什么好玩的") == [{"role": "user", "content": "杭州有什么好玩的"}]


def test_a_monologue_is_split_into_the_shape_the_weights_have_seen() -> None:
    """Measured: one message of 968 tokens came back empty 8/8, the same words
    as consecutive user turns 0/8. Live, 124 seconds of speech recognised to
    553 characters and was refused outright -- the speaker cannot split what
    they have already said."""
    from mindsurf_omni.service.cascade import SPOKEN_TURN_CHARACTERS, spoken_turn

    transcript = "杭州有什么好玩的地方呢？西湖我已经去过了。那边吃饭大概多少钱一个人？" * 12

    messages = spoken_turn(transcript)

    assert len(messages) > 1
    assert all(message["role"] == "user" for message in messages)
    # Every piece inside the length, and nothing of what was said is lost.
    assert all(len(message["content"]) <= SPOKEN_TURN_CHARACTERS for message in messages[:-1])
    assert "".join(message["content"] for message in messages) == transcript


def test_a_monologue_past_what_any_shape_answers_is_refused_not_truncated() -> None:
    """At 1502 tokens every shape tried came back empty 8/8. Answering from a
    piece of it would mean dropping the oldest half of what somebody just said,
    silently."""
    from mindsurf_omni.service.cascade import ANSWERABLE_CHARACTERS, spoken_turn

    with pytest.raises(TooLongForModel, match="Commit at the end"):
        spoken_turn("这是一句很长的话。" * (ANSWERABLE_CHARACTERS // 8))


@pytest.mark.asyncio
async def test_the_split_turn_reaches_the_model_with_the_history_before_it() -> None:
    seen: list[list[dict[str, str]]] = []
    long_transcript = "杭州有什么好玩的地方呢？西湖我已经去过了。那边吃饭多少钱一个人？" * 10

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return long_transcript, "zh"

    async def generate(messages: list[dict[str, str]], settings: object) -> AsyncIterator[str]:
        seen.append(list(messages))
        yield "好的。"

    async def synthesise(text: str, settings: object) -> bytes:
        return text.encode("utf-8")

    engine = CascadeEngine(transcribe, generate, synthesise, [], SPEC)  # type: ignore[arg-type]
    history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好。"}]

    [chunk async for chunk in engine.respond(b"", 16_000, GenerationSettings(), history=history)]

    assert seen[0][:2] == history
    assert len(seen[0]) > 3
    assert "".join(message["content"] for message in seen[0][2:]) == long_transcript


def test_history_is_split_the_same_way_the_current_turn_is() -> None:
    """A monologue answered on one turn is recorded whole, and on the next turn
    that single over-long history entry was refused -- the turn before had just
    worked. Splitting only what is about to be said fixes one turn and breaks
    the one after it."""
    from mindsurf_omni.service.cascade import SPOKEN_TURN_CHARACTERS, answerable

    monologue = "杭州有什么好玩的地方呢？西湖我已经去过了。那边吃饭多少钱一个人？" * 10
    history = [
        {"role": "user", "content": monologue},
        {"role": "assistant", "content": "好的。"},
    ]

    messages = answerable(history, "那高铁要多久")

    assert all(len(message["content"]) <= SPOKEN_TURN_CHARACTERS for message in messages)
    assert messages[-1] == {"role": "user", "content": "那高铁要多久"}
    # The assistant's turn stays where it was, after all the user's pieces.
    assert messages[-2] == {"role": "assistant", "content": "好的。"}
    said = "".join(m["content"] for m in messages if m["role"] == "user" and m is not messages[-1])
    assert said == monologue


def test_history_that_is_very_long_is_split_not_refused() -> None:
    """Refusing is reserved for what the speaker just said, which cannot be
    recovered any other way; old turns can be dropped, and already are."""
    from mindsurf_omni.service.cascade import ANSWERABLE_CHARACTERS, answerable

    history = [{"role": "user", "content": "这是一句很长的话。" * (ANSWERABLE_CHARACTERS // 8)}]

    messages = answerable(history, "嗯")  # must not raise

    assert len(messages) > 2
