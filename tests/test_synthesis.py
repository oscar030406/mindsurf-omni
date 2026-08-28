"""Synthesis input cleaning, and the screen for instructions read aloud."""

from __future__ import annotations

import pytest

from mindsurf_omni.data.synthesis import (
    EDGE_PROSODY,
    EMOTION_INSTRUCTIONS,
    EdgeSynthesiser,
    Utterance,
    clean_for_speech,
)


def test_markdown_markers_are_removed() -> None:
    """They are shown, not spoken, and a synthesiser reads them as something."""
    assert clean_for_speech("**灵山大佛**通高 `88` 米") == "灵山大佛通高 88 米"


def test_em_dashes_become_pauses_rather_than_being_read() -> None:
    assert clean_for_speech("很好——真的很好") == "很好，真的很好"


def test_a_short_aside_is_kept_as_an_aside() -> None:
    """A speaker would say it, so it stays -- between pauses."""
    assert clean_for_speech("门票（旺季）二百一十元") == "门票，旺季，二百一十元"


def test_a_long_parenthetical_is_dropped() -> None:
    """A speaker would not read a paragraph in brackets, and it buries the sentence."""
    text = "灵山大佛（这里有一段很长很长很长很长很长很长的补充说明文字）值得一看"

    assert clean_for_speech(text) == "灵山大佛值得一看"


def test_line_breaks_become_sentence_ends() -> None:
    assert clean_for_speech("第一句\n第二句") == "第一句。第二句"


def test_repeated_punctuation_is_collapsed() -> None:
    """Doubled marks come from cleaning itself and read as a stumble."""
    assert clean_for_speech("好的。。，，然后呢") == "好的。然后呢"


def test_cleaning_leaves_ordinary_text_alone() -> None:
    assert clean_for_speech("今天天气真好") == "今天天气真好"


def test_cleaning_an_empty_string_is_empty_not_punctuation() -> None:
    """A stray comma would be synthesised as a pause with no content."""
    assert clean_for_speech("") == ""
    assert clean_for_speech("\n\n") == ""


def test_emotion_lives_beside_the_text_not_inside_it() -> None:
    utterance = Utterance(text="今天天气真好", emotion="happy")

    assert utterance.text == "今天天气真好"
    assert "开心" in utterance.instruction()


def test_an_unknown_emotion_falls_back_rather_than_failing() -> None:
    """A typo in a client should not take the turn down."""
    assert Utterance(text="x", emotion="ecstatic").instruction() == EMOTION_INSTRUCTIONS["neutral"]


class _FakeEndpoint:
    """Records what was asked for and hands back real, decodable audio."""

    requests: list[tuple[str, str, dict[str, str]]] = []
    seconds: float = 0.5
    fail_first_n: int = 0

    def __init__(self, text: str, voice: str, **options: str) -> None:
        type(self).requests.append((text, voice, options))
        self._text = text

    async def stream(self):  # type: ignore[no-untyped-def]
        import io
        import math

        import numpy
        import soundfile

        cls = type(self)
        if cls.fail_first_n:
            cls.fail_first_n -= 1
            raise RuntimeError("NoAudioReceived")
        if not cls.seconds:
            return
        rate = 24_000
        count = int(rate * type(self).seconds)
        tone = numpy.array(
            [int(0.3 * 32767 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(count)],
            dtype=numpy.int16,
        )
        container = io.BytesIO()
        soundfile.write(container, tone, rate, format="WAV", subtype="PCM_16")
        yield {"type": "audio", "data": container.getvalue()}


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> type[_FakeEndpoint]:
    import sys
    import types

    _FakeEndpoint.requests = []
    _FakeEndpoint.seconds = 0.5
    _FakeEndpoint.fail_first_n = 0
    module = types.ModuleType("edge_tts")
    module.Communicate = _FakeEndpoint  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return _FakeEndpoint


async def test_emotion_is_carried_as_prosody_instead(endpoint: type[_FakeEndpoint]) -> None:
    await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好", emotion="happy"))

    _, _, options = endpoint.requests[0]
    assert options == EDGE_PROSODY["happy"]


async def test_an_unknown_emotion_falls_back_to_neutral_prosody(
    endpoint: type[_FakeEndpoint],
) -> None:
    await EdgeSynthesiser().synthesise(Utterance(text="x", emotion="ecstatic"))

    assert endpoint.requests[0][2] == EDGE_PROSODY["neutral"]


async def test_the_text_is_cleaned_before_it_is_spoken(endpoint: type[_FakeEndpoint]) -> None:
    """Otherwise every caller cleans it, and one forgets."""
    await EdgeSynthesiser().synthesise(Utterance(text="**灵山大佛**通高 `88` 米"))

    assert endpoint.requests[0][0] == "灵山大佛通高 88 米"


async def test_nothing_worth_saying_costs_no_request(endpoint: type[_FakeEndpoint]) -> None:
    """A request for silence spends a round trip and plays a click."""
    assert await EdgeSynthesiser().synthesise(Utterance(text="\n\n")) == b""
    assert endpoint.requests == []


async def test_returned_audio_is_pcm16_at_the_contract_rate(
    endpoint: type[_FakeEndpoint],
) -> None:
    """Anything else reaches the client as a chipmunk, reported as "sounds strange"."""
    from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE

    pcm = await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好"))

    assert len(pcm) == int(OUTPUT_SAMPLE_RATE * 0.5) * 2


async def test_an_empty_response_raises_rather_than_returning_silence(
    endpoint: type[_FakeEndpoint],
) -> None:
    """Silence scores as the model having said nothing, blaming the model for a network fault."""
    endpoint.seconds = 0.0

    with pytest.raises(RuntimeError, match="no audio"):
        await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好"))


def test_a_synthesiser_is_never_eligible_to_judge_its_own_output() -> None:
    assert EdgeSynthesiser().eligible_as_judge is False


async def test_a_single_transient_failure_is_retried(endpoint: type[_FakeEndpoint]) -> None:
    """Measured at 2 turns in 160: the endpoint sometimes returns nothing.

    It sits behind a network, so one retry is the difference between a 1.25%
    turn failure rate and a negligible one.
    """
    endpoint.fail_first_n = 1

    pcm = await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好"))

    assert pcm  # the second attempt produced audio
    assert len(endpoint.requests) == 2


def test_text_with_nothing_speakable_left_comes_back_empty() -> None:
    """ "。。。？！" survived the strip as "？！", which the hosted endpoint answers
    with no audio at all -- and this module turns that into a RuntimeError the
    speech endpoint had no choice but to serve as a 500."""
    assert clean_for_speech("。。。？！") == ""
    assert clean_for_speech("……") == ""
    assert clean_for_speech("？") == ""
    assert clean_for_speech("!!!") == ""


def test_punctuation_around_real_words_is_kept() -> None:
    """The check is "is there anything to say", not "strip punctuation"."""
    assert clean_for_speech("真的吗？太好了！") == "真的吗？太好了！"
    assert clean_for_speech("行，那就这样。") == "行，那就这样"
