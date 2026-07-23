"""The contract's promises, pinned.

These run before the model exists. That is the point: the backend and the
client are built against this contract in parallel with the model, so a shape
that changes later costs two other teams a day each.
"""

from __future__ import annotations

import pytest

from mindsurf_omni.contract import (
    AUDIO_ENCODING,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    ChatCompletionRequest,
    SpeechRequest,
)
from mindsurf_omni.service.engine import split_first_utterance


def test_audio_formats_are_fixed_at_the_rates_the_models_require() -> None:
    """SenseVoice takes 16 kHz and Mimi emits 24 kHz; neither is negotiable."""
    assert INPUT_SAMPLE_RATE == 16_000
    assert OUTPUT_SAMPLE_RATE == 24_000
    assert AUDIO_ENCODING == "pcm_s16le"


def test_sampling_parameters_are_bounded() -> None:
    """A caller that sends top_p=0 should get a 422, not silent nonsense."""
    with pytest.raises(ValueError):
        ChatCompletionRequest(messages=[], top_p=0.0)
    with pytest.raises(ValueError):
        ChatCompletionRequest(messages=[], temperature=-1.0)
    with pytest.raises(ValueError):
        SpeechRequest(input="hello", speed=10.0)


def test_emotion_rides_beside_the_text_not_inside_it() -> None:
    """Smuggling delivery into the spoken text is how a model reads it aloud.

    A sister project hit exactly this: emotion instructions prepended to the
    text were occasionally spoken by the synthesiser.
    """
    request = SpeechRequest(input="今天天气真好", emotion="happy")

    assert request.input == "今天天气真好"
    assert request.emotion == "happy"


@pytest.mark.parametrize(
    "accumulated,expected",
    [
        ("今天天气真好。剩下的还没说完", "今天天气真好。"),
        ("好的！", "好的！"),
        ("太短了", None),
        # Long run with no sentence end yet: cut at the last clause boundary
        # rather than keep the listener waiting for a full stop that may be
        # far away. Cuts at the last comma, not the first, so the spoken chunk
        # is as long as it can be without stalling.
        (
            "灵山大佛通高八十八米，用铜七百二十五吨，立在小灵山上",
            "灵山大佛通高八十八米，用铜七百二十五吨，",
        ),
        # Long enough, but every clause boundary is too early to be worth
        # speaking on its own.
        ("好，这是一个很长很长很长很长很长很长很长的句子", None),
        # Just under the threshold: wait for more rather than speak a fragment.
        ("灵山大佛通高八十八米，用铜七百吨", None),
    ],
)
def test_first_utterance_split(accumulated: str, expected: str | None) -> None:
    assert split_first_utterance(accumulated) == expected


def test_split_never_returns_an_empty_chunk() -> None:
    """An empty chunk would start a synthesis request that produces silence."""
    for text in ("", "。", "，", "a" * 100):
        result = split_first_utterance(text)
        assert result is None or len(result) > 0
