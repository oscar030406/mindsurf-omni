"""Tag stripping, and the rule that keeps the judge independent."""

from __future__ import annotations

import pytest

from mindsurf_omni.service.asr import (
    SenseVoiceRecogniser,
    WhisperRecogniser,
    require_independent_judge,
    strip_tags,
)


def test_inline_tags_are_not_counted_as_speech() -> None:
    """They are metadata; scoring them charges the model for the recogniser's notes."""
    text, language = strip_tags("<|zh|><|NEUTRAL|><|Speech|><|woitn|>今天天气真好")

    assert text == "今天天气真好"
    assert language == "zh"


def test_the_reported_language_survives_stripping() -> None:
    """Chinese audio read as English is a failure worth seeing before the CER."""
    assert strip_tags("<|en|><|HAPPY|>hello there")[1] == "en"
    assert strip_tags("<|yue|>食咗飯未")[1] == "yue"


def test_text_without_tags_passes_through() -> None:
    assert strip_tags("今天天气真好") == ("今天天气真好", None)


def test_a_transcript_of_nothing_is_empty_not_a_tag_soup() -> None:
    assert strip_tags("<|nospeech|><|Event_UNK|>") == ("", "nospeech")


def test_the_service_recogniser_is_refused_as_a_judge() -> None:
    """It is the native path's own audio encoder; scoring with it is circular."""
    service = SenseVoiceRecogniser(model_dir="/nonexistent")

    with pytest.raises(ValueError, match="not an independent judge"):
        require_independent_judge(service, model_lineage="mindsurf-omni")


def test_an_independent_recogniser_is_accepted() -> None:
    require_independent_judge(WhisperRecogniser(), model_lineage="mindsurf-omni")


def test_a_judge_sharing_lineage_with_the_model_is_refused() -> None:
    """The rule is about shared failure modes, not about which vendor it is."""
    judge = WhisperRecogniser()
    judge.lineage = "mindsurf-omni"

    with pytest.raises(ValueError, match="share lineage"):
        require_independent_judge(judge, model_lineage="mindsurf-omni")


def test_the_rule_is_enforced_not_merely_documented() -> None:
    """A convention only written down is one that gets skipped under deadline."""
    assert SenseVoiceRecogniser(model_dir="/x").eligible_as_judge is False
    assert WhisperRecogniser().eligible_as_judge is True


def _tone(seconds: float = 1.0, amplitude: float = 0.2, rate: int = 16_000) -> bytes:
    """Something with energy in it, without needing an audio fixture."""
    import math
    import struct

    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * 220 * index / rate))
        for index in range(int(rate * seconds))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


@pytest.mark.asyncio
async def test_silence_is_not_sent_to_the_recogniser() -> None:
    """Asked to read an empty room, SenseVoice invents rather than declines.

    Measured live on three seconds of digital silence: it returned the Korean
    "그.", the model answered that in English, and the caller heard a
    9.6-second reply to a button they had pressed by accident.
    """
    from typing import Any

    calls: list[Any] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            calls.append(kwargs)
            return [{"text": "그."}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    text, language = await recogniser.transcribe(b"\x00" * 16_000 * 2 * 3, 16_000)

    assert (text, language) == ("", None)
    assert calls == [], "the model was asked to read silence"


@pytest.mark.asyncio
async def test_audio_with_speech_in_it_still_reaches_the_recogniser() -> None:
    """The guard has to be quiet enough not to eat someone speaking softly."""
    from typing import Any

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            return [{"text": "<|zh|>你好"}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    text, _ = await recogniser.transcribe(_tone(), 16_000)

    assert text == "你好"


def test_the_silence_floor_sits_well_below_speech() -> None:
    """A threshold nobody can read is a threshold nobody keeps honest."""
    from mindsurf_omni.service.asr import SILENCE_RMS
    from mindsurf_omni.service.vad import frame_energy

    assert frame_energy(b"\x00" * 3200) < SILENCE_RMS
    # A quarter of full scale is ordinary speech: two orders of magnitude clear.
    assert frame_energy(_tone(0.1, amplitude=0.25)) > SILENCE_RMS * 50
