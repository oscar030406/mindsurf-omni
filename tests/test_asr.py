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
