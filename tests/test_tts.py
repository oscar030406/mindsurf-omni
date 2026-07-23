"""Synthesis input cleaning, and the screen for instructions read aloud."""

from __future__ import annotations

import pytest

from mindsurf_omni.service.tts import (
    EMOTION_INSTRUCTIONS,
    Utterance,
    clean_for_speech,
    instruction_leaked,
    screen_batch,
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


def test_an_instruction_read_aloud_is_detected() -> None:
    """The real failure: the reply is there, with the instruction in front."""
    transcript = EMOTION_INSTRUCTIONS["happy"] + "今天天气真好"

    assert instruction_leaked("今天天气真好", transcript)


def test_a_clean_transcript_is_not_flagged() -> None:
    assert not instruction_leaked("今天天气真好", "今天天气真好")
    assert not instruction_leaked("今天天气真好", "")


def test_a_reply_that_merely_discusses_tone_is_not_flagged() -> None:
    """Containment would fire here; a prefix match does not."""
    assert not instruction_leaked("x", "这句话应该用开心热情的语气说才自然")


def test_screening_covers_the_batch_not_a_sample() -> None:
    """The leak is intermittent, so a spot check finds it in the batches it spared."""
    utterances = [Utterance(text=f"第{i}句") for i in range(4)]
    transcripts = [
        "第0句",
        EMOTION_INSTRUCTIONS["neutral"] + "第1句",
        "第2句",
        "",  # produced no speech at all
    ]

    suspect = screen_batch(utterances, transcripts)

    assert [utterance.text for utterance, _ in suspect] == ["第1句", "第3句"]


def test_misaligned_inputs_are_refused_rather_than_silently_zipped() -> None:
    """A shifted pair would blame the wrong sample and hide the real one."""
    with pytest.raises(ValueError, match="misaligned|against"):
        screen_batch([Utterance(text="a")], ["a", "b"])
