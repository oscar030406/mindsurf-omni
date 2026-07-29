"""The two checks that decide whether a harvested reference is usable.

Neither is about the harvesting. Both are about the ways a harvested pair can
look fine and not be: two clips of different people that the clustering bar let
through, and an English clip conditioning Chinese generation. The first is
guarded by a threshold set against the corpus's own distribution; the second is
what this file covers, plus the frame-major unpacking that produces a plausible
tensor of noise if read the other way.
"""

from __future__ import annotations

from scripts.harvest_emotion_voices import SAME_SPEAKER, is_chinese


def test_the_speaker_bar_sits_past_the_corpus_background() -> None:
    """0.79 is the median between two random rows and 0.9172 the 99th.

    A bar inside that range would cluster strangers together and every downstream
    number would describe a chimera. It is set from the measured distribution
    rather than from what looks like a high number.
    """
    assert SAME_SPEAKER > 0.9172


def test_an_english_reference_is_recognised_as_not_chinese() -> None:
    """The first harvested pair came back in English, and it transcribed cleanly.

    Cleanliness is not the question -- a reference conditions voice and prosody,
    and an English one driving Chinese generation is a mismatch nobody here has
    measured. It gets skipped rather than silently shipped.
    """
    assert not is_chinese("wonder woman features a strong female lead played by galgado")
    assert is_chinese("大 家 好 欢 迎 收 听 今 天 的 节 目")
    # Chinese with digits and punctuation is still Chinese.
    assert is_chinese("今天是 2026 年 7 月 29 日，天气不错。")


def test_an_empty_transcript_is_not_chinese() -> None:
    """Empty is how the manufactured references failed; it must not pass a
    language check by having no counter-evidence in it."""
    assert not is_chinese("")
    assert not is_chinese("   ")


def test_reference_codes_are_unpacked_frame_major() -> None:
    """Read the other way this produces a tensor of the right shape and no meaning.

    Taken from omni_dataset.py, which walks the flat list eight at a time and
    appends one value per codebook.
    """
    import pytest

    pytest.importorskip("torch")

    from scripts.harvest_emotion_voices import unpack

    # Two frames, codebook j carrying value j in the first and 10+j in the second.
    flat = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]

    codes = unpack(flat)

    assert tuple(codes.shape) == (8, 2)
    assert codes[0].tolist() == [0, 10]
    assert codes[7].tolist() == [7, 17]
