"""The target rebuilt from the transcript, and the two ways that goes wrong."""

from __future__ import annotations

from scripts.retarget_polish_pairs import injected_spans, strip_injected


def test_the_recogniser_s_punctuation_survives() -> None:
    """The whole point: the corpus sentence has no comma, the transcript does.

    Against the corpus sentence that comma is a delete label, and it is the
    single biggest one in the set -- 98 deletes against 253 keeps. Against this
    target it is not a label at all.
    """
    clean = "白衬衫领子发黄怎么办"
    # The injector writes 然后， and stops there. The comma after 发黄 and the
    # question mark are the recogniser's, which is exactly the difference this
    # target is built to stop scoring.
    spoken = "然后，白衬衫领子发黄怎么办"
    heard = "然后，白衬衫领子发黄，怎么办？"

    assert strip_injected(clean, spoken, heard) == "白衬衫领子发黄，怎么办？"


def test_the_filler_s_own_comma_goes_with_it() -> None:
    """Injection writes 那个， and the comma is part of what was added."""
    assert strip_injected("我想问一下", "那个，我想问一下", "那个，我想问一下。") == "我想问一下。"


def test_a_word_the_recogniser_merged_with_the_filler_is_kept() -> None:
    """Partial overlap means the filler was heard as part of a real word.

    Dropping the whole rewritten span would take the word with it, which is the
    invention this line exists to prevent -- so a span is only dropped when the
    whole of its spoken side was injected.
    """
    clean = "这个月我花了多少钱"
    spoken = "这个这个月我花了多少钱"
    # The recogniser folded the repeat into one, so the span it rewrote covers
    # injected and original characters together.
    heard = "这个月我花了多少钱"

    assert strip_injected(clean, spoken, heard) == "这个月我花了多少钱"


def test_a_sentence_with_nothing_injected_comes_back_as_the_transcript() -> None:
    """The clean fraction of the pool is what makes "change nothing" an answer."""
    assert strip_injected("冥想有用吗", "冥想有用吗", "冥想有用吗？") == "冥想有用吗？"


def test_the_injected_span_is_read_off_the_diff_not_the_record() -> None:
    """The record says which clause, not which character; the diff is exact."""
    added = injected_spans("下雨天怎么晾衣服", "嗯，下雨天怎么晾衣服")

    assert added[:2] == [True, True]
    assert not any(added[2:])
