"""Combining two arms' deletions, and the mistake that would cross two sentences."""

from __future__ import annotations

from scripts.merge_polish_arms import dropped, merge


def test_the_deletions_are_read_back_off_the_alignment() -> None:
    """No field records them; the two strings are what every arm writes."""
    assert dropped("嗯，今天天气怎么样", "今天天气怎么样") == {0, 1}
    assert dropped("今天天气怎么样", "今天天气怎么样") == set()


def test_union_deletes_what_either_arm_deleted() -> None:
    """The two get things wrong in different places, which is the whole point."""
    source = "嗯，今天那个天气怎么样"
    left = {"source": source, "polished": "今天那个天气怎么样"}  # dropped 嗯，
    right = {"source": source, "polished": "嗯，今天天气怎么样"}  # dropped 那个

    assert merge([left, right], "union") == "今天天气怎么样"


def test_intersection_deletes_only_what_both_agreed_on() -> None:
    source = "嗯，今天那个天气怎么样"
    left = {"source": source, "polished": "今天那个天气怎么样"}
    right = {"source": source, "polished": "今天天气怎么样"}

    # Only 嗯， is in both drop sets; 那个 survives because one arm kept it.
    assert merge([left, right], "intersection") == "今天那个天气怎么样"


def test_an_arm_that_changed_nothing_leaves_the_intersection_empty_handed() -> None:
    source = "嗯，今天天气怎么样"
    assert (
        merge(
            [
                {"source": source, "polished": "今天天气怎么样"},
                {"source": source, "polished": source},
            ],
            "intersection",
        )
        == source
    )


def test_a_replaced_span_counts_as_dropped() -> None:
    """Under the copy constraint an arm cannot substitute, so a replacement in
    the alignment is a deletion the matcher paired with unrelated context."""
    assert 0 in dropped("那个天气", "天气")


def test_vocabulary_union_takes_the_head_only_where_it_is_strong() -> None:
    """The tagger clears 0.983 of vocabulary filler and 0.437 of repetition.
    So take its deletions on filler words and ignore the rest of its opinion."""
    source = "嗯，今天今天天气好"
    generator = {"source": source, "polished": "嗯，今天天气好"}  # removed the repeat
    tagger = {"source": source, "polished": "今天今天天气好"}  # removed 嗯，

    # 嗯 comes in from the tagger; the comma it also dropped is not a filler word
    # and stays.
    assert merge([generator, tagger], "vocabulary-union") == "，今天天气好"


def test_a_partly_deleted_filler_is_not_imported() -> None:
    """Half a filler is the defect this avoids, not one to bring along."""
    source = "你知道吧今天天气好"
    generator = {"source": source, "polished": source}
    tagger = {"source": source, "polished": "你知道今天天气好"}  # dropped only 吧

    assert merge([generator, tagger], "vocabulary-union") == source


def test_the_first_arm_is_taken_whole() -> None:
    source = "今天今天天气好"
    generator = {"source": source, "polished": "今天天气好"}
    tagger = {"source": source, "polished": source}

    assert merge([generator, tagger], "vocabulary-union") == "今天天气好"
