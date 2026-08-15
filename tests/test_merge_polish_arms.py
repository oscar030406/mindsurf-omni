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

    # 嗯 comes in from the tagger. The comma it also dropped is not a filler
    # word, so the merge does not import it -- but it is then leading nothing,
    # and tidy() takes it. Leaving it was the stranded-punctuation defect that
    # a whole round of tuning reported as an improvement.
    assert merge([generator, tagger], "vocabulary-union") == "今天天气好"


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


def test_veto_keeps_a_deletion_only_where_the_other_arm_agrees() -> None:
    """The head's confidence protects content instead of removing more of it."""
    source = "阳台的花要浇水"
    generator = {"source": source, "polished": "的花要浇水"}  # dropped 阳台
    tagger = {"source": source, "polished": source}  # kept everything

    assert merge([generator, tagger], "veto") == source


def test_veto_never_blocks_a_repetition() -> None:
    """A per-token head cannot see 时间时间 at all, so it would veto every time.
    The tagger clears 0.437 of injected repetition against the generator's 0.603."""
    source = "时间时间上选星月前后"
    generator = {"source": source, "polished": "时间上选星月前后"}
    tagger = {"source": source, "polished": source}

    assert merge([generator, tagger], "veto") == "时间上选星月前后"


def test_veto_never_blocks_a_whole_filler_word() -> None:
    source = "嗯今天天气好"
    generator = {"source": source, "polished": "今天天气好"}
    tagger = {"source": source, "polished": source}

    assert merge([generator, tagger], "veto") == "今天天气好"


def test_one_repeated_character_is_not_a_repetition() -> None:
    """今天天气 holds 天天 and 看看 is an ordinary word. A floor of one buys
    0.003 of filler clearance, inside the noise, and pays with this."""
    from scripts.merge_polish_arms import repetition_spans

    assert repetition_spans("今天天气好", {1}) == set()
    assert repetition_spans("时间时间上", {0, 1}) == {0, 1}


def test_an_arm_that_stopped_early_is_not_read_as_agreeing() -> None:
    """A generative arm that emits its stop token early leaves the whole tail
    looking deleted. Measured by hand: 82 characters returned out of 184, and
    the veto then dropped the tail's commas because the truncation counted as
    agreement."""
    from scripts.merge_polish_arms import reached

    source = "嗯，今天天气好，我出门散步"
    truncated = {"source": source, "polished": "今天天气好"}  # stopped after 好
    tagger = {"source": source, "polished": "今天天气好我出门散步"}  # dropped the comma

    assert reached(source, truncated["polished"]) == 7
    # The tail survives: the truncated arm had no opinion there, so the comma
    # the tagger dropped is not an agreed deletion.
    assert merge([truncated, tagger], "veto") == "今天天气好，我出门散步"
