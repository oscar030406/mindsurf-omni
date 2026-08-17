"""Combining two arms' deletions, and the mistake that would cross two sentences."""

from __future__ import annotations

# Moved into the service on 2026-08-16, with tidy, because the product runs the
# merge now. Imported from where it lives rather than re-exported.
from mindsurf_omni.service.polish import dropped, merge


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
    from mindsurf_omni.service.polish import repetition_spans

    assert repetition_spans("今天天气好", {1}) == set()
    assert repetition_spans("时间时间上", {0, 1}) == {0, 1}


def test_an_arm_that_stopped_early_is_not_read_as_agreeing() -> None:
    """A generative arm that emits its stop token early leaves the whole tail
    looking deleted. Measured by hand: 82 characters returned out of 184, and
    the veto then dropped the tail's commas because the truncation counted as
    agreement."""
    from mindsurf_omni.service.polish import reached

    source = "嗯，今天天气好，我出门散步"
    truncated = {"source": source, "polished": "今天天气好"}  # stopped after 好
    tagger = {"source": source, "polished": "今天天气好我出门散步"}  # dropped the comma

    assert reached(source, truncated["polished"]) == 7
    # The tail survives: the truncated arm had no opinion there, so the comma
    # the tagger dropped is not an agreed deletion.
    assert merge([truncated, tagger], "veto") == "今天天气好，我出门散步"


def test_a_repetition_loses_one_copy_not_both() -> None:
    """Measured on a dictated note: 因为，因为上午张老师有课 came back with no
    因为 at all and the causal link with it. Over 986 held-out transcripts the
    generator alone never did this; the merged arm did it in 11 sentences."""
    from mindsurf_omni.service.polish import keep_one_copy

    source = "因为因为上午张老师有课"
    both = set(range(0, 4))  # both copies of 因为

    kept = keep_one_copy(source, both)

    assert "".join(c for i, c in enumerate(source) if i not in kept) == "因为上午张老师有课"


def test_a_repeated_filler_still_loses_both_copies() -> None:
    """Both copies of 就是就是 are filler; taking both is the vocabulary doing
    its job, and counting it as damage read 1.52% where the real rate is a
    fifth of that."""
    from mindsurf_omni.service.polish import keep_one_copy

    source = "就是就是这样"
    both = set(range(0, 4))

    assert keep_one_copy(source, both) == both


def test_a_repetition_the_arms_left_alone_is_untouched() -> None:
    from mindsurf_omni.service.polish import keep_one_copy

    assert keep_one_copy("确定确定的事", set()) == set()


def test_a_word_is_dropped_whole_or_not_at_all() -> None:
    """Deletion is per character and nothing held it to a word boundary, so it
    ate 那 out of 那边 and left 客户边. Measured on 986 held-out transcripts,
    63 of them carried at least one such cut."""
    from mindsurf_omni.service.polish import whole_words

    source = "客户那边催了"
    # 那 alone, out of the word 那边.
    kept = whole_words(source, {2})

    assert kept == set()


def test_a_cut_that_spells_a_filler_is_left_alone() -> None:
    """jieba glues 就是 to its neighbour in 就是说, and undoing that deletion
    would undo the vocabulary's own work."""
    from mindsurf_omni.service.polish import whole_words

    source = "就是说需要发票"
    drop = {0, 1}  # 就是, out of the word 就是说

    assert whole_words(source, drop) == drop


def test_a_word_dropped_whole_stays_dropped() -> None:
    from mindsurf_omni.service.polish import whole_words

    source = "那个报销的事情"
    drop = {0, 1}  # 那个, a whole word

    assert whole_words(source, drop) == drop


def test_either_copy_of_a_repetition_counts_as_removing_one() -> None:
    """The first version asked only about the first copy, so an arm that removed
    the second imported nothing and the veto blocked the whole judgement -- which
    reads as the tagger not seeing the repetition rather than as this function
    not recognising it. Ten of the 24 repetitions the tagger wanted gone and the
    product kept were that shape: 发挥发挥, 故事故事, 鼻涕鼻涕."""
    from mindsurf_omni.service.polish import repetition_spans

    assert repetition_spans("时间时间上", {0, 1}) == {0, 1}
    assert repetition_spans("时间时间上", {2, 3}) == {2, 3}
    assert repetition_spans("时间时间上", {1, 2}) == set()


def test_a_tagger_that_deletes_the_second_copy_is_no_longer_vetoed() -> None:
    """End to end through the merge, because the import is only worth anything
    if the veto then lets it through."""
    from mindsurf_omni.service.polish import merge

    source = "他的故事故事线简单"
    generator = {"source": source, "polished": source}
    tagger = {"source": source, "polished": "他的故事线简单"}

    assert merge([generator, tagger], "veto") == "他的故事线简单"
