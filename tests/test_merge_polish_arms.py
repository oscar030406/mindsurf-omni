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


def test_a_cut_inside_a_repeated_run_is_rounded_to_whole_copies() -> None:
    """The deployed service wrote 久坐久坐站 for 久坐久坐久站.

    Inside a run of period two the copies are interchangeable -- deleting 久坐
    at 12, 久坐 at 14 or 坐久 at 15 all spell 久坐久站 -- so difflib recorded the
    deletion at the rightmost one and jieba, which only knows the leftmost,
    handed back half of it. Half a copy is not a smaller edit than a whole one,
    it is a different and wrong one."""
    from mindsurf_omni.service.polish import merge

    source = "盐分摄入过多，久坐久坐久站，或者睡眠不足"
    generator = {"source": source, "polished": "盐分摄入过多，久坐久站，或者睡眠不足"}
    tagger = {"source": source, "polished": "盐分摄入过多，坐坐久站，或者睡眠不足"}

    assert merge([generator, tagger], "veto") == "盐分摄入过多，久坐久站，或者睡眠不足"


def test_rounding_a_cut_never_reaches_into_the_word_after_the_run() -> None:
    """The rejected fix for this defect moved the deletion's phase instead of
    its size, which silently switched off the whole-word repair for the rest of
    the sentence: 绿茶树 lost its 树 and 想吃面还是想吃炒菜 lost its 面, both on
    inputs where the two arms agree. Rounding to whole copies leaves the run's
    neighbours alone by construction, and these two are the guard on that."""
    from mindsurf_omni.service.polish import merge

    for source, arm, expected in (
        (
            "准备绿茶叶，如绿茶花、绿茶树、绿茶、绿茶豆等。",
            "准备绿茶叶，如绿茶花、绿茶、绿茶豆等。",
            "准备绿茶叶，如绿茶花、绿茶树、绿茶豆等。",
        ),
        (
            "先定的大方向，比如今天想吃面还是想吃炒菜",
            "先定的大方向，比如今天想吃炒菜",
            "先定的大方向，比如今天想吃面炒菜",
        ),
    ):
        assert merge([{"source": source, "polished": arm}] * 2, "veto") == expected


def test_rounding_leaves_ordinary_reduplication_alone() -> None:
    """A run of period one is 慢慢 and 试试, where "one copy" is a single
    character and the interchangeable-copies reasoning does not hold."""
    from mindsurf_omni.service.polish import merge

    source = "我慢慢看，你试试，出去走走。"
    assert merge([{"source": source, "polished": source}] * 2, "veto") == source


def test_rounding_leaves_a_repeated_filler_to_keep_one_copy() -> None:
    """就是就是 loses both copies on purpose -- that judgement belongs to
    keep_one_copy, and rounding must not put one of them back."""
    from mindsurf_omni.service.polish import merge

    source = "这个方案就是就是可以的"
    arm = {"source": source, "polished": "这个方案可以的"}

    assert merge([arm, arm], "veto") == "这个方案可以的"


def test_rounding_never_takes_a_run_down_to_nothing() -> None:
    """Deleting every copy is a judgement the stages above make deliberately;
    this one only decides where an existing cut sits."""
    from mindsurf_omni.service.polish import snap_periods

    source = "他的故事故事线简单"
    # Both copies asked for: left exactly as it came in, for keep_one_copy to own.
    assert snap_periods(source, {2, 3, 4, 5}) == {2, 3, 4, 5}
    # One whole copy asked for: already whole, so nothing moves.
    assert snap_periods(source, {4, 5}) == {4, 5}
    # Half a copy asked for, plus a character outside the run: the half is
    # rounded up to one whole copy, and what lies outside the run is not this
    # function's business.
    assert snap_periods(source, {5, 6}) == {2, 3, 6}


def test_a_stutter_across_a_word_boundary_is_not_handed_back() -> None:
    """土长土长期 kept its stutter all the way to the deployed output. An arm
    asked for 土长, jieba reads 土/长期, and the word constraint handed the copy
    back -- 19 of the 66 repetitions that survived, every one by this route."""
    from mindsurf_omni.service.polish import merge

    source = "浇水太多，土长土长期潮湿，盆底积水"
    arm = {"source": source, "polished": "浇水太多，土长期潮湿，盆底积水"}

    assert merge([arm, arm], "veto") == "浇水太多，土长期潮湿，盆底积水"


def test_a_question_is_not_a_stutter() -> None:
    """是不是不舒服 is how Chinese asks; taking one 是不 leaves 是不舒服, which
    answers instead. Same for the written 是否, and for 2020法则, which names a
    rule that 20法则 does not."""
    from mindsurf_omni.service.polish import merge

    for source, arm in (
        ("先看看是不是不舒服，别强迫", "先看看是不舒服，别强迫"),
        ("每20分钟看远处，2020法则", "每20分钟看远处，20法则"),
    ):
        assert merge([{"source": source, "polished": arm}] * 2, "veto") == source


def test_the_stutter_test_reads_the_unit_not_the_sentence() -> None:
    from mindsurf_omni.service.polish import stutter

    assert stutter("土长") and stutter("花椒") and stutter("慢慢")
    assert not stutter("是不") and not stutter("是否")
    assert not stutter("2020") and not stutter("20") and not stutter("5G")


def test_the_copies_are_reported_one_at_a_time_as_well_as_flattened() -> None:
    """A stage that takes some copies and leaves others needs to know which
    characters belong together; the flattened set cannot say."""
    from mindsurf_omni.service.polish import repetition_copies, repetition_spans

    source = "他的故事故事线简单"

    assert repetition_copies(source, {2, 3}) == [{2, 3}]
    assert repetition_copies(source, {4, 5}) == [{4, 5}]
    assert repetition_copies(source, {3, 4}) == []
    assert repetition_spans(source, {4, 5}) == {4, 5}


def test_a_repeated_word_neither_arm_saw_is_still_taken() -> None:
    """Of the 66 repetitions that survived into the deployed output, 45 were
    proposed by neither arm. No amount of merging recovers a deletion nobody
    asked for, so this one rule proposes -- under a word constraint."""
    from mindsurf_omni.service.polish import merge

    source = "四川菜里花椒花椒带来的麻"
    arm = {"source": source, "polished": source}

    assert merge([arm, arm], "veto") == "四川菜里花椒带来的麻"


def test_the_proposing_rule_needs_a_boundary_on_all_three_sides() -> None:
    """The word constraint is the whole safety argument, so the shapes that are
    not stutters are checked rather than assumed: 是不是不舒服 segments as
    是不是/不/舒服, 2020法则 as 2020/法则, and 慢慢 试试 走走 are single tokens.
    None offers a boundary-aligned pair."""
    from mindsurf_omni.service.polish import merge

    for source in (
        "先看看是不是不舒服，别强迫",
        "412时2020法则，每20分钟看远处",
        "我慢慢看，你试试，出去走走",
    ):
        arm = {"source": source, "polished": source}
        assert merge([arm, arm], "veto") == source


def test_the_proposing_rule_leaves_repeated_filler_to_keep_one_copy() -> None:
    """就是就是 loses both copies on purpose; taking one here would put the
    other back."""
    from mindsurf_omni.service.polish import duplicate_words

    source = "这个方案就是就是可以的"

    assert duplicate_words(source, set()) == set()
