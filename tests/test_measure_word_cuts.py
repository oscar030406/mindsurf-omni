"""The cut reading, on outputs whose answer a person can check by eye.

Deletions inside a repeated run have several equivalent placements; judging one
arbitrary alignment reports broken words that are not broken."""

from __future__ import annotations

from scripts.measure_word_cuts import a_whole_word_placement_exists, both_copies_gone, cuts


def test_a_word_really_broken_is_reported() -> None:
    assert cuts("客户那边催了两次", "客户边催了两次") == ["边"]
    assert cuts("我们去图书馆看书", "我们去图馆看书") == ["图馆"]


def test_a_deletion_inside_a_repeated_run_is_not_a_cut() -> None:
    """太平太平洋 to 太平洋 deletes one 太平 -- whichever one you point at."""
    assert cuts("太平太平洋和大西洋", "太平洋和大西洋") == []
    assert cuts("久坐久坐久站，或者", "久坐久站，或者") == []
    assert cuts("能看能看到更多", "能看到更多") == []


def test_nothing_deleted_is_not_a_cut() -> None:
    assert cuts("完全一样的句子", "完全一样的句子") == []


def test_a_cut_that_spells_a_filler_is_not_counted() -> None:
    """jieba glues 就是 to its neighbour in 就是说; deleting the filler out of
    that is the vocabulary working, not a word breaking."""
    assert cuts("这个方案就是说可以", "这个方案说可以") == []


def test_the_placement_question_can_answer_no() -> None:
    """A reading that only ever says "fine" measures nothing."""
    assert a_whole_word_placement_exists("太平太平洋和大西洋", "太平洋和大西洋")
    assert not a_whole_word_placement_exists("客户那边催了两次", "客户边催了两次")


def test_both_copies_gone_still_sees_a_repetition_taken_whole() -> None:
    assert both_copies_gone("他的故事故事线简单", "他的线简单") == ["故事"]
    assert both_copies_gone("他的故事故事线简单", "他的故事线简单") == []
