"""The split only means anything if the parts add back up to the whole."""

from __future__ import annotations

from scripts.measure_polish import FOLD_NUMERALS, polished_cer
from scripts.split_polish_cer import CLASSES, classify, filler_mask, operations

from mindsurf_omni.evaluation.metrics import character_error_rate, normalise_for_cer


def test_the_parts_add_up_to_the_criterion() -> None:
    """If they did not, the split would be a second opinion rather than a
    breakdown, and pushing on the largest share would be pushing on nothing."""
    target = "会议改到下午六点，因为上午张老师有课，地点在B203"
    polished = "那个会议改到下午6点，上午张老师有可，地点在B203"

    counts, length, _ = classify(target, polished)

    assert length == len(normalise_for_cer(target, fold_numbers=FOLD_NUMERALS))
    assert sum(counts.values()) / length == polished_cer(target, polished)


def test_the_edit_script_costs_exactly_the_edit_distance() -> None:
    """difflib's script would be shorter to write and longer than the distance,
    because it spends a delete and an insert where one substitution does."""
    reference, hypothesis = "慕田峪长城", "牧田御长城"

    script = operations(reference, hypothesis)

    assert len(script) == 2
    assert all(kind == "substitute" for kind, _, _ in script)
    assert len(script) == round(character_error_rate(reference, hypothesis) * len(reference))


def test_filler_left_in_and_content_taken_out_are_not_the_same_error() -> None:
    """One is the polisher not working, the other is it working on the wrong
    characters. They cost the same in CER and want opposite fixes."""
    target = "会议改到明天下午"

    lazy, _, _ = classify(target, "那个会议改到明天下午")
    greedy, _, _ = classify(target, "会议改到明天")

    assert lazy["filler_left"] == 2 and lazy["content_deleted"] == 0
    assert greedy["content_deleted"] == 2 and greedy["filler_left"] == 0


def test_deleting_the_corpus_own_filler_is_charged_but_named() -> None:
    """The targets keep native filler -- only the injected spans were stripped --
    so a dictation product doing the right thing lands in the error column. It
    has to be visible, or the gap looks bigger than the part worth closing."""
    target = "我们那个下周再说"

    counts, _, _ = classify(target, "我们下周再说")

    assert counts["native_filler_deleted"] == 2
    assert counts["content_deleted"] == 0


def test_a_filler_the_recogniser_spelled_differently_is_its_own_class() -> None:
    """呃 arriving as 恶 is filler the door cannot open on, and the first cut of
    this file called it "repetition and restarts" -- which would have sent the
    next round at the tagger when the target is the vocabulary."""
    counts, _, _ = classify("猫喜欢纸箱", "猫喜欢恶纸箱")

    assert counts["misspelled_filler"] == 1
    assert counts["kept_other"] == 0


def test_a_spelling_the_service_never_listed_stays_in_the_leftover_class() -> None:
    """遏 and a bare n both turned up standing in for a filler on the hold-out,
    and neither is in the service's list -- 遏制 is a word, so adding it is not
    free. The class has to keep them rather than quietly claim coverage."""
    counts, _, _ = classify("猫喜欢纸箱", "猫喜欢遏纸箱")

    assert counts["kept_other"] == 1
    assert counts["misspelled_filler"] == 0


def test_a_digit_disagreement_belongs_to_the_fold_not_the_model() -> None:
    """cn2an turns 四万 into 40000 and leaves 4万 alone, so both sides can be
    right about the number and still differ. Counting that as over-deletion is
    reading the ruler as evidence about the polisher."""
    counts, _, _ = classify("壁画四万多平", "壁画4万多平")

    assert counts["numeral"] > 0
    assert counts["content_deleted"] == 0


def test_a_substitution_is_the_recognisers_and_says_so() -> None:
    counts, _, _ = classify("慕田峪长城值得去", "牧田御长城值得去")

    assert counts["substituted"] == 2
    assert all(counts[name] == 0 for name in CLASSES if name != "substituted")


def test_the_vocabulary_is_matched_literally_and_that_is_a_judgement() -> None:
    """那个 inside 那个人 is content, and this marks it as filler. Stated in the
    docstring, pinned here so nobody reads the number as a defect count."""
    mask = filler_mask("那个人来了")

    assert mask[:2] == [True, True]
    assert mask[2:] == [False, False, False]
