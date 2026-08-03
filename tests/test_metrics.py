"""What the metrics promise, including when they refuse to judge."""

from __future__ import annotations

import random

import pytest

from mindsurf_omni.evaluation.metrics import (
    assess,
    bootstrap_noise_floor,
    character_error_rate,
    cluster_bootstrap_noise_floor,
    compare,
    compare_paired,
    compare_paired_clustered,
    edit_distance,
    normalise_for_cer,
)


def test_identical_text_scores_zero() -> None:
    assert character_error_rate("今天天气真好", "今天天气真好") == 0.0


def test_punctuation_and_case_do_not_count_as_errors() -> None:
    """Otherwise the score measures the recogniser's punctuation habits."""
    assert character_error_rate("今天天气真好。", "今天天气真好") == 0.0
    assert character_error_rate("Hello, World!", "hello world") == 0.0


def test_the_judges_choice_of_script_does_not_count_as_error() -> None:
    """The single largest term in the measured floor, and none of it was speech.

    Whisper picks traditional or simplified per utterance regardless of the
    input. Over the hundred-probe floor run this alone was mean CER 0.1788
    against 0.0388 with it folded away -- four fifths of the number.
    """
    assert character_error_rate("地铁几点开始运营", "地鐵幾點開始運營") == 0.0
    assert character_error_rate("为什么树叶秋天会变黄", "為什麼樹葉秋天會變黃") == 0.0


def test_folding_script_does_not_forgive_a_wrong_word() -> None:
    """It must collapse the writing system, not the vocabulary."""
    assert character_error_rate("猫为什么喜欢纸箱", "毛為什麼喜歡紙箱") == pytest.approx(1 / 8)


def test_folding_script_does_not_forgive_regional_vocabulary() -> None:
    """軟體 and 软件 are the same idea and different words, spoken differently.

    A locale conversion maps one to the other and would score them equal; a
    script fold makes 軟體 into 软体, which stays distinct from 软件. The
    difference is what keeps this a measurement of whether the audio said the
    words rather than whether it meant them.
    """
    assert character_error_rate("软件很好用", "軟體很好用") > 0.0


def test_reading_a_date_aloud_is_only_forgiven_when_asked() -> None:
    """The worst pair on the fixed 160 both arms scored 0.500, and correctly.

    The reference says 2021年9月23日, the synthesiser reads it out as
    二零二一年九月二十三日, the judge transcribes what it heard, and every
    digit costs a substitution. Folding both sides makes it free -- but the
    retrain's thresholds were calibrated unfolded, so it has to be asked for.
    """
    reference, spoken = "2021年9月23日", "二零二一年九月二十三日"
    assert character_error_rate(reference, spoken) > 0.4
    pytest.importorskip("cn2an")  # the fold lives in the optional evaluate extra
    assert character_error_rate(reference, spoken, fold_numbers=True) == 0.0


def test_folding_numerals_does_not_forgive_a_wrong_number() -> None:
    pytest.importorskip("cn2an")  # the fold lives in the optional evaluate extra
    assert character_error_rate("9月23日", "9月24日", fold_numbers=True) > 0.0


def test_folding_numerals_hits_both_sides_of_an_ordinary_word() -> None:
    """一般 becomes 1般 -- harmless because it happens to reference and
    hypothesis alike, which is the same argument the script fold rests on."""
    pytest.importorskip("cn2an")  # the fold lives in the optional evaluate extra
    assert character_error_rate("一般的问题", "一般的问题", fold_numbers=True) == 0.0
    assert normalise_for_cer("一般", fold_numbers=True) == "1般"


def test_spacing_differences_do_not_count() -> None:
    """Chinese output is unspaced and English is not; keeping spaces would
    make the two scripts incomparable."""
    assert character_error_rate("今天 天气 真好", "今天天气真好") == 0.0


@pytest.mark.parametrize(
    "reference,hypothesis,expected",
    [
        ("今天天气真好", "今天天气真差", 1 / 6),  # one substitution
        ("今天天气真好", "今天天气真", 1 / 6),  # one deletion
        ("今天天气真好", "今天天气真好啊", 1 / 6),  # one insertion
    ],
)
def test_each_edit_type_costs_one(reference: str, hypothesis: str, expected: float) -> None:
    assert character_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_a_model_that_rambles_scores_worse_than_one_that_stops() -> None:
    """Clamping at 1.0 would hide runaway generation, which is a real failure.

    The pretraining phase shipped an instrument that rewarded stopping early;
    this one must not repeat it in the opposite direction.
    """
    rambling = character_error_rate("好的", "好的" + "然后呢" * 20)

    assert rambling > 1.0


def test_empty_reference_with_output_is_infinite_not_zero() -> None:
    """Saying something when nothing was expected is not a perfect score."""
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "意外的输出") == float("inf")


def test_edit_distance_matches_known_values() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3


def test_normalisation_keeps_the_characters_a_listener_hears() -> None:
    assert normalise_for_cer("Ｈｅｌｌｏ，世界！") == "hello世界"


def test_noise_floor_shrinks_as_samples_grow() -> None:
    """The reason a 192-question set could not resolve a 3pp effect."""
    import random

    rng = random.Random(0)
    small = [rng.gauss(0.1, 0.05) for _ in range(20)]
    large = [rng.gauss(0.1, 0.05) for _ in range(2000)]

    assert bootstrap_noise_floor(small) > bootstrap_noise_floor(large) * 3


def test_a_single_sample_admits_it_cannot_estimate_noise() -> None:
    assert bootstrap_noise_floor([0.1]) == float("inf")


def test_clustered_rows_get_a_wider_floor_than_pretending_they_are_independent() -> None:
    """The voice-identification shape: twenty clips of a voice are almost all
    right or almost all wrong together, so the unit that varies is the voice.
    Resampling rows counts 240 independent draws where there are only 12."""
    all_right = [1.0] * 20
    all_wrong = [0.0] * 20
    groups = [all_right] * 6 + [all_wrong] * 6
    rows = [value for group in groups for value in group]

    assert cluster_bootstrap_noise_floor(groups) > bootstrap_noise_floor(rows) * 2


def test_one_cluster_cannot_estimate_between_cluster_noise() -> None:
    """Twenty clips of a single voice say nothing about how the next voice
    behaves, and a finite floor here would invite exactly that claim."""
    assert cluster_bootstrap_noise_floor([[1.0] * 20]) == float("inf")


def test_empty_clusters_do_not_count_toward_the_unit_of_variation() -> None:
    assert cluster_bootstrap_noise_floor([[1.0] * 5, []]) == float("inf")


def test_a_clustered_effect_can_read_improved_per_row_and_indistinguishable_per_cluster() -> None:
    """The verdict this function exists to correct, in miniature.

    A small mean carried by a few clusters looks solid when every row counts as
    its own evidence, and stops looking solid when the clusters do. Reporting
    the first is how a project publishes a direction it has not earned.
    """
    groups = [[0.05] * 20 for _ in range(9)] + [[-0.02] * 20 for _ in range(3)]
    rows = [value for group in groups for value in group]

    assert "improved" in compare_paired("m", rows, lower_is_better=False)
    assert "indistinguishable" in compare_paired_clustered("m", groups, lower_is_better=False)


def test_a_clustered_comparison_needs_more_than_one_cluster() -> None:
    assert "reported only" in compare_paired_clustered("m", [[0.5] * 40])


def test_an_instrument_too_coarse_for_the_effect_is_refused() -> None:
    """The multiple-choice failure, stated as arithmetic.

    Resolution +/-7pp against an effect of 3pp is not a weak instrument; it is
    the wrong one, and its direction is noise.
    """
    import random

    rng = random.Random(1)
    noisy = [rng.gauss(0.5, 0.35) for _ in range(200)]

    measurement = assess("mcq.accuracy", noisy, effect_of_interest=0.03)

    assert not measurement.gating_eligible
    assert "effect of interest" in measurement.note


def test_a_small_sample_is_refused_however_tight_it_looks() -> None:
    """Ten probes flipping a verdict between 5 and 4 is a coin toss."""
    measurement = assess("generation.degenerate", [0.0] * 10, effect_of_interest=0.5)

    assert not measurement.gating_eligible
    assert "below the 100 required" in measurement.note


def test_a_sharp_instrument_on_enough_samples_may_judge() -> None:
    import random

    rng = random.Random(2)
    tight = [rng.gauss(0.12, 0.02) for _ in range(500)]

    measurement = assess("cer", tight, effect_of_interest=0.05)

    assert measurement.gating_eligible
    assert measurement.note == ""


def test_no_samples_is_not_a_score_of_zero() -> None:
    measurement = assess("cer", [], effect_of_interest=0.05)

    assert not measurement.gating_eligible
    assert measurement.sample_size == 0


def test_a_difference_inside_the_noise_has_no_direction() -> None:
    """The single rule that stops a project talking itself into a result."""
    import random

    rng = random.Random(3)
    candidate = assess("cer", [rng.gauss(0.120, 0.02) for _ in range(500)], 0.05)
    reference = assess("cer", [rng.gauss(0.121, 0.02) for _ in range(500)], 0.05, seed=1)

    assert "indistinguishable" in compare("cer", candidate, reference)


def test_a_real_improvement_is_reported_with_its_margin() -> None:
    import random

    rng = random.Random(4)
    candidate = assess("cer", [rng.gauss(0.10, 0.02) for _ in range(500)], 0.05)
    reference = assess("cer", [rng.gauss(0.18, 0.02) for _ in range(500)], 0.05, seed=1)

    verdict = compare("cer", candidate, reference)

    assert "improved" in verdict


def test_direction_follows_whether_lower_is_better() -> None:
    import random

    rng = random.Random(5)
    candidate = assess("utmos", [rng.gauss(4.0, 0.2) for _ in range(500)], 0.3)
    reference = assess("utmos", [rng.gauss(3.0, 0.2) for _ in range(500)], 0.3, seed=1)

    assert "improved" in compare("utmos", candidate, reference, lower_is_better=False)
    assert "regressed" in compare("cer", candidate, reference, lower_is_better=True)


def test_an_ineligible_instrument_reports_but_never_judges() -> None:
    """It may appear in the report; it may not decide pass or fail."""
    candidate = assess("mcq", [0.5] * 10, effect_of_interest=0.03)
    reference = assess("mcq", [0.4] * 10, effect_of_interest=0.03, seed=1)

    verdict = compare("mcq", candidate, reference)

    assert "reported only" in verdict
    assert "improved" not in verdict and "regressed" not in verdict


def test_two_reference_sets_that_disagree_may_not_gate() -> None:
    """The failure this exists for: the length-tuned arm read `regressed` on one
    neutral author and `improved` on the other, and per probe the two agreed in
    sign 20% of the time -- worse than a coin."""
    from mindsurf_omni.evaluation.metrics import cross_reference_agreement

    rising = [0.1] * 40 + [-0.1] * 10
    falling = [-0.1] * 40 + [0.1] * 10

    verdict = cross_reference_agreement(rising, falling)

    assert verdict["agreement"] == 0.0
    assert not verdict["gating_eligible"]


def test_two_reference_sets_that_agree_may_gate() -> None:
    from mindsurf_omni.evaluation.metrics import cross_reference_agreement

    one = [0.1] * 45 + [-0.1] * 5
    two = [0.1] * 43 + [-0.1] * 7

    verdict = cross_reference_agreement(one, two)

    assert verdict["agreement"] > 0.8
    assert verdict["gating_eligible"]


def test_the_round_that_agreed_only_just_clears_the_bar() -> None:
    """59% on 158 probes is what the DPO round whose aggregates agreed actually
    scored. It passes -- by 0.012 over a noise band of 0.078 -- and recording
    that here is the point: the check separates that round from the length arm's
    20%, but it does not call it comfortable."""
    from mindsurf_omni.evaluation.metrics import cross_reference_agreement

    n = 158
    agree = round(0.59 * n)
    one = [0.1] * n
    two = [0.1] * agree + [-0.1] * (n - agree)

    verdict = cross_reference_agreement(one, two)

    assert 0.58 < verdict["agreement"] < 0.60
    assert verdict["gating_eligible"]
    assert verdict["agreement"] - 0.5 - verdict["noise"] < 0.02


def test_a_few_probes_fewer_and_the_same_rate_would_not_clear() -> None:
    """The margin above depends on the set size, so a smaller probe set at the
    same agreement is not eligible -- which is the honest reading of 59%."""
    from mindsurf_omni.evaluation.metrics import cross_reference_agreement

    n = 80
    agree = round(0.59 * n)
    one = [0.1] * n
    two = [0.1] * agree + [-0.1] * (n - agree)

    assert not cross_reference_agreement(one, two)["gating_eligible"]


def test_a_sharp_instrument_may_not_convict_below_the_effect_of_interest() -> None:
    """chat_nll failed the length arm at 0.0479 while 0.05 was the declared line."""
    deltas = [0.0479 + (0.001 if index % 2 else -0.001) for index in range(158)]
    without = compare_paired("chat_nll", deltas, effect_of_interest=None)
    with_line = compare_paired("chat_nll", deltas, effect_of_interest=0.05)
    assert "regressed" in without and "仅报告" not in without
    assert "regressed" in with_line and "低于关心的效应" in with_line


def test_a_real_regression_above_the_line_still_gates() -> None:
    deltas = [0.77 + (0.01 if index % 2 else -0.01) for index in range(158)]
    verdict = compare_paired("chat_nll", deltas, effect_of_interest=0.05)
    assert "regressed" in verdict and "仅报告" not in verdict


def test_a_blunt_instrument_may_not_certify_a_null() -> None:
    """rank-1 resolved 0.70 where 0.10 mattered, and passed a guardrail on it."""
    # Six voices move one way and six the other, so the mean sits near zero
    # while the spread between voices keeps the clustered floor wide -- the
    # shape the identification axis actually has.
    rng = random.Random(11)
    groups = [
        [(1.0 if index < 6 else -1.0) + rng.gauss(0, 0.05) for _ in range(20)]
        for index in range(12)
    ]
    verdict = compare_paired_clustered("rank1", groups, effect_of_interest=0.10)
    assert "indistinguishable" in verdict
    assert "仅报告" in verdict and "读不出这么小的劣化" in verdict


def test_a_null_from_an_instrument_sharp_enough_still_gates() -> None:
    deltas = [0.0005 if index % 2 else -0.0005 for index in range(240)]
    verdict = compare_paired("f0", deltas, effect_of_interest=15.0)
    assert "indistinguishable" in verdict and "仅报告" not in verdict


def test_omitting_the_effect_of_interest_changes_nothing() -> None:
    """Every existing caller keeps its verdict until it opts in."""
    rng = random.Random(3)
    deltas = [rng.gauss(0.02, 0.01) for _ in range(200)]
    assert compare_paired("m", deltas) == compare_paired("m", deltas, effect_of_interest=None)
