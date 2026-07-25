"""The listening panel, and the arithmetic that keeps it from overclaiming."""

from __future__ import annotations

import math

from scripts.listening_test import (
    OBSERVED_PAIRED_SD,
    RATER_SD_RANGE,
    required_clips,
    spearman,
    stratified_ids,
)


def test_twenty_clips_cannot_certify_the_effect_the_plan_asks_about() -> None:
    """The plan says 20 clips and 3 raters; the arithmetic says otherwise.

    This is the check the plan itself asked for -- work out what the panel can
    resolve before anyone spends an afternoon listening. If this assertion ever
    flips, the panel became a gate and the docs have to be rewritten with it.
    """
    for rater_sd in RATER_SD_RANGE:
        assert required_clips(0.29, raters=3, rater_sd=rater_sd) > 200


def test_more_raters_and_bigger_effects_both_shrink_the_requirement() -> None:
    """Sanity on the formula's direction, so a sign error cannot hide in it."""
    assert required_clips(0.29, 10, 0.8) < required_clips(0.29, 3, 0.8)
    assert required_clips(1.0, 3, 0.8) < required_clips(0.29, 3, 0.8)


def test_the_requirement_never_falls_below_the_clip_spread_alone() -> None:
    """Even with infinite raters the clip-by-clip difference still varies.

    A formula that let raters buy away the material's own spread would promise
    a panel that cannot exist.
    """
    floor_n = math.ceil((3 * 1.96 * OBSERVED_PAIRED_SD / 0.29) ** 2)
    assert required_clips(0.29, raters=1000, rater_sd=0.8) >= floor_n * 0.95


def test_spearman_matches_known_values() -> None:
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert abs(spearman([1, 2, 3, 4], [1, 3, 2, 4]) - 0.8) < 1e-9


def test_selection_spans_the_quality_range_rather_than_clustering() -> None:
    """Twenty drawn flat from a hundred can miss the bad tail entirely, and the
    bad tail is where a listener has something to report."""
    ids = [f"s{i:03d}" for i in range(100)]
    scores = {identifier: float(index) for index, identifier in enumerate(ids)}

    picked = stratified_ids(ids, scores, count=20, seed=1)

    assert len(picked) == 20
    values = [scores[identifier] for identifier in picked]
    # Something from the worst fifth and something from the best fifth.
    assert min(values) < 20
    assert max(values) > 79


def test_selection_without_scores_still_returns_the_asked_for_count() -> None:
    ids = [f"s{i}" for i in range(50)]

    assert len(stratified_ids(ids, {}, count=12, seed=1)) == 12
