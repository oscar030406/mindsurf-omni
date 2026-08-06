"""The pairing, and the eligibility check that retired the old ruler."""

from __future__ import annotations

import pytest
from scripts.measure_reply_length import (
    median_seconds_reading,
    paired_deltas,
)


def arm(lengths: dict[str, int]) -> dict[str, str]:
    return {probe: "字" * n for probe, n in lengths.items()}


def test_only_shared_probes_are_paired() -> None:
    """Subtracting across different probes subtracts two unrelated numbers."""
    candidate = arm({"a": 10, "b": 20, "only_here": 999})
    reference = arm({"a": 12, "b": 18, "only_there": 1})

    assert paired_deltas(candidate, reference) == [-2.0, 2.0]


def test_the_delta_is_candidate_minus_reference() -> None:
    """A shorter candidate reads negative, which is the direction three rounds
    of reports are written in."""
    assert paired_deltas(arm({"a": 40}), arm({"a": 55})) == [-15.0]


def test_a_ruler_wider_than_its_line_is_marked_ineligible() -> None:
    """The retired reading gated two rounds without the resolution to do it.

    The reference arm here spreads the way sft_merge's 608 replies do, so the
    bootstrap interval on its median is wider than the 7-character line.
    """
    # Spread like the real thing: sft_merge's 608 replies run from a few
    # characters to a few hundred, and it is that width -- not the sample
    # count -- that leaves the median loose.
    reference = arm({f"p{i}": 10 + (i * 37) % 200 for i in range(608)})
    candidate = arm({f"p{i}": 10 + (i * 37) % 200 for i in range(608)})

    reading = median_seconds_reading(candidate, reference, seed=1)

    lo, hi = reading["eligibility"]["reference_median_bootstrap_3sigma_chars"]
    assert hi - lo > reading["eligibility"]["line_chars"]
    assert reading["eligibility"]["gating_eligible"] is False


def test_a_tight_arm_keeps_its_eligibility() -> None:
    """The check is about this corpus, not a blanket dismissal of medians."""
    reference = arm({f"p{i}": 50 + (i % 3) for i in range(608)})
    candidate = arm({f"p{i}": 50 + (i % 3) for i in range(608)})

    reading = median_seconds_reading(candidate, reference, seed=1)

    assert reading["eligibility"]["gating_eligible"] is True


def test_the_old_verdict_is_still_computed_so_rounds_stay_comparable() -> None:
    reference = arm({f"p{i}": 55 for i in range(100)})
    candidate = arm({f"p{i}": 40 for i in range(100)})

    reading = median_seconds_reading(candidate, reference)

    assert reading["moved_seconds"] == pytest.approx(15 / 4.67, abs=1e-6)
    assert reading["old_verdict"] == "不过"
