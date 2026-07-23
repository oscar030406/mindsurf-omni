"""Catching the failure the audio metrics cannot see."""

from __future__ import annotations

import random

from mindsurf_omni.evaluation.text_regression import (
    BASELINE_STRICT_VAL,
    assess_text_regression,
)


def _losses(mean: float, spread: float = 0.02, count: int = 200, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, spread) for _ in range(count)]


def test_an_unchanged_model_is_reported_as_unchanged() -> None:
    verdict = assess_text_regression(_losses(BASELINE_STRICT_VAL))

    assert verdict.verdict == "unchanged"
    assert abs(verdict.difference) <= verdict.threshold


def test_a_real_regression_is_caught() -> None:
    """This is the whole point: audio training eroding the language ability."""
    verdict = assess_text_regression(_losses(BASELINE_STRICT_VAL + 0.15))

    assert verdict.verdict == "regressed"
    assert verdict.difference > 0


def test_an_improvement_is_named_as_such_not_as_a_regression() -> None:
    """Loss going down is good; a check that only looks at magnitude would confuse them."""
    verdict = assess_text_regression(_losses(BASELINE_STRICT_VAL - 0.15))

    assert verdict.verdict == "improved"


def test_a_difference_inside_the_combined_noise_is_not_called_a_regression() -> None:
    """Rerunning the same recipe would produce differences of this size."""
    verdict = assess_text_regression(_losses(BASELINE_STRICT_VAL + 0.001))

    assert verdict.verdict == "unchanged"


def test_too_few_batches_report_without_judging() -> None:
    """A dozen batches cannot resolve a difference the size of the seed spread."""
    verdict = assess_text_regression(_losses(BASELINE_STRICT_VAL + 0.5, count=12))

    assert verdict.verdict == "reported only"
    assert "below the 30 required" in verdict.note


def test_both_noise_sources_enter_the_threshold() -> None:
    """Sampling noise alone would call a rerun of the same recipe a change."""
    tight = assess_text_regression(_losses(BASELINE_STRICT_VAL, spread=0.0001, count=500))

    # Even with almost no sampling noise, the seed spread keeps a floor under
    # the threshold.
    assert tight.threshold >= 3.0 * 0.0020 * 0.99


def test_the_verdict_records_that_the_seed_spread_is_a_lower_bound() -> None:
    """It came from two seeds that varied initialisation only.

    A reader who takes it as a full estimate would over-trust a narrow verdict.
    """
    verdict = assess_text_regression(_losses(BASELINE_STRICT_VAL))

    assert "lower bound" in verdict.note


def test_the_verdict_prints_both_sides_of_the_comparison() -> None:
    """A bare "regressed" cannot be checked by whoever reads the report."""
    text = str(assess_text_regression(_losses(BASELINE_STRICT_VAL + 0.15)))

    assert "1.7268" in text  # the baseline
    assert "regressed" in text


def test_the_measurement_script_pins_what_makes_runs_comparable() -> None:
    """A different sequence length yields a number that looks comparable and is not.

    The baseline is 1.7268 at max_length 384. Measuring at another length and
    comparing against it is the quiet way to invent a regression or hide one.
    """
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "measure_strict_loss.py"
    ).read_text(encoding="utf-8")

    assert "BASELINE_MAX_LENGTH = 384" in script
    assert "不可比" in script  # warns when the length differs
    # The holdout's digest is printed, so two runs can be told apart.
    assert "sha256" in script


def test_the_measurement_script_refuses_a_checkpoint_that_is_not_the_thinker() -> None:
    """Loading the wrong file gives random weights and a plausible-looking loss."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "measure_strict_loss.py"
    ).read_text(encoding="utf-8")

    assert "not the Thinker" in script
    assert "loaded < 50" in script
