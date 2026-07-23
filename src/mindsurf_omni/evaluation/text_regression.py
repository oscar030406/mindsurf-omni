"""Did adding audio break the text ability we started with?

The Thinker arrives already trained: 89,864,448 parameters, 2.06B tokens, a
strict validation loss of 1.7268. Every audio stage after that updates those
same weights, and nothing in the audio metrics would notice if the language
ability were quietly eroding underneath.

MiniMind-O says as much about its own vision stage -- it freezes all but the
projector precisely because image data "over-rewrites the language and speech
abilities". The audio stages do not have that protection, so the check has to
be explicit.

The baseline is a number from the pretraining phase, measured on a holdout that
this project never trains on. Comparing against it is the only way to separate
"the audio path is bad" from "the model forgot how to speak Chinese".
"""

from __future__ import annotations

from dataclasses import dataclass

from mindsurf_omni.evaluation.metrics import Measurement, assess

# Measured when the base was released, on the strict holdout.
BASELINE_STRICT_VAL = 1.7268
# Two seeds of the same recipe differed by this much, so anything smaller is
# not a regression, it is the same model twice. This belongs in the threshold,
# not in the eligibility test -- an instrument does not need to resolve the
# seed spread, it needs to resolve a regression worth acting on.
BASELINE_SEED_SPREAD = 0.0020

# The smallest degradation worth catching. The pretraining gate treated margins
# of 0.10-0.17 nats as meaningful against a seed spread of 0.0020, so 0.05 is
# comfortably inside "a real change" while still being small enough that
# damage is caught before it is obvious.
SMALLEST_REGRESSION_WORTH_CATCHING = 0.05


@dataclass(frozen=True, slots=True)
class RegressionVerdict:
    value: float
    baseline: float
    difference: float
    threshold: float
    verdict: str
    note: str = ""

    def __str__(self) -> str:
        return (
            f"strict_val {self.value:.4f} against baseline {self.baseline:.4f}: "
            f"{self.difference:+.4f} (±{self.threshold:.4f}) -> {self.verdict}"
        )


def assess_text_regression(
    losses: list[float],
    baseline: float = BASELINE_STRICT_VAL,
    seed_spread: float = BASELINE_SEED_SPREAD,
    tolerance: float = 3.0,
    effect_of_interest: float = SMALLEST_REGRESSION_WORTH_CATCHING,
) -> RegressionVerdict:
    """Compare current strict loss against the base's, with both noise sources.

    Two floors combine here and both matter. Sampling noise comes from the
    holdout being finite; seed noise comes from the baseline itself being one
    run of a recipe. Using only the first would call a difference real when a
    rerun of the same recipe would have produced it.

    The seed spread is a *lower bound*: the two seeds behind it perturbed
    initialisation only, and saw identical data order. So this errs toward
    calling things indistinguishable, which is the safer direction for a check
    whose job is to catch damage.
    """
    measurement = assess(
        "strict_val", losses, effect_of_interest=effect_of_interest, minimum_samples=30
    )
    if not measurement.gating_eligible:
        return RegressionVerdict(
            value=measurement.value,
            baseline=baseline,
            difference=measurement.value - baseline,
            threshold=float("inf"),
            verdict="reported only",
            note=measurement.note,
        )

    difference = measurement.value - baseline
    threshold = tolerance * (measurement.noise_floor**2 + seed_spread**2) ** 0.5

    if abs(difference) <= threshold:
        verdict = "unchanged"
    elif difference > 0:
        verdict = "regressed"  # loss went up: the model got worse at text
    else:
        verdict = "improved"

    return RegressionVerdict(
        value=measurement.value,
        baseline=baseline,
        difference=difference,
        threshold=threshold,
        verdict=verdict,
        note=(
            "the seed spread is a lower bound -- the two seeds behind it varied "
            "initialisation only and saw the same data order"
        ),
    )


def gating_measurement(losses: list[float]) -> Measurement:
    """The same numbers as a Measurement, for a report that lists everything."""
    return assess(
        "strict_val",
        losses,
        effect_of_interest=SMALLEST_REGRESSION_WORTH_CATCHING,
        minimum_samples=30,
    )
