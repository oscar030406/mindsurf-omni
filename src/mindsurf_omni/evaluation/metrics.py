"""Speech metrics, and the arithmetic that decides whether they may judge.

The pretraining phase of this project shipped three instruments that measured
something other than what they claimed: a repetition score that gave a perfect
mark to a seven-character reply, because stopping early meant nothing to
repeat; a quality filter that dropped 6% of English and 0% of Chinese, because
"character diversity" is a writing-system proxy; a multiple-choice set of 192
questions whose resolution was +/-7pp against effects of 3-5pp.

All three looked fine until someone computed what they could actually resolve.
So the rule here is that a metric may be reported freely and may only *judge*
after its noise floor is measured and shown to be smaller than the effect being
claimed.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass


def normalise_for_cer(text: str) -> str:
    """Strip what a listener would not hear before comparing transcripts.

    Punctuation and case survive into ASR output inconsistently, so leaving
    them in measures the recogniser's punctuation habits rather than the
    speech. Whitespace goes too: Chinese output is unspaced and English is not,
    and keeping it would make the two scripts incomparable.

    Traditional characters are folded to simplified for the same reason, and it
    is not a small correction. Whisper picks a script per utterance with no
    regard for the input, so 地铁 comes back 地鐵 and every character of it
    counts as a substitution. Measured over the hundred-probe floor run: mean
    CER 0.1788 before folding, 0.0388 after, exact matches 46 of 100 to 80 of
    100. Four fifths of what looked like synthesis error was the judge's choice
    of script.

    Both sides are folded, not just the hypothesis. What is being measured is
    whether the audio said the words; which script someone wrote them in is a
    property of the text, visible in the text, and not something a measurement
    of speech should be charging to the speaker.

    ``zh-hans`` and not ``zh-cn``: the latter also swaps regional vocabulary,
    turning 軟體 into 软件 rather than 软体, and that would forgive a genuinely
    different word rather than a differently written one. On this data the two
    agree to 0.0018, well inside the noise floor -- so the strict fold costs
    nothing measurable and keeps the metric to what it claims to measure.
    """
    from zhconv import convert

    text = unicodedata.normalize("NFKC", text).lower()
    stripped = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith(("P", "Z", "C"))
    )
    return str(convert(stripped, "zh-hans"))


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance, two rows at a time.

    Full-matrix would be clearer but allocates len(a)*len(b); transcripts here
    run to thousands of characters across hundreds of samples.
    """
    if not reference:
        return len(hypothesis)
    previous = list(range(len(reference) + 1))
    for j, target in enumerate(hypothesis, start=1):
        current = [j]
        for i, source in enumerate(reference, start=1):
            current.append(
                previous[i - 1]
                if source == target
                else 1 + min(previous[i - 1], previous[i], current[i - 1])
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Errors per reference character.

    Can exceed 1.0: a model that says far more than it was asked to is worse
    than one that says nothing, and clamping would hide exactly that failure.
    """
    reference = normalise_for_cer(reference)
    hypothesis = normalise_for_cer(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    return edit_distance(reference, hypothesis) / len(reference)


@dataclass(frozen=True, slots=True)
class Measurement:
    """A number with the uncertainty that decides what it can say."""

    name: str
    value: float
    noise_floor: float
    sample_size: int
    gating_eligible: bool
    note: str = ""

    def __str__(self) -> str:
        return f"{self.name} {self.value:.4f} ± {self.noise_floor:.4f} (n={self.sample_size})"


def bootstrap_noise_floor(
    values: list[float], resamples: int = 1000, seed: int = 0, confidence: float = 0.95
) -> float:
    """Half-width of the confidence interval for the mean, by resampling.

    This is sampling noise only -- how much the number would move on a
    different draw of the same size. It says nothing about run-to-run
    variation, which needs a second training run and is reported separately.
    """
    if len(values) < 2:
        return float("inf")
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    tail = (1.0 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(int((1 - tail) * resamples), resamples - 1)]
    return (high - low) / 2


def resolvable_effect(noise_floor: float, tolerance_multiple: float = 3.0) -> float:
    """The smallest difference this instrument may claim a direction for."""
    return noise_floor * tolerance_multiple


def assess(
    name: str,
    values: list[float],
    effect_of_interest: float,
    minimum_samples: int = 100,
    seed: int = 0,
) -> Measurement:
    """Measure, and decide whether the result is allowed to judge.

    ``effect_of_interest`` is the difference the experiment is trying to
    detect. An instrument that cannot resolve it is not a weak instrument, it
    is the wrong instrument -- and reporting its direction anyway is reporting
    noise.
    """
    if not values:
        return Measurement(name, float("nan"), float("inf"), 0, False, "no samples")

    value = sum(values) / len(values)
    noise = bootstrap_noise_floor(values, seed=seed)
    resolvable = resolvable_effect(noise)

    reasons = []
    if len(values) < minimum_samples:
        reasons.append(f"n={len(values)} below the {minimum_samples} required")
    if not math.isfinite(noise):
        reasons.append("noise floor could not be estimated")
    elif resolvable > effect_of_interest:
        reasons.append(
            f"resolves {resolvable:.4f} at best, but the effect of interest is "
            f"{effect_of_interest:.4f}"
        )

    return Measurement(
        name=name,
        value=value,
        noise_floor=noise,
        sample_size=len(values),
        gating_eligible=not reasons,
        note="; ".join(reasons),
    )


def compare(
    name: str, candidate: Measurement, reference: Measurement, lower_is_better: bool = True
) -> str:
    """Improved, regressed, or indistinguishable -- never a fourth answer.

    A difference inside the combined noise has no direction, and saying it does
    is the most common way a project talks itself into a result.
    """
    if not candidate.gating_eligible:
        return f"{name}: reported only ({candidate.note})"

    difference = candidate.value - reference.value
    combined = math.hypot(candidate.noise_floor, reference.noise_floor)
    threshold = resolvable_effect(combined)

    if abs(difference) <= threshold:
        return f"{name}: indistinguishable ({difference:+.4f}, within ±{threshold:.4f})"
    improved = difference < 0 if lower_is_better else difference > 0
    verdict = "improved" if improved else "regressed"
    return f"{name}: {verdict} ({difference:+.4f} against ±{threshold:.4f})"
