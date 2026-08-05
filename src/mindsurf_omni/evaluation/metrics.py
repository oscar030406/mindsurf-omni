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
from typing import Any


def fold_numerals(text: str) -> str:
    """Write Chinese number words as digits, so 二零二一 and 2021 compare equal.

    A synthesiser that reads 2021年9月23日 as 二零二一年九月二十三日 is doing
    exactly what it should; the judge transcribes what it hears; and the metric
    then charges a substitution for every digit. On the fixed 160 texts, 71
    contain digits and the split is the whole floor: 0.0050 without them,
    0.0746 with. Same class of correction as the traditional-to-simplified
    fold above -- same word, different notation -- and applied to both sides
    for the same reason.

    Ordinary words get caught too (一般 becomes 1般), which is harmless because
    both sides pass through it. What it cannot do is context: a handful of
    sentences fold asymmetrically and score worse, 3 of 160 on each arm.

    Off by default. The acceptance thresholds were calibrated on the unfolded
    metric, and moving the ruler after the run is not allowed here.
    """
    import cn2an

    try:
        return str(cn2an.transform(text, "cn2an"))
    except Exception:
        # Its parser raises on inputs it cannot segment. An unfolded sample is
        # a sample scored the old way, which is a worse number, not a wrong
        # one -- the alternative is one bad row killing a 160-sample run.
        return text


def normalise_for_cer(text: str, fold_numbers: bool = False) -> str:
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
    if fold_numbers:
        text = fold_numerals(text)
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


def character_error_rate(reference: str, hypothesis: str, fold_numbers: bool = False) -> float:
    """Errors per reference character.

    Can exceed 1.0: a model that says far more than it was asked to is worse
    than one that says nothing, and clamping would hide exactly that failure.
    """
    reference = normalise_for_cer(reference, fold_numbers)
    hypothesis = normalise_for_cer(hypothesis, fold_numbers)
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


def cluster_bootstrap_noise_floor(
    groups: list[list[float]], resamples: int = 1000, seed: int = 0, confidence: float = 0.95
) -> float:
    """Half-width for a mean whose samples are not independent, by resampling groups.

    Resampling individual rows assumes each row carries its own information.
    When rows cluster -- twenty clips of one voice that are almost all correct
    or almost all wrong together -- that assumption inflates the effective
    sample size and shrinks the floor to a number the data does not support.
    The unit that varies is the voice, so the voice is what gets resampled.

    This is the difference between "240 clips" and "12 voices", and on the
    identification measurement it is an order of magnitude: a per-row floor of
    +/-0.1750 was already too wide to gate, and the honest one is wider still.
    """
    populated = [group for group in groups if group]
    if len(populated) < 2:
        return float("inf")
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        drawn = [populated[rng.randrange(len(populated))] for _ in range(len(populated))]
        rows = [value for group in drawn for value in group]
        means.append(sum(rows) / len(rows))
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


def _paired_verdict(
    name: str,
    difference: float,
    threshold: float,
    tail: str,
    lower_is_better: bool,
    effect_of_interest: float | None,
) -> str:
    """Verdict plus, when an effect of interest is declared, its licence to gate.

    Statistical resolution and practical relevance are different questions, and
    a comparison that answers only the first decides guardrails by how sharp the
    instrument happens to be. The two failures are opposite and both have
    happened here:

    A blunt instrument certifies. Anything inside a wide band reads
    indistinguishable, so a guardrail worded as "did not regress" passes because
    nothing could have been seen -- rank-1 identification resolves 0.70 where the
    effect of interest is 0.10, and passed a guardrail on that basis.

    A sharp instrument convicts. Once the floor is small enough, a difference
    nobody argued was meaningful clears it and reads regressed -- chat_nll failed
    the length arm at 0.0479 nat while measure_chat_loss had declared 0.05 the
    threshold for two comparable runs, printed it, and never applied it.

    So the demotion is one-sided in each branch. A null result is worthless when
    the instrument cannot resolve the effect of interest. A non-null result is
    still real when the instrument is blunt -- it saw something anyway, and
    anything it can see exceeds what it cannot -- but it may not gate when the
    difference is below what was declared worth acting on.
    """
    if abs(difference) <= threshold:
        verdict = f"{name}: indistinguishable ({difference:+.4f}, within ±{threshold:.4f}, {tail})"
        if effect_of_interest is not None and threshold > effect_of_interest:
            return (
                f"{verdict} —— 仅报告：只分辨得了 {threshold:.4f}，"
                f"而关心的是 {effect_of_interest:.4f}，读不出这么小的劣化"
            )
        return verdict

    improved = difference < 0 if lower_is_better else difference > 0
    word = "improved" if improved else "regressed"
    verdict = f"{name}: {word} ({difference:+.4f} against ±{threshold:.4f}, {tail})"
    if effect_of_interest is not None and abs(difference) < effect_of_interest:
        return f"{verdict} —— 仅报告：低于关心的效应 {effect_of_interest:.4f}"
    return verdict


def compare_paired(
    name: str,
    deltas: list[float],
    lower_is_better: bool = True,
    seed: int = 0,
    effect_of_interest: float | None = None,
) -> str:
    """The same three verdicts, from per-sample differences on shared items.

    Pairing is what the fixed-text protocol buys. Unpaired comparison combines
    two whole-set noise floors, most of which is per-item difficulty that both
    systems share; on the same items that difficulty cancels in the subtraction
    and the floor shrinks to what actually differs between the systems. The
    caller is responsible for only pairing rows whose reference text matches --
    a pair over different texts subtracts two unrelated numbers.

    ``effect_of_interest`` is the difference worth acting on. Leaving it unset
    keeps the older behaviour, where the noise floor alone decides; see
    ``_paired_verdict`` for what setting it buys and which failure it caught.
    """
    if len(deltas) < 2:
        return f"{name}: reported only (only {len(deltas)} pairs)"

    difference = sum(deltas) / len(deltas)
    threshold = resolvable_effect(bootstrap_noise_floor(deltas, seed=seed))
    return _paired_verdict(
        name,
        difference,
        threshold,
        f"paired n={len(deltas)}",
        lower_is_better,
        effect_of_interest,
    )


def compare_paired_clustered(
    name: str,
    groups: dict[str, list[float]] | list[list[float]],
    lower_is_better: bool = True,
    seed: int = 0,
    effect_of_interest: float | None = None,
) -> str:
    """``compare_paired`` where the pairs are not independent of each other.

    Same three verdicts, same threshold rule, but the floor comes from
    resampling groups rather than rows. Use it whenever one label produces many
    rows that move together -- twenty clips of a voice, several turns of one
    conversation -- because the per-row floor there is an artefact of counting
    correlated rows as independent evidence.

    This is not a stricter version of ``compare_paired`` for the cautious. It
    is the correct one for clustered data, and the difference is large enough
    to change verdicts: the voice-identification margin reads ``improved``
    against a per-row threshold and ``indistinguishable`` against this one.
    """
    values = list(groups.values()) if isinstance(groups, dict) else groups
    populated = [group for group in values if group]
    rows = [value for group in populated for value in group]
    if len(populated) < 2:
        return f"{name}: reported only (only {len(populated)} clusters)"

    difference = sum(rows) / len(rows)
    threshold = resolvable_effect(cluster_bootstrap_noise_floor(populated, seed=seed))
    tail = f"paired n={len(rows)} in {len(populated)} clusters"
    return _paired_verdict(name, difference, threshold, tail, lower_is_better, effect_of_interest)


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


def cross_reference_agreement(
    first: list[float], second: list[float], confidence: float = 1.96
) -> dict[str, Any]:
    """Do two reference sets rank the same probes the same way?

    `chat_nll` scores how likely a checkpoint finds one author's replies, and
    this project runs it against two neutral authors so that neither one's style
    decides the verdict. That guard only works if the two agree about individual
    probes -- and nobody checked until an intervention came along that made them
    disagree outright.

    The length-tuned DPO arm reads `regressed` on one author and `improved` on
    the other, and per probe the two deltas agree in sign only 20% of the time,
    which is worse than a coin. The mechanism is that the metric is sensitive to
    the reference's own length (r = +0.62 within one author), so any change to
    how long the model's replies are moves the score in whichever direction that
    particular reference happens to sit. Even the round where the two authors
    agreed on the aggregate agreed on only 59% of probes.

    So this is the eligibility check the metric was missing: agreement must beat
    chance by more than sampling noise before either author's verdict may gate.
    Below that, both numbers are reported and neither decides -- the same rule
    `assess` applies to resolution, one axis over.
    """
    paired = list(zip(first, second, strict=True))
    if len(paired) < 2:
        return {"n": len(paired), "agreement": None, "gating_eligible": False}

    agree = sum(1 for a, b in paired if (a > 0) == (b > 0))
    share = agree / len(paired)
    noise = confidence * math.sqrt(0.25 / len(paired))
    eligible = share - 0.5 > noise
    return {
        "n": len(paired),
        "agreement": share,
        "noise": noise,
        "gating_eligible": eligible,
        "note": (
            "the two reference sets rank probes alike often enough to gate"
            if eligible
            else "the two reference sets do not agree per probe beyond chance; "
            "report both, let neither decide"
        ),
    }
