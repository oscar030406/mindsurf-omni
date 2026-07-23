"""The conversion check's teeth, pinned.

The tolerance was widened once, after a faithful conversion was rejected for a
2.4e-05 logit gap that turned out to be float32 accumulating differently across
torch versions. Widening a check because it fired is how a check stops being a
check, so these tests hold the widened version to catching what it was built to
catch.

The magnitudes here come from injecting each fault into the real released
weights and measuring; see docs/CONVERSION.md.
"""

from __future__ import annotations

import pytest

EPS_MULTIPLE = 16.0
MAX_SPREAD = 5.0
# float32 epsilon at this model's logit scale (max |logit| ~ 35.5).
EPS_AT_SCALE = 4.233854e-06


def _verdict(max_absolute: float, spread_ratio: float, argmax_agrees: bool) -> list[str]:
    """Reproduce the script's decision, so the thresholds are tested not restated."""
    reasons = []
    if max_absolute > EPS_AT_SCALE * EPS_MULTIPLE:
        reasons.append("magnitude")
    if spread_ratio > MAX_SPREAD:
        reasons.append("spread")
    if not argmax_agrees:
        reasons.append("argmax")
    return reasons


def test_a_faithful_conversion_is_accepted() -> None:
    """Measured on the released weights under torch 2.6 against a torch 2.11 fixture."""
    assert _verdict(max_absolute=2.384e-05, spread_ratio=1.6, argmax_agrees=True) == []


@pytest.mark.parametrize(
    "fault,max_absolute,spread_ratio,argmax_agrees",
    [
        # A single weight moved by 0.01 -- the smallest fault worth catching,
        # and the one a loose tolerance would let through.
        ("one_weight_nudged", 3.285e-04, 3.5, True),
        # gate_proj takes the silu and up_proj does not, so swapping them
        # leaves a model that still runs.
        ("swap_gate_up", 1.753e01, 2.5, False),
        ("swap_q_k_norm", 1.032e00, 3.7, True),
        ("swap_layernorms", 1.235e01, 5.1, False),
    ],
)
def test_every_injected_fault_is_rejected(
    fault: str, max_absolute: float, spread_ratio: float, argmax_agrees: bool
) -> None:
    assert _verdict(max_absolute, spread_ratio, argmax_agrees), f"{fault} slipped through"


def test_the_tolerance_leaves_a_real_margin() -> None:
    """The gap between noise and the quietest fault is what makes the bound safe."""
    noise = 2.384e-05
    quietest_fault = 3.285e-04
    allowed = EPS_AT_SCALE * EPS_MULTIPLE

    assert noise < allowed < quietest_fault
    # An order of magnitude of headroom on each side, so neither a slightly
    # noisier host nor a slightly subtler fault flips the verdict.
    assert allowed / noise > 2.5
    assert quietest_fault / allowed > 4.0


def test_magnitude_alone_would_have_sufficed_here() -> None:
    """Stated plainly rather than implied: spread caught nothing magnitude missed.

    It is kept because it is the check that distinguishes *why* a conversion
    failed, and because these four faults are not the only ones possible -- but
    claiming it as load-bearing would overstate what was measured.
    """
    faults = [(3.285e-04, 3.5, True), (1.753e01, 2.5, False), (1.032e00, 3.7, True)]

    assert all("magnitude" in _verdict(*fault) for fault in faults)
