"""A zero clipping ratio means two different things and only one is good news.

Archived audio that was normalised before it was written has had every pinned
sample scaled off the rail, so the detector reads zero on a clip that really
did clip. The guard against reading that zero as "clean" is the peak
distribution, and this is what keeps that guard honest: it has to fire on
normalised audio and stay quiet on audio that is merely not loud.
"""

from __future__ import annotations

from scripts.measure_clipping import normalised_already


def arm(p50: float, p100: float) -> dict[str, dict[str, float]]:
    return {"peak": {"p50": p50, "p100": p100}}


def test_uniform_peaks_at_the_normaliser_target_are_flagged() -> None:
    assert normalised_already(arm(31_129, 31_129))


def test_ordinary_audio_with_a_spread_of_peaks_is_not_flagged() -> None:
    """The archived model arms look like this -- peaks scattered well below the
    rail. Flagging them would put a warning on every honest measurement."""
    assert not normalised_already(arm(22_863, 32_642))


def test_quiet_audio_is_not_mistaken_for_normalised_audio() -> None:
    """Both have no sample near the rail; only one has had evidence removed."""
    assert not normalised_already(arm(5_000, 9_000))


def test_an_arm_with_no_peaks_measured_is_not_claimed_either_way() -> None:
    assert not normalised_already({"peak": {}})
