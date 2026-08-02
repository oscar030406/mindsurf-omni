"""SNR is the whole claim of this script, so the arithmetic is what gets pinned.

A gain that is wrong by a factor still produces plausible-sounding noisy audio
and a curve that looks reasonable -- it just labels every point with the wrong
dB. Nothing downstream would catch it.
"""

from __future__ import annotations

import math

from scripts.add_noise import scale_for_snr


def test_zero_db_matches_the_speech_level() -> None:
    gain = scale_for_snr(speech_rms=0.5, noise_rms=0.25, snr_db=0.0)

    assert math.isclose(0.25 * gain, 0.5)


def test_twenty_db_puts_the_noise_a_tenth_of_the_amplitude_down() -> None:
    """20 dB is a factor of ten in amplitude, not in power."""
    gain = scale_for_snr(speech_rms=1.0, noise_rms=1.0, snr_db=20.0)

    assert math.isclose(gain, 0.1)


def test_higher_snr_means_less_noise() -> None:
    quiet = scale_for_snr(1.0, 1.0, 20.0)
    loud = scale_for_snr(1.0, 1.0, 0.0)

    assert quiet < loud


def test_silence_does_not_divide_by_zero() -> None:
    assert scale_for_snr(0.0, 1.0, 10.0) == 0.0
    assert scale_for_snr(1.0, 0.0, 10.0) == 0.0
