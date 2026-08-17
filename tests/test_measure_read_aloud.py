"""The rulers this round added, on signals whose answer is known in advance."""

from __future__ import annotations

import numpy as np
from scripts.measure_read_aloud import (
    f0_track,
    gaps,
    pick_reading,
    punctuation_profile,
    slope_before_gap,
    to_input_rate,
    voiced_seconds,
)

RATE = 16_000


def _tone(hertz: float | tuple[float, float], seconds: float) -> np.ndarray:
    """A sine at a fixed pitch, or sweeping between two."""
    samples = np.arange(int(RATE * seconds))
    if isinstance(hertz, tuple):
        track = np.linspace(hertz[0], hertz[1], len(samples))
    else:
        track = np.full(len(samples), float(hertz))
    return 0.5 * np.sin(2 * np.pi * np.cumsum(track) / RATE)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds))


def test_the_leading_silence_is_not_a_pause() -> None:
    """The first version counted it, put the first pause at offset zero, and then
    had nothing in front of it to measure a pitch slope over -- every case came
    back None. Interior only, or the instrument reports nothing at all."""
    signal = np.concatenate([_silence(0.5), _tone(200, 0.4), _silence(0.3), _tone(200, 0.4)])

    found = gaps(signal, RATE)

    assert len(found) == 1
    at, length = found[0]
    assert at == 0.9
    assert 0.25 <= length <= 0.35


def test_a_gap_shorter_than_the_floor_is_not_a_pause() -> None:
    signal = np.concatenate([_tone(200, 0.4), _silence(0.05), _tone(200, 0.4)])

    assert gaps(signal, RATE) == []


def test_the_carrier_silence_is_taken_off_before_lengths_are_compared() -> None:
    """Two tokens are compared by duration, and raw file length would measure the
    silence the synthesiser puts at both ends rather than the token."""
    padded = np.concatenate([_silence(0.5), _tone(200, 0.3), _silence(0.5)])

    assert abs(voiced_seconds(padded, RATE) - 0.3) < 0.01


def test_pitch_direction_is_the_direction_it_is() -> None:
    """The misplaced question mark is read off this sign: held flat or rising is a
    question, falling is a statement. If the sign were backwards the reading
    would say the opposite of what happened."""
    rising = np.concatenate([_tone((180, 300), 0.5), _silence(0.3), _tone(200, 0.2)])
    falling = np.concatenate([_tone((300, 180), 0.5), _silence(0.3), _tone(200, 0.2)])

    assert slope_before_gap(rising, RATE)["slope_hz_per_s"] > 50
    assert slope_before_gap(falling, RATE)["slope_hz_per_s"] < -50


def test_an_unvoiced_run_up_says_so_rather_than_reporting_a_slope() -> None:
    """Breath and unvoiced consonants carry no pitch. Saying None beats fitting a
    line through whatever the autocorrelation happened to pick."""
    noise = 0.3 * np.random.default_rng(0).standard_normal(int(RATE * 0.3))
    signal = np.concatenate([noise, _silence(0.3), _tone(200, 0.2)])

    measured = slope_before_gap(signal, RATE)

    assert measured["pause_at"] is not None
    assert measured["slope_hz_per_s"] is None
    assert measured["voiced_frames"] < 5


def test_the_tracker_finds_the_pitch_it_was_given() -> None:
    track = f0_track(_tone(220, 0.5), RATE)
    voiced = track[track > 0]

    assert len(voiced) > 20
    assert abs(float(np.median(voiced)) - 220) < 10


def test_the_average_hides_the_run_that_a_reader_meets() -> None:
    """Same mark count, same length, and only one of these is unreadable."""
    spread = "一二三四五六七八，九十一二三四五六，七八九十一二三四，五六七八九十。"
    bunched = "一二，三四，五六，七八九十一二三四五六七八九十一二三四五六七八九十。"

    even, lumpy = punctuation_profile(spread), punctuation_profile(bunched)

    assert even["marks"] == lumpy["marks"]
    assert even["characters"] == lumpy["characters"]
    assert even["marks_per_100"] == lumpy["marks_per_100"]
    assert even["runs_over_20"] == 0
    assert lumpy["runs_over_20"] == 1
    assert lumpy["longest_run"] > even["longest_run"]


def _candidate(spelling: str, carried: float) -> dict:
    return {"spelling": spelling, "bare_seconds": carried - 0.7, "carried_seconds": carried}


def test_a_spelling_that_matches_is_named() -> None:
    picked = pick_reading(
        [
            _candidate("B203", 1.52),
            _candidate("B二零三", 1.522),
            _candidate("B两百零三", 1.772),
        ]
    )

    assert picked["read_as"] == "B二零三"


def test_none_of_them_matching_says_none_rather_than_the_nearest() -> None:
    """demo read in English has no Chinese candidate anywhere near it. Reporting
    the closest anyway would put an artefact next to a real answer in the same
    field, with nothing to tell them apart."""
    picked = pick_reading(
        [
            _candidate("demo", 1.18),
            _candidate("地谋", 0.985),
            _candidate("demo演示", 1.615),
        ]
    )

    assert picked["read_as"] is None
    assert picked["nearest"] == "地谋"
    assert picked["nearest_delta"] == -0.195


def test_resampling_keeps_the_length_in_seconds() -> None:
    """24k out of the synthesiser, 16k into the recogniser; a wrong ratio here
    would stretch the audio and every duration in the report with it."""
    at_24k = np.zeros(24_000)  # one second at the synthesiser's rate

    assert len(to_input_rate(at_24k, 24_000)) == 16_000
    assert to_input_rate(_tone(200, 0.5), RATE).shape == (8_000,)
