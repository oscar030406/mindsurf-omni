"""The two checks that keep an emotional voice pack from shipping broken.

Both failure modes are quiet. A pitch shift large enough to carry emotion is
also large enough to move formants until the voice is somebody else, and the
speaker vector is where the clone metric reads identity from -- a pack built
past that point would clone the wrong person while sounding cheerful. And Mimi
at 12.5 Hz is a narrow pipe, so a manipulation can survive in the waveform and
not through encode-decode, producing a pack that changes nothing at all.

Neither shows up as an error. Both show up as a number, which is why the
numbers are measured per variant rather than trusted from the manipulation
parameters.
"""

from __future__ import annotations

import math

import pytest
from scripts.build_emotion_voices import (
    EMOTIONS,
    IDENTITY_FRACTION,
    MAX_SEMITONES,
    f0_median,
    manipulate,
    semitones_for,
)

# librosa and numpy are in the tts/train extras, not the test environment; the
# arithmetic below runs where the pack is built.
pytest.importorskip("librosa")
pytest.importorskip("numpy")


def _tone(frequency: float, seconds: float = 0.5, rate: int = 24_000):
    import numpy

    t = numpy.arange(int(rate * seconds)) / rate
    # A couple of harmonics, because yin on a pure sine is not representative
    # of anything this will ever be handed.
    wave = numpy.sin(2 * math.pi * frequency * t)
    wave += 0.5 * numpy.sin(4 * math.pi * frequency * t)
    return (wave / 2).astype("float32")


def test_identity_is_judged_against_the_control_not_against_one() -> None:
    """The codec round trip costs cosine on its own -- 0.8409, measured.

    Charging a variant for the pipeline's own loss is the mistake this project
    has made twice with floors, so the requirement is a fraction of the control
    rather than an absolute.
    """
    assert 0.5 < IDENTITY_FRACTION < 1.0


def test_the_shift_is_computed_per_voice_because_hz_is_not_semitones() -> None:
    """+50 Hz is a different manipulation on a 117 Hz voice than on a 260 Hz one.

    A fixed semitone count under-moves the high voices and destroys the low
    ones; this is the conversion, and the cap is what stopped the first build
    from shipping a different speaker.
    """
    low, _ = semitones_for(117.0, 50.0)
    high, _ = semitones_for(260.0, 50.0)

    # Uncapped the low voice would need far more semitones than the high one.
    assert 12.0 * math.log2((117.0 + 50) / 117.0) > 12.0 * math.log2((260.0 + 50) / 260.0)
    # Both are held at the cap, and the cap is reported rather than silent.
    assert low == MAX_SEMITONES and high == MAX_SEMITONES
    assert semitones_for(117.0, 50.0)[1] is True

    # A rate-only arm asks for no shift and must get exactly none.
    assert semitones_for(200.0, 0.0) == (0.0, False)


def test_the_manipulations_move_pitch_in_the_directions_they_claim() -> None:
    original = _tone(200.0)

    up, _ = semitones_for(200.0, EMOTIONS["happy"]["hz"])
    down, _ = semitones_for(200.0, EMOTIONS["care"]["hz"])
    happy = manipulate(original, 24_000, up, EMOTIONS["happy"]["rate"])
    care = manipulate(original, 24_000, down, EMOTIONS["care"]["rate"])

    assert f0_median(happy, 24_000) > f0_median(original, 24_000)
    assert f0_median(care, 24_000) < f0_median(original, 24_000)


def test_the_manipulations_change_duration_in_the_directions_they_claim() -> None:
    """Rate is half the emotion; a pack that only moves pitch is half a pack."""
    original = _tone(200.0)

    happy = manipulate(original, 24_000, 0.0, EMOTIONS["happy"]["rate"])
    care = manipulate(original, 24_000, 0.0, EMOTIONS["care"]["rate"])

    assert len(happy) < len(original)
    assert len(care) > len(original)


def test_too_little_voiced_audio_reports_nan_rather_than_a_number() -> None:
    """Silence has no pitch, and a fabricated one would set the whole pack's
    shift figures against a value that means nothing."""
    import numpy

    assert math.isnan(f0_median(numpy.zeros(4_000, dtype="float32"), 24_000))
