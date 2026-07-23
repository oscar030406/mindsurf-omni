"""Endpoint detection, including the ways it can end a turn wrongly."""

from __future__ import annotations

import math
import struct

from mindsurf_omni.service.vad import FRAME_MS, EndpointDetector, frame_energy

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000


def tone(frames: int, amplitude: float = 0.3) -> bytes:
    """A sine, which is what speech looks like to an energy detector."""
    samples = []
    for index in range(frames * SAMPLES_PER_FRAME):
        value = amplitude * math.sin(2 * math.pi * 220 * index / SAMPLE_RATE)
        samples.append(int(value * 32767))
    return struct.pack(f"<{len(samples)}h", *samples)


def quiet(frames: int, amplitude: float = 0.0005) -> bytes:
    samples = []
    for index in range(frames * SAMPLES_PER_FRAME):
        samples.append(int(amplitude * 32767 * ((index % 7) - 3) / 3))
    return struct.pack(f"<{len(samples)}h", *samples)


def test_energy_separates_speech_from_room_tone() -> None:
    assert frame_energy(tone(1)) > frame_energy(quiet(1)) * 20


def test_energy_of_an_empty_frame_is_zero_not_an_error() -> None:
    assert frame_energy(b"") == 0.0
    assert frame_energy(b"\x00") == 0.0  # odd byte count, truncated cleanly


def test_a_turn_ends_after_the_configured_silence() -> None:
    detector = EndpointDetector(silence_ms=300)

    assert detector.push_stream(tone(30)) is None  # still talking
    ended = detector.push_stream(quiet(20))  # 400 ms of quiet

    assert ended is not None


def test_silence_alone_never_ends_a_turn_that_never_began() -> None:
    """Otherwise an empty room ends a turn and the model answers nothing."""
    detector = EndpointDetector(silence_ms=300)

    assert detector.push_stream(quiet(200)) is None
    assert detector.speech_started is False


def test_a_pause_shorter_than_the_threshold_does_not_cut_the_speaker_off() -> None:
    """Mid-sentence breath must not be read as the end of the turn."""
    detector = EndpointDetector(silence_ms=300)

    detector.push_stream(tone(20))
    assert detector.push_stream(quiet(10)) is None  # 200 ms pause
    assert detector.push_stream(tone(20)) is None


def test_a_click_too_short_to_be_speech_does_not_arm_the_detector() -> None:
    """A door closing should not start and then end a turn."""
    detector = EndpointDetector(silence_ms=300, min_speech_ms=200)

    detector.push_stream(tone(5))  # 100 ms, below the minimum

    assert detector.speech_started is False
    assert detector.push_stream(quiet(30)) is None


def test_the_floor_adapts_so_a_quiet_microphone_is_not_read_as_silence() -> None:
    """A fixed absolute threshold makes the detector deaf on quiet hardware."""
    detector = EndpointDetector(initial_noise_floor=0.02)
    detector.push_stream(quiet(100, amplitude=0.0002))

    # Speech well below the initial threshold is still detected once the floor
    # has settled to the room.
    detector.push_stream(tone(20, amplitude=0.01))

    assert detector.speech_started is True


def test_loud_speech_does_not_drag_the_floor_up_and_deafen_the_detector() -> None:
    """If the floor tracked during speech, everything after it would be silence."""
    detector = EndpointDetector()
    before = detector._noise_floor

    detector.push_stream(tone(50, amplitude=0.8))

    assert detector._noise_floor == before


def test_reset_clears_the_turn_but_the_object_stays_usable() -> None:
    detector = EndpointDetector(silence_ms=300)
    detector.push_stream(tone(20))
    detector.reset()

    assert detector.speech_started is False
    assert detector.push_stream(quiet(30)) is None
    detector.push_stream(tone(20))
    assert detector.speech_started is True


def test_the_endpoint_offset_points_past_the_frame_that_ended_the_turn() -> None:
    """The caller slices at this offset; an off-by-one would clip or duplicate."""
    detector = EndpointDetector(silence_ms=100)
    speech = tone(20)
    detector.push_stream(speech)

    offset = detector.push_stream(quiet(20))

    assert offset is not None
    assert offset % detector.frame_bytes == 0
    assert offset <= len(quiet(20))
