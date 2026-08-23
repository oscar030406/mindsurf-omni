"""Audio conversions at the boundary, where a quiet mistake sounds like a bad model."""

from __future__ import annotations

import array
import struct

from mindsurf_omni.service.audio import (
    clipping_ratio,
    peak_normalise,
    resample,
    to_wav,
    trim_silence,
    wav_header,
)


def pcm(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


def samples_of(data: bytes) -> list[int]:
    out = array.array("h")
    out.frombytes(data)
    return list(out)


def test_resampling_changes_length_by_the_rate_ratio() -> None:
    """Wrong length means wrong playback speed, which sounds like a broken model."""
    source = pcm([1000] * 1600)  # 100 ms at 16 kHz

    upsampled = resample(source, 16_000, 24_000)

    assert len(samples_of(upsampled)) == 2400  # 100 ms at 24 kHz


def test_resampling_to_the_same_rate_is_a_no_op() -> None:
    source = pcm([1, 2, 3, 4])

    assert resample(source, 16_000, 16_000) == source


def test_resampling_preserves_a_constant_signal() -> None:
    """Interpolation between equal neighbours must not drift."""
    source = pcm([5000] * 800)

    result = samples_of(resample(source, 24_000, 16_000))

    assert all(abs(sample - 5000) <= 1 for sample in result)


def test_empty_and_odd_length_input_do_not_raise() -> None:
    assert resample(b"", 16_000, 24_000) == b""
    assert resample(b"\x01", 16_000, 24_000) == b""


def test_a_streaming_header_does_not_declare_zero_length() -> None:
    """A zero-length header makes most players report an empty file and stop."""
    header = wav_header(24_000, data_bytes=None)

    declared = struct.unpack("<I", header[40:44])[0]

    assert len(header) == 44
    assert declared > 0


def test_a_complete_clip_declares_its_real_length() -> None:
    audio = pcm([100] * 480)

    container = to_wav(audio, 24_000)

    assert struct.unpack("<I", container[40:44])[0] == len(audio)
    assert container[:4] == b"RIFF"
    assert container[8:12] == b"WAVE"
    assert container[44:] == audio


def test_the_header_records_the_rate_the_caller_asked_for() -> None:
    """A client that plays 24 kHz audio at 16 kHz hears a slowed voice."""
    assert struct.unpack("<I", wav_header(24_000)[24:28])[0] == 24_000
    assert struct.unpack("<I", wav_header(16_000)[24:28])[0] == 16_000


def test_loud_audio_is_brought_down_to_the_target_peak() -> None:
    loud = pcm([32000, -32000, 16000])

    result = samples_of(peak_normalise(loud, target_peak=0.5))

    assert max(abs(sample) for sample in result) <= int(0.5 * 32768) + 1


def test_quiet_audio_is_left_alone_rather_than_amplified() -> None:
    """Amplifying a near-silent clip amplifies its noise floor into hiss."""
    faint = pcm([50, -40, 30])

    assert peak_normalise(faint) == faint


def test_normalising_silence_does_not_divide_by_zero() -> None:
    silence = pcm([0] * 100)

    assert peak_normalise(silence) == silence


def test_leading_and_trailing_dead_air_is_trimmed() -> None:
    """Concatenated across a streamed reply, dead air becomes an audible stutter."""
    clip = pcm([0] * 2400 + [8000] * 2400 + [0] * 2400)

    trimmed = samples_of(trim_silence(clip, keep_ms=0, rate=24_000))

    assert len(trimmed) == 2400
    assert all(sample == 8000 for sample in trimmed)


def test_a_margin_is_kept_so_the_first_syllable_is_not_clipped() -> None:
    clip = pcm([0] * 2400 + [8000] * 2400 + [0] * 2400)

    trimmed = samples_of(trim_silence(clip, keep_ms=50, rate=24_000))

    assert len(trimmed) > 2400  # 50 ms either side survives
    assert len(trimmed) <= 2400 + 2 * 1200 + 1


def test_an_entirely_silent_clip_trims_to_nothing_not_to_itself() -> None:
    """Returning it unchanged would play dead air the model never meant to emit."""
    assert trim_silence(pcm([0] * 4800)) == b""


def test_the_negative_rail_counts_as_clipped_too() -> None:
    """int16 runs to -32768 and only +32767, so a symmetric test has to reach
    the deeper side; missing it would report a fully rectified fault as clean."""
    ratio, longest = clipping_ratio(pcm([-32768] * 4 + [0] * 6))

    assert ratio == 0.4
    assert longest == 4


def test_the_longest_run_is_reported_not_the_last_one() -> None:
    """A single long square edge is the audible pop; scattered single samples
    at the rail are ordinary plosives. The two must not read the same."""
    scattered = pcm([32767, 0, 32767, 0, 32767, 0])
    edge = pcm([32767, 32767, 32767, 0, 0, 0])

    assert clipping_ratio(scattered) == (0.5, 1)
    assert clipping_ratio(edge) == (0.5, 3)


def test_normalising_first_hides_the_clipping_this_is_meant_to_find() -> None:
    """The reason the measurement has to run before peak_normalise: the scale
    factor moves every sample off the rail while leaving the square edge."""
    clipped = pcm([32767] * 8 + [100] * 8)

    assert clipping_ratio(clipped)[0] > 0
    assert clipping_ratio(peak_normalise(clipped))[0] == 0.0


def test_ordinary_speech_levels_are_not_flagged() -> None:
    assert clipping_ratio(pcm([12000, -9000, 3000, 0])) == (0.0, 0)


def test_an_empty_clip_reports_nothing_rather_than_dividing_by_zero() -> None:
    assert clipping_ratio(b"") == (0.0, 0)


def test_audio_with_no_silence_survives_intact() -> None:
    clip = pcm([8000] * 1200)

    assert trim_silence(clip, keep_ms=0) == clip


def test_a_clip_is_split_into_frames_a_transport_will_carry() -> None:
    """A whole clause of speech exceeds the common 1 MB WebSocket frame limit.

    The peer's answer to an oversized frame is to close the connection with
    1009 and no error event, so the turn stops and reads as the model having
    gone quiet rather than as a transport fault.
    """
    from mindsurf_omni.service.audio import MAX_FRAME_BYTES, frames

    clip = b"\x00\x01" * 200_000  # 400 kB, about 8 seconds at 24 kHz

    pieces = frames(clip)

    assert b"".join(pieces) == clip
    assert all(len(piece) <= MAX_FRAME_BYTES for piece in pieces)


def test_frames_never_split_a_sample_in_half() -> None:
    """An odd offset shifts every following byte and turns the rest into noise."""
    from mindsurf_omni.service.audio import frames

    pieces = frames(b"\x00\x01" * 10, limit=7)

    assert all(len(piece) % 2 == 0 for piece in pieces)
    assert b"".join(pieces) == b"\x00\x01" * 10


def test_no_audio_produces_no_frames() -> None:
    """An empty frame would be a delta event carrying nothing."""
    from mindsurf_omni.service.audio import frames

    assert frames(b"") == []


def test_an_odd_number_of_bytes_is_trimmed_not_refused() -> None:
    """A recorder flushed mid-sample sends one byte too many, and numpy refuses
    the buffer outright -- served as a 500 from /v1/audio/transcriptions."""
    from mindsurf_omni.service.audio import whole_samples

    assert whole_samples(b"\x01\x02\x03") == b"\x01\x02"
    assert whole_samples(b"\x01") == b""
    assert whole_samples(b"") == b""
    assert whole_samples(b"\x01\x02") == b"\x01\x02"


def test_a_whole_wav_body_is_read_rather_than_guessed_at() -> None:
    """Posting a container instead of raw samples has always worked by accident:
    the 44-byte header decodes to about twenty samples of noise and the
    recogniser reads past it. The header is also the one place the real rate is
    written down."""
    from mindsurf_omni.service.audio import to_wav, unwrap_wav

    samples = bytes(range(0, 200, 2)) * 20
    for rate in (16_000, 24_000, 48_000):
        body, heard = unwrap_wav(to_wav(samples, rate), 16_000)
        assert (body, heard) == (samples, rate)


def test_raw_samples_pass_through_untouched() -> None:
    from mindsurf_omni.service.audio import unwrap_wav

    raw = bytes(range(0, 200, 2)) * 20
    assert unwrap_wav(raw, 16_000) == (raw, 16_000)
    assert unwrap_wav(b"", 16_000) == (b"", 16_000)
    assert unwrap_wav(b"RIFF", 16_000) == (b"RIFF", 16_000)


def test_a_container_whose_samples_cannot_be_read_is_refused_not_guessed_at() -> None:
    """This used to hand the whole container on as if it were raw PCM.

    The recogniser does not fail on nonsense, it writes something. Scored over
    12 clips against what each gives when read properly: 16 kHz stereo mean CER
    0.029, 8-bit 0.852, 24-bit empty on all twelve -- every one of them HTTP
    200. The header states exactly what the file is, so the refusal can name it;
    wrong words returned confidently cost more than a 400 naming the conversion.
    """
    import struct

    import pytest

    from mindsurf_omni.service.audio import UnsupportedAudio, unwrap_wav

    def container(channels: int, bits: int) -> bytes:
        width = channels * bits // 8
        return b"".join(
            [
                b"RIFF",
                struct.pack("<I", 236),
                b"WAVEfmt ",
                struct.pack(
                    "<IHHIIHH", 16, 1, channels, 16_000, 16_000 * width, width, bits
                ),
                b"data",
                struct.pack("<I", 200),
                bytes(200),
            ]
        )

    with pytest.raises(UnsupportedAudio, match="2-channel"):
        unwrap_wav(container(channels=2, bits=16), 16_000)
    with pytest.raises(UnsupportedAudio, match="8-bit"):
        unwrap_wav(container(channels=1, bits=8), 16_000)

    # A body that is not a container at all is still taken as raw samples --
    # that is the endpoint's documented input, not an accident.
    assert unwrap_wav(b"\x00\x01" * 100, 16_000) == (b"\x00\x01" * 100, 16_000)
