"""Audio format handling at the service boundary.

SenseVoice takes 16 kHz and Mimi emits 24 kHz, so every turn crosses a rate
boundary at least once. Doing that conversion in the client would mean every
client re-implements it, and one of them would get it wrong quietly -- audio
played at the wrong rate is not silent, it is a chipmunk, which people report
as "the model sounds strange" rather than as a bug.

WAV headers are written by hand rather than through a library. The format is
forty-four bytes, the standard library's ``wave`` module wants a file object
and buffers the whole clip, and streaming means emitting a header before the
length is known.
"""

from __future__ import annotations

import array
import struct

PCM16_MAX = 32768.0


def resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear interpolation between rates.

    Not a windowed sinc: the conversions here are 16k<->24k on speech that a
    recogniser or a person will consume, and the aliasing a linear resampler
    leaves at that ratio is far below what the codec already introduces.
    Reach for a proper filter if this ever touches music.
    """
    if source_rate == target_rate or not pcm:
        return pcm

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return b""

    ratio = source_rate / target_rate
    count = max(1, int(len(samples) / ratio))
    output = array.array("h", [0]) * count
    for index in range(count):
        position = index * ratio
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        weight = position - left
        output[index] = int(samples[left] * (1 - weight) + samples[right] * weight)
    return output.tobytes()


def wav_header(sample_rate: int, channels: int = 1, data_bytes: int | None = None) -> bytes:
    """A 44-byte PCM header.

    ``data_bytes=None`` writes the maximum length, which is what a streaming
    response needs: the header goes out before the audio is generated, and
    players read until the connection closes. Writing zero instead makes most
    players report an empty file and stop.
    """
    size = 0xFFFFFFFF - 36 if data_bytes is None else data_bytes
    byte_rate = sample_rate * channels * 2
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", min(size + 36, 0xFFFFFFFF)),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, channels * 2, 16),
            b"data",
            struct.pack("<I", size),
        ]
    )


def to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    return wav_header(sample_rate, channels, len(pcm)) + pcm


# A frame the transport will actually carry. WebSocket implementations cap
# frame size -- the common default is 1 MB -- and base64 adds a third on top,
# so a whole clause of 24 kHz audio can exceed it. When it does the peer closes
# the connection with 1009 and no error event: the turn simply stops, which
# reads as the model having gone quiet rather than as a transport fault.
# 48000 bytes is one second at OUTPUT_SAMPLE_RATE, about 64 kB encoded.
MAX_FRAME_BYTES = 48_000


def frames(pcm: bytes, limit: int = MAX_FRAME_BYTES) -> list[bytes]:
    """Split PCM into pieces small enough to send, on sample boundaries.

    Splitting mid-sample would shift every following byte by one and turn the
    rest of the clip into noise, so the limit is rounded down to an even
    number rather than trusted to be one.
    """
    if limit < 2:
        raise ValueError("a frame must hold at least one 16-bit sample")
    limit -= limit % 2
    if not pcm:
        return []
    return [pcm[start : start + limit] for start in range(0, len(pcm), limit)]


def peak_normalise(pcm: bytes, target_peak: float = 0.95) -> bytes:
    """Scale so the loudest sample sits just below full scale.

    Generated speech arrives at whatever level the decoder produced, which
    varies between utterances. Without this the volume jumps between chunks of
    one reply, which sounds like a fault even when every chunk is correct.
    """
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return pcm

    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        return pcm
    gain = (target_peak * PCM16_MAX) / peak
    # Only ever attenuate loud audio; amplifying quiet audio also amplifies the
    # noise floor, and a near-silent clip would be blown up into hiss.
    if gain >= 1.0:
        return pcm
    return array.array("h", [int(sample * gain) for sample in samples]).tobytes()


# Full scale is 32767 positive and -32768 negative, so a symmetric test has to
# sit below the smaller of the two. Eight counts of headroom is under a
# thousandth of a decibel -- a sample this close to the rail was either clipped
# or was going to be.
CLIPPED_AMPLITUDE = 32_760


def clipping_ratio(pcm: bytes, threshold: int = CLIPPED_AMPLITUDE) -> tuple[float, int]:
    """How much of a clip is pinned at the rail, and the longest stretch of it.

    Two numbers rather than one because they describe different faults. A high
    ratio spread thinly is a clip that was mastered too hot and will sound
    harsh; a single long run is a decoder that produced a square edge, which is
    the audible pop. One sample at full scale is neither -- speech legitimately
    touches the rail on a plosive -- so the run length is what a threshold
    should eventually be set on, and this returns both rather than deciding.

    Measure before ``peak_normalise``. Normalisation scales the peak down to
    0.95 of full scale, so every clipped sample stops looking clipped while the
    square edge it left in the waveform stays exactly where it was.

    Deliberately not a verdict. The threshold that separates "a plosive" from
    "a fault" is not knowable from first principles here; it comes from the
    distribution over clips we already have. See section 2.4 of the group OKR
    for where the requirement comes from, and scripts/measure_clipping.py for
    the measurement that is meant to set the line.
    """
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0, 0

    pinned = 0
    longest = 0
    run = 0
    for sample in samples:
        if abs(sample) >= threshold:
            pinned += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return pinned / len(samples), longest


def trim_silence(
    pcm: bytes, threshold: float = 0.01, keep_ms: int = 50, rate: int = 24_000
) -> bytes:
    """Drop leading and trailing near-silence, keeping a short margin.

    Generated clips often start and end with dead air. Concatenated across a
    streamed reply that becomes an audible stutter between clauses.
    """
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return pcm

    limit = threshold * PCM16_MAX
    first = next((i for i, s in enumerate(samples) if abs(s) > limit), None)
    if first is None:
        return b""  # entirely silent
    last = next(i for i in range(len(samples) - 1, -1, -1) if abs(samples[i]) > limit)

    margin = int(rate * keep_ms / 1000)
    start = max(0, first - margin)
    end = min(len(samples), last + margin + 1)
    return samples[start:end].tobytes()
