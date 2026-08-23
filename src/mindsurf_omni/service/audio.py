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

# What a recorder can plausibly have been running at. Below the first, a header
# turns a short clip into hours; above the second, into microseconds.
# Output samples converted at a time. Big enough that the per-block overhead
# does not show, small enough that the working set is a few tens of megabytes
# whatever arrives.
RESAMPLE_BLOCK = 1 << 18

SLOWEST_RATE = 4_000
FASTEST_RATE = 384_000


class UnsupportedAudio(ValueError):
    """A container whose header parsed and whose samples this service cannot read.

    Its own type so the boundary can turn it into a 400: the caller has to send
    a different file, and no amount of retrying the same one helps. Distinct
    from a body that is not a container at all -- that one is taken as raw
    samples and always has been.
    """


def whole_samples(pcm: bytes) -> bytes:
    """The part of a PCM16 buffer that is complete samples.

    A recorder flushed mid-sample sends an odd number of bytes, which numpy refuses
    outright. Trimmed rather than rejected: half a sample is 31 microseconds, and a
    client that chunks its stream on a byte boundary is not doing anything wrong.
    """
    return pcm[: len(pcm) // 2 * 2]


def unwrap_wav(body: bytes, declared_rate: int) -> tuple[bytes, int]:
    """Samples and their real rate, whether the caller sent a container or not.

    The endpoint takes raw PCM, and a client that posts a whole .wav instead has
    always worked by accident: the 44-byte header decodes to about twenty
    samples of noise in front of the speech and the recogniser reads past it.
    Read properly it costs nothing and the rate comes from the one place that
    actually records it -- a header inside the file is worth more than a rate
    the caller declares out of band, because a client that gets one wrong
    usually gets the other wrong the same way.

    Anything that is not a RIFF/WAVE container is returned untouched under the
    rate the caller declared. A container that *is* one and holds something
    other than 16-bit mono raises: see ``UnsupportedAudio``.
    """
    if len(body) < 44 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return body, declared_rate

    cursor, rate, bits, channels = 12, declared_rate, 16, 1
    while cursor + 8 <= len(body):
        name = body[cursor : cursor + 4]
        size = int.from_bytes(body[cursor + 4 : cursor + 8], "little")
        payload = cursor + 8
        if name == b"fmt " and payload + 16 <= len(body):
            channels = int.from_bytes(body[payload + 2 : payload + 4], "little") or 1
            stated = int.from_bytes(body[payload + 4 : payload + 8], "little")
            # Only inside the range a recorder can produce. The header is the
            # caller's to write, and four bytes of it decided three things:
            # duration_seconds became whatever the caller said (rate 1 turned a
            # one-second clip into 16000 seconds), the /stats audio counter went
            # with it, and resample allocated against the ratio.
            rate = stated if SLOWEST_RATE <= stated <= FASTEST_RATE else declared_rate
            bits = int.from_bytes(body[payload + 14 : payload + 16], "little") or 16
        elif name == b"data":
            samples = body[payload : payload + size] if size else body[payload:]
            if bits != 16 or channels != 1:
                # Refused rather than handed on. Handing it on meant the whole
                # container went to the recogniser as if it were samples, and
                # the recogniser does not fail on nonsense -- it writes
                # something. Measured over 12 clips, each scored against what
                # the same recording gives when it is read properly:
                #
                #   16 kHz stereo    mean CER 0.029, worst 0.073
                #   16 kHz 8-bit     mean CER 0.852, every clip wrong
                #   16 kHz 24-bit    empty transcript on all 12
                #
                # all of them HTTP 200. Stereo is the mild one and is still
                # refused: interleaving two copies of a mono source reads as
                # the recording at half speed, so the duration this endpoint
                # reports would be double as well, and a dictation user cannot
                # see either. The header states exactly what the file is, so
                # the answer can name it and name the fix. Downmixing instead
                # is a few lines, and is the right change the day somebody
                # actually sends stereo -- guessing is what this replaces.
                raise UnsupportedAudio(
                    f"this wav holds {channels}-channel {bits}-bit samples; this service "
                    "reads 16-bit mono. Convert it first -- ffmpeg -ac 1 -sample_fmt s16 -- "
                    "or post the samples with no container at all"
                )
            return samples, rate
        cursor = payload + size + (size & 1)
    return body, declared_rate


def resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear interpolation between rates.

    Not a windowed sinc: the conversions here are 16k<->24k on speech that a
    recogniser or a person will consume, and the aliasing a linear resampler
    leaves at that ratio is far below what the codec already introduces.
    Reach for a proper filter if this ever touches music.

    Done in numpy rather than a Python loop over samples. The loop holds the
    GIL for its whole length, and this runs on the thread the recogniser runs
    on: four concurrent minute-long conversions took the event loop's heartbeat
    from 6 ms late to 205 ms late, which is the health check missing its window
    for a reason nothing in the logs would explain.
    """
    if source_rate == target_rate or not pcm:
        return pcm

    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16)
    if samples.size == 0:
        return b""

    count = max(1, int(samples.size * target_rate / source_rate))
    step = source_rate / target_rate
    out = np.empty(count, dtype=np.int16)
    # In blocks, and never through ``np.interp``. That function takes float64
    # for all three arrays and materialises an index for every input sample, so
    # an hour of 48 kHz wav -- a body this endpoint accepts -- peaked at 6 GB,
    # eighteen times the bytes that arrived. A block is bounded whatever the
    # recording is, and the positions inside one still come out in float64
    # because a float32 index stops being exact past 2**24 samples, which is
    # seventeen minutes.
    for start in range(0, count, RESAMPLE_BLOCK):
        stop = min(start + RESAMPLE_BLOCK, count)
        where = np.arange(start, stop, dtype=np.float64) * step
        # int32 is enough for any body this endpoint accepts: the transport
        # limit is under 2**31 samples by an order of magnitude.
        left = np.clip(where.astype(np.int32), 0, samples.size - 1)
        rise = (where - left).astype(np.float32)
        right = np.minimum(left + 1, samples.size - 1)
        # Widened before subtracting: two int16 samples at opposite rails
        # differ by more than int16 holds, and the wrap showed up as a spike of
        # 55064 against the reference.
        base = samples[left].astype(np.float32)
        drawn = base + (samples[right].astype(np.float32) - base) * rise
        out[start:stop] = np.clip(drawn, -32768, 32767).astype(np.int16)
    return out.tobytes()


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

    Two numbers because they describe different faults: a high ratio spread thinly
    is a clip mastered too hot, a single long run is a decoder producing a square
    edge, which is the audible pop. One sample at full scale is neither -- speech
    touches the rail on a plosive.

    Measure before ``peak_normalise``: normalisation scales the peak down to 0.95,
    so every clipped sample stops looking clipped while the square edge stays where
    it was.

    Deliberately not a verdict. Where the line between a plosive and a fault sits
    comes from the distribution over clips, not from first principles; see
    ``scripts/measure_clipping.py``.
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
