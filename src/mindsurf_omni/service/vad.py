"""Decide when the user has stopped talking.

This is the first term in the latency budget and the one most often mis-set.
Too eager and the model answers a half-finished sentence; too patient and every
turn pays the wait, whether or not the user had more to say. It is also the
only stage that can be improved without touching a model.

Energy-based, deliberately. A neural detector is more robust in noise, but it
is another model to load, another dependency, and another thing to be wrong
about -- and the product's first environment is someone at a desk with a
headset, not a factory floor. The threshold adapts to the observed noise floor
so it does not have to be tuned per microphone.

**Nothing in the service calls this, on purpose, and that has to be said here
rather than discovered.** The frozen contract endpoints turns with an explicit
``input_audio_buffer.commit`` from the client, so the client's VAD decides and
this module is not in the path. A tested module with no caller is the shape of
a feature that looks shipped and is not -- the same shape as the evaluation
stub that was more correct than its implementation. It is kept rather than
deleted because the group document's own audio-input section asks for
server-side endpointing (a 300 ms silence auto-commit), and this is what would
serve that if the contract ever grows the event. See DECISIONS.md section 11.

ponytail: energy VAD with an adaptive floor; swap in Silero if the measured
false-endpoint rate on real recordings exceeds a few percent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

FRAME_MS = 20


def frame_energy(pcm: bytes) -> float:
    """Root-mean-square amplitude of one frame, normalised to 0..1."""
    if len(pcm) < 2:
        return 0.0
    import array

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    total = sum(float(sample) * sample for sample in samples)
    return math.sqrt(total / len(samples)) / 32768.0


@dataclass(slots=True)
class EndpointDetector:
    """Streaming endpoint detection over fixed-size frames.

    ``silence_ms`` is the wait after speech stops before the turn is declared
    over. It buys robustness against mid-sentence pauses and it is paid on
    every single turn, which is why it is stated here rather than buried.
    """

    sample_rate: int = 16_000
    silence_ms: int = 300
    min_speech_ms: int = 200
    # Speech must exceed the noise floor by this factor. Ratio rather than an
    # absolute level, so a quiet microphone does not read as permanent silence.
    speech_ratio: float = 3.0
    initial_noise_floor: float = 0.005

    _noise_floor: float = field(default=0.0, init=False)
    _speech_ms: int = field(default=0, init=False)
    _silence_ms_seen: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._noise_floor = self.initial_noise_floor

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * FRAME_MS / 1000) * 2

    def reset(self) -> None:
        self._noise_floor = self.initial_noise_floor
        self._speech_ms = 0
        self._silence_ms_seen = 0
        self._started = False

    @property
    def speech_started(self) -> bool:
        return self._started

    def push(self, frame: bytes) -> bool:
        """Feed one frame; True once the turn is over.

        Endpoint requires speech to have been heard first. Without that, a
        silent room would end a turn that never began, and the model would
        answer nothing.
        """
        energy = frame_energy(frame)
        is_speech = energy > self._noise_floor * self.speech_ratio

        if is_speech:
            self._speech_ms += FRAME_MS
            self._silence_ms_seen = 0
            if self._speech_ms >= self.min_speech_ms:
                self._started = True
            return False

        # Track the floor only while quiet, so loud speech cannot drag it up
        # and deafen the detector to everything that follows.
        self._noise_floor = 0.95 * self._noise_floor + 0.05 * energy

        if not self._started:
            return False
        self._silence_ms_seen += FRAME_MS
        return self._silence_ms_seen >= self.silence_ms

    def push_stream(self, pcm: bytes) -> int | None:
        """Feed a buffer; return the byte offset where the turn ended, or None."""
        size = self.frame_bytes
        for offset in range(0, len(pcm) - size + 1, size):
            if self.push(pcm[offset : offset + size]):
                return offset + size
        return None


# Where a long recording gets cut. Below this it goes through the encoder in
# one pass, which is what every measurement in the repository was taken on.
#
# Swept on the 4090 over concatenated speech from 22 to 900 seconds, at two
# pause lengths, against the text the clips were read from:
#
#     seconds    whole CER   split CER    whole ms   split ms
#        22        0.0128      0.0128         54         55
#        61        0.0271      0.0271         84        115
#        81        0.0198      0.0198         97        169
#       110        0.0189      0.0142        135        226
#       217        0.0177      0.0118        359        450
#       315        0.0194      0.0122        648        632
#       900        0.1373      0.0125       4263       1861
#
# Two regimes with a knee between 81 and 110 seconds. Under it the two agree to
# four places and splitting only buys a second forward pass; over it the single
# pass starts dropping content -- at 900 seconds it wrote 3409 characters where
# 4243 were spoken and lost every list number and full stop, at HTTP 200 with
# nothing in the log. Measured on synthesised speech, so the pauses are a
# synthesiser's rather than a person's; the knee could move on real recordings.
SEGMENT_ABOVE_SECONDS = 90.0

# What each piece aims for. SenseVoice was trained on utterance-length audio,
# and the polisher groups transcripts at 160 characters -- about 35 seconds of
# speech -- so a piece near this length gives the polisher the same shape of
# input it saw in training. Longer wastes memory, shorter fragments sentences.
SEGMENT_TARGET_SECONDS = 25.0

# Cut here even if nobody has paused. Somebody reading aloud without breathing
# is rare; an open microphone next to a fan is not, and it has no silences at
# all. Above the target so an ordinary pause is always preferred to this.
SEGMENT_HARD_SECONDS = 45.0

# A gap this long is a place to cut. Shorter than the 300 ms end-of-turn wait:
# this is not deciding the turn is over, only where a seam does least harm.
SEGMENT_QUIET_MS = 200


def speech_spans(
    pcm: bytes, sample_rate: int = 16_000, quiet_ms: int = SEGMENT_QUIET_MS
) -> list[tuple[int, int]]:
    """Byte ranges that hold speech, separated by gaps of at least ``quiet_ms``.

    The floor adapts the same way ``EndpointDetector`` adapts it, so a quiet
    microphone does not read as one long silence and a loud one does not read
    as one long span.
    """
    frame = int(sample_rate * FRAME_MS / 1000) * 2
    if frame <= 0 or len(pcm) < frame:
        return [(0, len(pcm))] if pcm else []

    # Read off an instance rather than restated, so the two stay one setting.
    tuning = EndpointDetector(sample_rate=sample_rate)
    floor = tuning.initial_noise_floor
    quiet_frames = max(1, quiet_ms // FRAME_MS)

    spans: list[tuple[int, int]] = []
    start: int | None = None
    spoke = 0
    quiet = 0
    for offset in range(0, len(pcm) - frame + 1, frame):
        energy = frame_energy(pcm[offset : offset + frame])
        if energy > floor * tuning.speech_ratio:
            if start is None:
                start = offset
            spoke = offset + frame
            quiet = 0
            continue
        floor = 0.95 * floor + 0.05 * energy
        if start is None:
            continue
        quiet += 1
        if quiet >= quiet_frames:
            spans.append((start, spoke))
            start = None
            quiet = 0
    # Ends at the last frame that held speech, not at the end of the buffer.
    # Somebody who keeps the key down for ten seconds after they finish would
    # otherwise hand the last piece ten seconds of nothing, and the guard that
    # reads a speaking rate would throw the piece away for being too slow.
    if start is not None:
        spans.append((start, spoke))
    return spans


def segments(
    pcm: bytes,
    sample_rate: int = 16_000,
    target_seconds: float = SEGMENT_TARGET_SECONDS,
    hard_seconds: float = SEGMENT_HARD_SECONDS,
) -> list[tuple[int, int]]:
    """Cut a recording into pieces of about ``target_seconds``, cutting in the pauses.

    Returned as offsets into ``pcm`` rather than as copies so the caller decides
    what to hold in memory; the whole reason this exists is that holding all of
    a long recording in the encoder at once asks the allocator for tens of
    gigabytes.

    Pieces keep the silence between the speech they contain -- only the gap at
    a seam is dropped -- because a recogniser given speech with the pauses
    removed writes the two sides as one word.
    """
    per_second = sample_rate * 2
    spans = speech_spans(pcm, sample_rate)
    if not spans:
        return []

    out: list[tuple[int, int]] = []
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        # Two ways to close a piece: it has reached the size it aims for, or
        # taking the next span would push it past the size it may not exceed.
        if end - start >= target_seconds * per_second or (
            next_end - start > hard_seconds * per_second
        ):
            out.append((start, end))
            start, end = next_start, next_end
        else:
            end = next_end
    out.append((start, end))

    # A span longer than the hard limit on its own has no pause to cut in, so
    # it gets cut where the limit falls. This is the case that breaks a word,
    # and it is preferred to the allocator failure it replaces.
    limit = int(hard_seconds * per_second)
    forced: list[tuple[int, int]] = []
    for begin, finish in out:
        while finish - begin > limit:
            forced.append((begin, begin + limit))
            begin += limit
        forced.append((begin, finish))
    return forced
