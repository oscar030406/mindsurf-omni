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
