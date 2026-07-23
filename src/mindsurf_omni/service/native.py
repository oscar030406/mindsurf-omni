"""The native path: audio tokens straight through the model.

No transcript in the middle. Speech goes in as Mimi codes, the Thinker reasons
over them alongside text, and the Talker emits codes back -- so tone, pacing
and interruption survive a round trip that a transcript would flatten.

Written against the trained checkpoint's interface rather than waiting for it,
because the shape of this file is decided by the contract and by MiniMind-O's
model, both of which exist now. What is missing is the weights.

The chunking here is the whole latency argument. The Talker emits Mimi codes at
12.5 Hz, so 25 codes are two seconds of speech; decoding every 4 frames means
audio starts flowing about 320 ms into a reply instead of after it. That is the
same lever the cascade pulls by cutting at the first clause -- applied at the
token level, where it costs less.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from mindsurf_omni.contract import ComponentInfo, TokenSpec
from mindsurf_omni.service.audio import peak_normalise, resample, trim_silence
from mindsurf_omni.service.engine import (
    EngineDescription,
    GenerationSettings,
    SpeechChunk,
    SpeechEngine,
)

# Mimi's frame rate. Everything about the streaming budget follows from it.
MIMI_FRAME_RATE_HZ = 12.5
DEFAULT_CHUNK_FRAMES = 4  # about 320 ms


class OmniModel(Protocol):
    """What this path needs from MiniMind-O's model.

    Declared as a Protocol so the engine can be built and tested without the
    checkout: the real class satisfies it structurally, and a stub in a test
    satisfies it too.
    """

    def stream_generate(
        self, input_ids: Any, audio_inputs: Any | None, **kwargs: Any
    ) -> Any: ...


@dataclass(slots=True)
class NativeConfig:
    chunk_frames: int = DEFAULT_CHUNK_FRAMES
    max_new_tokens: int = 512
    # Trailing dead air between chunks concatenates into an audible stutter.
    trim_chunks: bool = True

    def __post_init__(self) -> None:
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be at least 1")

    @property
    def chunk_ms(self) -> float:
        return self.chunk_frames / MIMI_FRAME_RATE_HZ * 1000


class NativeEngine(SpeechEngine):
    """Speech in, speech out, with no transcript in between."""

    def __init__(
        self,
        model: OmniModel,
        codec: Any,
        tokenizer: Any,
        token_spec: TokenSpec,
        components: list[ComponentInfo],
        config: NativeConfig | None = None,
    ) -> None:
        self._model = model
        self._codec = codec
        self._tokenizer = tokenizer
        self._spec = token_spec
        self._components = components
        self._config = config or NativeConfig()

    def describe(self) -> EngineDescription:
        return EngineDescription(path="native", components=self._components)

    def token_spec(self) -> TokenSpec:
        return self._spec

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        """Available, but not on the path a reply takes.

        The native path never needs a transcript to answer. This exists so the
        endpoint behaves the same whichever engine is configured, and so a
        caller can read back what was heard.
        """
        return await self._recognise(pcm, sample_rate)

    async def _recognise(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        raise NotImplementedError("wire to the SenseVoice adapter once weights are loaded")

    def complete(
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        raise NotImplementedError("wire to the Thinker once weights are loaded")

    def speak(self, text: str, settings: GenerationSettings) -> AsyncIterator[SpeechChunk]:
        raise NotImplementedError("wire to the Talker once weights are loaded")

    def respond(
        self, pcm: bytes, sample_rate: int, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        raise NotImplementedError("wire to the Thinker-Talker loop once weights are loaded")


def chunk_codes(codes: list[list[int]], chunk_frames: int) -> list[list[list[int]]]:
    """Group Mimi frames into decodable chunks.

    Each frame is one value per codebook, so a chunk is a list of frames rather
    than a flat list. Splitting on the wrong axis would interleave codebooks
    and decode to noise -- which is why this is a named function with a test
    rather than a slice expression inside the generation loop.

    The final chunk is kept even when short: dropping it would clip the end of
    every reply.
    """
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be at least 1")
    return [codes[start : start + chunk_frames] for start in range(0, len(codes), chunk_frames)]


def prepare_output(pcm: bytes, sample_rate: int, target_rate: int, trim: bool) -> bytes:
    """Normalise one decoded chunk for playback.

    Trimming happens before resampling: the trim threshold is in sample
    amplitude, which resampling preserves, but the margin is in milliseconds,
    which only matches the rate it was measured at.
    """
    if trim:
        pcm = trim_silence(pcm, rate=sample_rate)
    pcm = peak_normalise(pcm)
    return resample(pcm, sample_rate, target_rate)
