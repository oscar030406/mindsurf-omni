"""The two speech paths, behind one interface.

The native path is the product's differentiator: audio tokens straight through
the model, no transcription round trip. The cascade path is ASR, then text,
then TTS, each a separate model.

Both exist on purpose. Whether a ~140M model can speak Chinese well enough to
ship is an open question -- MiniMind-O's own documentation says the Chinese
Talker is markedly harder than English -- while the cascade is already known
to work: a sister project measured P95 1.93 s end to end with this same class
of components. A product cannot have exactly one way to make sound.

So the choice is configuration, and callers cannot tell which one answered
except by asking /v1/models.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from mindsurf_omni.contract import ComponentInfo, TokenSpec

PathName = Literal["native", "cascade"]


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    """One piece of speech, ready to play.

    Chunks are emitted as they are produced rather than at the end, because
    the metric that matters is time to first audio, not total time. Waiting
    for a complete reply before speaking is the single largest avoidable cost
    in the latency budget.
    """

    pcm: bytes  # PCM16 little-endian, mono, OUTPUT_SAMPLE_RATE
    text: str | None = None  # the text this audio speaks, when known
    is_final: bool = False


@dataclass(slots=True)
class GenerationSettings:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    voice: str = "default"
    emotion: str = "neutral"


@dataclass(slots=True)
class EngineDescription:
    path: PathName
    components: list[ComponentInfo] = field(default_factory=list)
    licence: str = "CC-BY-NC-4.0"
    # Inherited from the text base's training data and carried by every
    # derivative, fine-tuned or not. Stated here so a caller cannot use the
    # service for months without meeting it.
    commercial_use_permitted: bool = False


class SpeechEngine(abc.ABC):
    """What both paths must provide."""

    @abc.abstractmethod
    def describe(self) -> EngineDescription: ...

    @abc.abstractmethod
    def token_spec(self) -> TokenSpec: ...

    @abc.abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        """Return the transcript and the detected language, if the encoder reports one."""

    @abc.abstractmethod
    def complete(
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        """Yield text deltas."""

    @abc.abstractmethod
    def speak(self, text: str, settings: GenerationSettings) -> AsyncIterator[SpeechChunk]:
        """Yield speech for already-decided text."""

    @abc.abstractmethod
    def respond(
        self, pcm: bytes, sample_rate: int, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        """Speech in, speech out.

        The native path answers this without ever materialising a transcript;
        the cascade path implements it as transcribe, complete, speak. Keeping
        it as one method is what lets the native path skip work the cascade
        cannot.
        """


def split_first_utterance(accumulated: str, minimum: int = 24, comma_after: int = 12) -> str | None:
    """Find the first chunk of text worth speaking, or None if it is too early.

    Speech synthesis cannot start until there is a sentence to say, so where
    this cuts sets the floor on time-to-first-audio. Sentence-final
    punctuation is the natural boundary, but Chinese text often runs long
    before reaching one; a clause boundary is used once the run is long enough
    that waiting costs more than the slightly early cut.

    The thresholds come from a sister project's measured voice pipeline, where
    this split is what took first audio from whole-reply latency down to
    P50 1.18 s.
    """
    for index, character in enumerate(accumulated):
        if character in "。！？!?":
            return accumulated[: index + 1]
    if len(accumulated) >= minimum:
        cut = max(accumulated.rfind(mark) for mark in "，、,")
        if cut >= comma_after:
            return accumulated[: cut + 1]
    return None
