"""The fallback path: transcribe, think, speak -- three models in a row.

Slower in principle than sending audio tokens straight through, and it exists
anyway. Whether a ~140M model can speak Chinese well enough to ship is an open
question, while this arrangement is already known to work: a sister project
measured P95 1.93 s to first audio with these same components.

The latency here is not dominated by the models. It is dominated by *when*
synthesis starts. Waiting for a complete reply before speaking spends the whole
generation time before the listener hears anything; starting at the first
clause boundary spends only the time to produce that clause. That is the
single largest lever in the budget, so it is the thing this module is built
around.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from mindsurf_omni.contract import ComponentInfo, TokenSpec
from mindsurf_omni.service.engine import (
    EngineDescription,
    GenerationSettings,
    SpeechChunk,
    SpeechEngine,
    split_first_utterance,
)

# Callables rather than concrete models, so the pieces can be swapped -- local
# CosyVoice2, a hosted endpoint, or a stub in a test -- without this logic
# knowing which.
Transcriber = Callable[[bytes, int], Awaitable[tuple[str, str | None]]]
TextGenerator = Callable[[list[dict[str, str]], GenerationSettings], AsyncIterator[str]]
Synthesiser = Callable[[str, GenerationSettings], Awaitable[bytes]]


@dataclass(slots=True)
class CascadeTimings:
    """Where the time went, per turn.

    Recorded because the budget is a system property: shaving the model alone
    rarely moves the number the user feels.
    """

    transcribe_ms: float = 0.0
    first_clause_ms: float = 0.0
    first_synthesis_ms: float = 0.0
    time_to_first_audio_ms: float = 0.0


class CascadeEngine(SpeechEngine):
    def __init__(
        self,
        transcriber: Transcriber,
        generator: TextGenerator,
        synthesiser: Synthesiser,
        components: list[ComponentInfo],
        token_spec: TokenSpec,
    ) -> None:
        self._transcribe = transcriber
        self._generate = generator
        self._synthesise = synthesiser
        self._components = components
        self._token_spec = token_spec
        self.last_timings = CascadeTimings()

    def describe(self) -> EngineDescription:
        return EngineDescription(path="cascade", components=self._components)

    def token_spec(self) -> TokenSpec:
        return self._token_spec

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return await self._transcribe(pcm, sample_rate)

    def complete(
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        return self._generate(messages, settings)

    async def speak(  # type: ignore[override]
        self, text: str, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        pcm = await self._synthesise(text, settings)
        yield SpeechChunk(pcm=pcm, text=text, is_final=True)

    async def respond(  # type: ignore[override]
        self, pcm: bytes, sample_rate: int, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        started = time.perf_counter()
        timings = CascadeTimings()

        transcript, _ = await self._transcribe(pcm, sample_rate)
        timings.transcribe_ms = (time.perf_counter() - started) * 1000

        messages = [{"role": "user", "content": transcript}]
        pending = ""
        spoken_anything = False

        async for delta in self._generate(messages, settings):
            pending += delta
            clause = split_first_utterance(pending)
            if clause is None:
                continue

            if not spoken_anything:
                timings.first_clause_ms = (time.perf_counter() - started) * 1000
            synthesis_started = time.perf_counter()
            audio = await self._synthesise(clause, settings)
            if not spoken_anything:
                timings.first_synthesis_ms = (time.perf_counter() - synthesis_started) * 1000
                timings.time_to_first_audio_ms = (time.perf_counter() - started) * 1000
                spoken_anything = True

            pending = pending[len(clause) :]
            yield SpeechChunk(pcm=audio, text=clause)

        # Whatever is left has no sentence end -- say it anyway rather than
        # dropping the tail of the reply.
        remainder = pending.strip()
        if remainder:
            audio = await self._synthesise(remainder, settings)
            if not spoken_anything:
                timings.time_to_first_audio_ms = (time.perf_counter() - started) * 1000
            yield SpeechChunk(pcm=audio, text=remainder, is_final=True)
        elif spoken_anything:
            yield SpeechChunk(pcm=b"", is_final=True)

        self.last_timings = timings
