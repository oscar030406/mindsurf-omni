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
from typing import Any

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
# The same job, handed over as it is produced. Optional because only a local
# synthesiser can do it: a hosted one pays a round trip that cannot be divided.
StreamingSynthesiser = Callable[[str, GenerationSettings], AsyncIterator[bytes]]


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
        unwired: tuple[str, ...] = (),
        stream_synthesiser: StreamingSynthesiser | None = None,
        polisher: Any = None,
    ) -> None:
        self._transcribe = transcriber
        self._generate = generator
        self._synthesise = synthesiser
        # Absent is the ordinary case, not a degraded one: assembly passes this
        # only when the wired synthesiser produces audio incrementally.
        self._stream_synthesise = stream_synthesiser
        # The dictation path's second stage. None is the ordinary case -- the
        # cascade also serves conversation, where polishing a question would
        # edit the user's words for no reason.
        self._polisher = polisher
        self._components = components
        self._token_spec = token_spec
        # Which of the three stages will refuse if called. Assembly knows this
        # and the health check cannot work it out by looking, so it is carried
        # rather than probed -- an engine that answers "ready" while a stage
        # raises is worse than no health check, because a backend routes on it.
        self.unwired = unwired
        self.last_timings = CascadeTimings()

    def describe(self) -> EngineDescription:
        return EngineDescription(path="cascade", components=self._components)

    def token_spec(self) -> TokenSpec:
        return self._token_spec

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return await self._transcribe(pcm, sample_rate)

    def warm(self) -> None:
        """Load what the first request would otherwise wait for.

        Called once at startup and never again. Only the recogniser: the
        polisher is already loaded at assembly, and the synthesiser is a third
        party's latency that the operator may not want paid at boot.
        """
        loader = getattr(self, "_warm_recogniser", None)
        if loader is not None:
            loader()

    async def polish(self, transcript: str) -> str | None:
        """The transcript tidied, or None when no polisher is wired.

        None rather than the transcript unchanged: a caller has to be able to
        tell "this service does not polish" from "this text needed no polish",
        and the dictation product routes on that difference.
        """
        if self._polisher is None:
            return None
        return await self._polisher.polish(transcript)

    def complete(
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        return self._generate(messages, settings)

    async def speak(  # type: ignore[override]
        self, text: str, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        # Streamed when the synthesiser can: waiting for the whole clause is
        # 2133.9 ms on the deployment card against 93 ms for the first piece,
        # and every caller of this method is playing the audio as it arrives.
        if self._stream_synthesise is None:
            pcm = await self._synthesise(text, settings)
            yield SpeechChunk(pcm=pcm, text=text, is_final=True)
            return

        said = False
        async for piece in self._stream_synthesise(text, settings):
            yield SpeechChunk(pcm=piece, text=None if said else text)
            said = True
        # The final marker is a separate empty chunk rather than a flag on the
        # last piece: the loop cannot know which piece is last until it ends.
        yield SpeechChunk(pcm=b"", text=None if said else text, is_final=True)

    async def respond(  # type: ignore[override]
        self,
        pcm: bytes,
        sample_rate: int,
        settings: GenerationSettings,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[SpeechChunk]:
        started = time.perf_counter()
        timings = CascadeTimings()

        transcript, _ = await self._transcribe(pcm, sample_rate)
        timings.transcribe_ms = (time.perf_counter() - started) * 1000

        # Before any audio, so a caller that stops reading early still learns
        # what was said -- the session needs it to record this turn.
        yield SpeechChunk(pcm=b"", transcript=transcript)

        messages = [*(history or []), {"role": "user", "content": transcript}]
        pending = ""
        spoken_anything = False

        async for delta in self._generate(messages, settings):
            # Text goes out as it is decided, not when its audio is ready.
            # Bound to the audio it measured 1748 ms against 1752 ms on a live
            # run: the reader saw nothing for the whole synthesis round trip,
            # while the words had been available since 100 ms. Audio chunks
            # below carry no text, so nothing is rendered twice.
            yield SpeechChunk(pcm=b"", text=delta)
            pending += delta
            clause = split_first_utterance(pending)
            if clause is None:
                continue

            if not spoken_anything:
                timings.first_clause_ms = (time.perf_counter() - started) * 1000
            synthesis_started = time.perf_counter()
            pending = pending[len(clause) :]

            if self._stream_synthesise is None:
                audio = await self._synthesise(clause, settings)
                if not spoken_anything:
                    timings.first_synthesis_ms = (time.perf_counter() - synthesis_started) * 1000
                    timings.time_to_first_audio_ms = (time.perf_counter() - started) * 1000
                    spoken_anything = True
                yield SpeechChunk(pcm=audio)
                continue

            async for piece in self._stream_synthesise(clause, settings):
                if not spoken_anything:
                    timings.first_synthesis_ms = (time.perf_counter() - synthesis_started) * 1000
                    timings.time_to_first_audio_ms = (time.perf_counter() - started) * 1000
                    spoken_anything = True
                yield SpeechChunk(pcm=piece)

        # Whatever is left has no sentence end -- say it anyway rather than
        # dropping the tail of the reply.
        remainder = pending.strip()
        if remainder:
            audio = await self._synthesise(remainder, settings)
            if not spoken_anything:
                timings.time_to_first_audio_ms = (time.perf_counter() - started) * 1000
            # No text: it went out with the deltas that produced it.
            yield SpeechChunk(pcm=audio, is_final=True)
        elif spoken_anything:
            yield SpeechChunk(pcm=b"", is_final=True)

        self.last_timings = timings
