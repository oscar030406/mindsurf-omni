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
    TooLongForModel,
    split_first_utterance,
)
from mindsurf_omni.service.polish import group_sentences

# How much of one spoken turn goes into one message, and how much of it the
# model will answer at all. Both measured on sft_merge_768.pth, 8 prompts per
# point, held-out text:
#
#   one message of 968 tokens        8/8 empty
#   the same words as user turns     0/8 empty  (6 turns, 998 tokens)
#   the same words at 1502 tokens    8/8 empty, in every shape tried
#
# So a long turn is answerable, but only in the shape the weights have seen --
# short messages -- and only up to a point past which nothing helps. 240 and
# 1300 characters are those two token figures at the ~1.39 characters per token
# this tokenizer gives Chinese, rounded in.
SPOKEN_TURN_CHARACTERS = 240
ANSWERABLE_CHARACTERS = 1300

# Callables rather than concrete models, so the pieces can be swapped -- local
# CosyVoice2, a hosted endpoint, or a stub in a test -- without this logic
# knowing which.
Transcriber = Callable[[bytes, int], Awaitable[tuple[str, str | None]]]
TextGenerator = Callable[[list[dict[str, str]], GenerationSettings], AsyncIterator[str]]
Synthesiser = Callable[[str, GenerationSettings], Awaitable[bytes]]
# The same job, handed over as it is produced. Optional because only a local
# synthesiser can do it: a hosted one pays a round trip that cannot be divided.
StreamingSynthesiser = Callable[[str, GenerationSettings], AsyncIterator[bytes]]


def spoken_turn(transcript: str) -> list[dict[str, str]]:
    """One dictated turn as messages the model will answer.

    The caller of /v1/chat/completions who sends one very long message is told
    to split it, because they can. Nobody can split this one: the recogniser
    produced it, from however long the person spoke before they stopped, and by
    the time it exists the words have been said. Measured live, 124 seconds of
    continuous speech recognised to 553 characters and came back as a refusal
    with the turn not even recorded -- two minutes of talking, nothing.

    So the split happens here, on sentence boundaries, into the short messages
    the weights have seen. Same reasoning as the polish path's grouping and the
    same helper: a transcript inside the length goes through untouched, so the
    ordinary turn behaves exactly as it did.
    """
    if len(transcript) <= SPOKEN_TURN_CHARACTERS:
        return [{"role": "user", "content": transcript}]
    if len(transcript) > ANSWERABLE_CHARACTERS:
        # Past this the split stops helping -- measured at 1502 tokens, every
        # shape came back empty 8 times out of 8. Refused rather than answered
        # from a piece of it: dropping the oldest half of what somebody just
        # said, silently, to answer the rest is the failure this whole round
        # has been about.
        raise TooLongForModel(
            f"the committed audio recognised to {len(transcript)} characters, and this "
            f"checkpoint answers up to about {ANSWERABLE_CHARACTERS} in one turn even when "
            "the turn is split into messages. Commit at the end of each thing the speaker "
            "says rather than buffering the whole monologue"
        )
    return [
        {"role": "user", "content": piece}
        for piece in group_sentences(transcript, SPOKEN_TURN_CHARACTERS)
    ]


def answerable(history: list[dict[str, str]], transcript: str) -> list[dict[str, str]]:
    """The whole prompt, in messages this checkpoint answers.

    History gets the same treatment as the current turn, because it is made of
    turns that were once current: a monologue answered on one turn was recorded
    whole, and on the next turn that single 487-token history entry was refused
    -- the turn before had just worked. Splitting only what is about to be said
    fixes one turn and breaks the one after it.

    History is split rather than refused when it is very long. Dropping old
    turns is already what this conversation does when the budget runs out, and
    it is the right trade here too; refusing is reserved for what the speaker
    just said, which is not recoverable any other way.
    """
    fitted: list[dict[str, str]] = []
    for message in history:
        content = message.get("content", "")
        if message.get("role") == "user" and len(content) > SPOKEN_TURN_CHARACTERS:
            fitted += [
                {"role": "user", "content": piece}
                for piece in group_sentences(content, SPOKEN_TURN_CHARACTERS)
            ]
        else:
            fitted.append(message)
    return [*fitted, *spoken_turn(transcript)]


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


def worth_answering(transcript: str) -> str | None:
    """The transcript, or None when answering it would be answering nobody.

    Empty is the obvious half: two seconds of silence came back as a real reply
    to a question nobody asked, in English, synthesised and played.

    The other half is a transcript that is nothing but one hesitation. Half a
    second of a fan, of mains hum, of a room with people in it, all come back
    from the recogniser as 嗯。 -- measured on 45 synthetic noise clips, 22 of
    them wrote something and the ones under a second all wrote a single filler.
    The recogniser cannot separate those from somebody actually saying 嗯,
    because nothing in the audio does: the envelope test that would have is
    unusable, since ordinary speech through a speakerphone in a live room lands
    in the same range as steady noise.

    So the split is made here instead, where the cost is not symmetric. In
    dictation a stray 嗯 is two characters in the text box. In conversation it
    is a spoken reply to a button somebody pressed by accident, and a person
    whose whole turn was one 嗯 did not ask for one either.
    """
    from mindsurf_omni.service.polish import LEADING_FILLERS, RECOGNISED_FILLERS

    said = transcript.strip(" .。,，!！?？、;；:：\n")
    if not said:
        return None
    alone = {*LEADING_FILLERS, *RECOGNISED_FILLERS, "恩", "温", "饿", "恶", "啊", "扼"}
    return None if said in alone else transcript


class CascadeEngine(SpeechEngine):
    def __init__(
        self,
        transcriber: Transcriber,
        generator: TextGenerator,
        # None since the product stopped speaking. The dictation path never
        # reaches it; the conversation path refuses without it, which is what a
        # deployment that wired no synthesiser already got. Kept in position so
        # the callers that pass this positionally keep working.
        synthesiser: Synthesiser | None,
        components: list[ComponentInfo],
        token_spec: TokenSpec,
        unwired: tuple[str, ...] = (),
        stream_synthesiser: StreamingSynthesiser | None = None,
        polisher: Any = None,
        recogniser: Any = None,
    ) -> None:
        self._transcribe = transcriber
        self._generate = generator
        self._synthesise = synthesiser
        self._stream_synthesise = stream_synthesiser
        # Absent is the ordinary case, not a degraded one: assembly passes this
        # only when the wired synthesiser produces audio incrementally.
        # The dictation path's second stage. None is the ordinary case -- the
        # cascade also serves conversation, where polishing a question would
        # edit the user's words for no reason.
        self._polisher = polisher
        # The object, not just its transcribe method: a recogniser that can
        # write while the speaker is still talking exposes that separately, and
        # a callable cannot carry it. Named rather than reached for with
        # getattr, because a getattr that misses returns None and a socket that
        # never streams looks exactly like a recogniser that cannot.
        self.recogniser = recogniser
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

    async def polish(self, transcript: str, language: str | None = None) -> str | None:
        """The transcript tidied, or None when no polisher is wired.

        None rather than the transcript unchanged: a caller has to be able to
        tell "this service does not polish" from "this text needed no polish",
        and the dictation product routes on that difference.

        ``language`` is what the recogniser reported for this turn. The stage
        edits Chinese and English and hands everything else back untouched; see
        POLISHED_LANGUAGES for what a Chinese-trained model did to Cantonese.
        """
        if self._polisher is None:
            return None
        return await self._polisher.polish(transcript, language)

    def complete(
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        return self._generate(messages, settings)

    async def speak(  # type: ignore[override]
        self, text: str, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        """Read the given text back, when a deployment turned that on.

        On request, never on its own. Nothing in a dictation turn calls this --
        it exists so a client can put a speaker button next to a finished note.
        Unwired, the callable assembly handed us refuses and says how to turn it
        on, which is the ordinary case.
        """
        pcm = await self._synthesise(text, settings)
        yield SpeechChunk(pcm=pcm, text=text, is_final=True)

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

        # Nothing was said, so there is nothing to answer. Sending the empty
        # transcript on as a user message produced a real reply to a question
        # nobody asked: two seconds of silence came back as "Sure, what's your
        # question?", in English, synthesised and played. The session drops
        # empty turns from history for the same reason -- a bare
        # <|im_start|>user<|im_end|> is a shape the model has never seen -- but
        # the current turn was reaching it unfiltered.
        if not worth_answering(transcript):
            self.last_timings = timings
            return

        messages = answerable(history or [], transcript)
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
