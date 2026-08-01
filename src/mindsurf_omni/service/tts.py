"""Speech synthesis for the cascade path.

Two things here are not obvious and both come from a sister project that ran
this stack in production.

Emotion is delivered as an instruction prefix, and the synthesiser sometimes
reads the instruction aloud. Their fix was to screen every batch through a
recogniser before shipping it; ours is to strip the instruction from what gets
compared and to keep the screening hook, because the failure is intermittent
and a spot check misses it.

And the text handed to a synthesiser is not the text shown on screen. Markdown
markers, long parenthetical asides and em dashes are all read out as something,
and none of it is what the writer meant. Cleaning is therefore part of
synthesis rather than a caller's responsibility -- every caller would have to
do it, and one would forget.
"""

from __future__ import annotations

import re
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE
from mindsurf_omni.service.audio import resample

# Delivery instructions, kept beside the text rather than inside it.
EMOTION_INSTRUCTIONS = {
    "neutral": "用亲切自然、娓娓道来的语气说",
    "happy": "用开心热情的语气说",
    "care": "用温柔关切、放慢语速的语气说",
}

# The same three emotions as prosody, for a synthesiser that has no instruct
# mode. Numbers are a starting point, not a measurement -- they are the knob to
# turn when someone listens and says it sounds wrong.
EDGE_PROSODY = {
    "neutral": {"rate": "+0%", "pitch": "+0Hz"},
    "happy": {"rate": "+8%", "pitch": "+25Hz"},
    "care": {"rate": "-8%", "pitch": "-10Hz"},
}

_MARKDOWN = re.compile(r"[*#`_>~]+")
_SHORT_ASIDE = re.compile(r"[（(]([^）)]{1,20})[）)]")
_LONG_ASIDE = re.compile(r"[（(][^）)]{21,}[）)]")
_REPEATED_PUNCTUATION = re.compile(r"[。，]{2,}")
_WHITESPACE_RUN = re.compile(r"\s*\n+\s*")


def clean_for_speech(text: str) -> str:
    """Remove what reads badly aloud while keeping what was meant.

    A short parenthetical becomes an aside between commas, because a speaker
    would say it. A long one is dropped, because a speaker would not read a
    paragraph inside brackets and doing so buries the sentence it interrupts.
    """
    cleaned = _MARKDOWN.sub("", text)
    cleaned = cleaned.replace("——", "，").replace("—", "，")
    cleaned = _LONG_ASIDE.sub("", cleaned)
    cleaned = _SHORT_ASIDE.sub(r"，\1，", cleaned)
    cleaned = _WHITESPACE_RUN.sub("。", cleaned)
    cleaned = _REPEATED_PUNCTUATION.sub("。", cleaned)
    return cleaned.strip("，。 \t\n")


@dataclass(frozen=True, slots=True)
class Utterance:
    """What was asked for, and what should come back."""

    text: str
    voice: str = "default"
    emotion: str = "neutral"

    def instruction(self) -> str:
        return EMOTION_INSTRUCTIONS.get(self.emotion, EMOTION_INSTRUCTIONS["neutral"])


class Synthesiser(Protocol):
    async def synthesise(self, utterance: Utterance) -> bytes: ...


async def stream_utterance(synthesiser: Any, utterance: Utterance) -> AsyncIterator[bytes]:
    """Audio as it becomes available, or the whole utterance once.

    The cascade spends 94% of its budget waiting for a clause to finish
    synthesising -- 2133.9 ms at the median against 126.0 ms for recognition
    and the Thinker together, measured on the deployment card. Time to first
    audio does not need the clause to be finished; it needs the first samples.

    Not every synthesiser can do that, and the split is not about effort. The
    hosted one pays a network round trip that cannot be divided, so chunking it
    buys the tail and not the head. A local model pays compute, which can be
    handed over as it is produced. So this asks, and falls back to one whole
    piece for anything that cannot -- a caller written against it does not
    branch.
    """
    streamer = getattr(synthesiser, "stream", None)
    if streamer is None:
        whole = await synthesiser.synthesise(utterance)
        if whole:
            yield whole
        return
    async for piece in streamer(utterance):
        if piece:
            yield piece


@dataclass(slots=True)
class EdgeSynthesiser:
    """Hosted synthesis, so the cascade can make sound before CosyVoice2 lands.

    Not the plan and not the default. It reaches a Microsoft endpoint, which
    means an inference container that cannot speak without outbound internet
    and replies that leave the machine -- both reasons this stays behind an
    explicit ``MINDSURF_TTS=edge``.

    It is here because the evaluation chain has never run on real audio. A
    synthesiser whose fidelity is already good turns that chain from untested
    code into a measured noise floor: whatever CER it reports is the floor the
    instrument itself contributes, and the Talker's number has to be read
    against it. Replacing this with CosyVoice2 changes the number, so the
    component list names it and every report carries that name.

    Output is 24 kHz mono, which is already ``OUTPUT_SAMPLE_RATE``; the
    resample call is there for the day that stops being true.
    """

    voice: str = "zh-CN-XiaoxiaoNeural"
    # Read by the same provenance checks the recognisers carry: a synthesiser
    # is never eligible to judge, least of all its own output.
    lineage: str = "edge-tts"
    eligible_as_judge: bool = False

    async def synthesise(self, utterance: Utterance) -> bytes:
        import io

        import edge_tts
        import soundfile

        spoken = clean_for_speech(utterance.text)
        if not spoken:
            # Synthesising nothing spends a request and plays a click.
            return b""

        # Deliberately not ``utterance.instruction()``. That prefix is what
        # CosyVoice2's instruct mode consumes; this endpoint has no such mode
        # and would simply read it out -- not the intermittent leak
        # ``instruction_leaked`` screens for, but every single utterance.
        prosody = EDGE_PROSODY.get(utterance.emotion, EDGE_PROSODY["neutral"])

        # Measured at 2 turns in 160: the endpoint intermittently returns no
        # audio at all for text it accepts. It is a hosted service on the far
        # side of a network, so one retry is the difference between a 1.25%
        # turn failure rate and a negligible one. Only one, and the second
        # failure is raised rather than swallowed -- a persistent fault is not
        # something to paper over with attempts.
        for attempt in (1, 2):
            encoded = bytearray()
            try:
                speech = edge_tts.Communicate(
                    spoken, self.voice, rate=prosody["rate"], pitch=prosody["pitch"]
                )
                async for chunk in speech.stream():
                    if chunk["type"] == "audio":
                        encoded += chunk["data"]
            except Exception as error:  # noqa: BLE001 - the endpoint's own failures
                if attempt == 2:
                    raise RuntimeError(
                        f"the synthesiser returned no audio for {spoken[:20]!r} twice "
                        f"({type(error).__name__}); a silent sample would be scored as "
                        "the model having said nothing"
                    ) from error
                continue
            if encoded:
                break
            if attempt == 2:
                raise RuntimeError(
                    f"the synthesiser returned no audio for {spoken[:20]!r} twice; "
                    "a silent sample would be scored as the model having said nothing"
                )

        audio, rate = soundfile.read(io.BytesIO(bytes(encoded)), dtype="int16")
        return resample(audio.tobytes(), rate, OUTPUT_SAMPLE_RATE)


@dataclass(slots=True)
class VoxCPMSynthesiser:
    """Local synthesis, so speaking stops being a network round trip.

    Why this is the lever: the cascade's P50 is 1976 ms, of which synthesis is
    1463 ms, of which 1222 ms is the trip to a hosted endpoint and back. The
    models in this path cost 393 ms between them. Nothing else on the list buys
    a second without training something.

    What it costs instead is a card. Weights load once and stay resident beside
    the Thinker, the Talker and SenseVoice, so this is 0.5B of company on a GPU
    that is already shared -- and the P95 does not disappear, it changes shape
    from network jitter into compute that grows with the sentence.

    Emotion is not carried. This model has no instruct mode and no prosody
    arguments, so the three delivery instructions have nowhere to go: putting
    one in the text would have it read aloud on every single utterance, which
    is the failure ``instruction_leaked`` exists to catch. The knob that does
    exist is voice cloning from a prompt clip, and an emotion-per-clip mapping
    would need recordings this repository does not have. Until then a caller
    asking for 'happy' gets neutral delivery, and the report says so rather
    than the audio quietly not matching the request.

    Loading is deferred to the first call. Assembly happens at process start,
    when a card may be busy with something else, and a container that cannot
    load weights should answer 503 rather than fail to boot.
    """

    model_id: str = "openbmb/VoxCPM-0.5B"
    device: str = "cuda"
    # A clip to clone, with what it says. Both or neither -- the model rejects
    # one without the other, and it is right to: the prompt text is what tells
    # it which sounds in the clip belong to which characters.
    prompt_wav: str | None = None
    prompt_text: str | None = None
    # wetext folds "10 美元" and "5G" into what a reader would say. 71 of the
    # 160 fixed evaluation texts contain digits, so leaving this off would
    # measure the front end rather than the voice.
    normalise_text: bool = True
    # VoxCPM emits 16 kHz. Stated rather than read off the loaded object: a
    # wrong rate here is not an exception, it is a chipmunk, and the bug report
    # says "sounds strange".
    sample_rate: int = 16_000
    lineage: str = "voxcpm"
    eligible_as_judge: bool = False
    _model: Any = None
    # Two first requests arriving together would otherwise each load half a
    # billion parameters onto the same card.
    _guard: threading.Lock = field(default_factory=threading.Lock)

    def load(self) -> Any:
        """The weights, loaded once and kept. Safe to call from a worker thread."""
        with self._guard:
            if self._model is None:
                from voxcpm import VoxCPM

                self._model = VoxCPM.from_pretrained(
                    self.model_id,
                    # The denoiser cleans a *prompt* clip before cloning, which
                    # is a separate ModelScope download and does nothing for
                    # text this path synthesises from scratch.
                    load_denoiser=False,
                    # torch.compile costs a warm-up per process and is not
                    # available on every host this runs on. Revisit when a
                    # measurement on the deployment card asks for it.
                    optimize=False,
                    device=self.device,
                )
            return self._model

    async def synthesise(self, utterance: Utterance) -> bytes:
        import asyncio

        import numpy

        spoken = clean_for_speech(utterance.text)
        if not spoken:
            # Synthesising nothing spends a forward pass and plays a click.
            return b""

        def speak() -> Any:
            # Loading happens in the worker thread too: the first request would
            # otherwise hold the event loop for the length of a weight load,
            # and every health check behind it.
            return self.load().generate(
                text=spoken,
                prompt_wav_path=self.prompt_wav,
                prompt_text=self.prompt_text,
                normalize=self.normalise_text,
            )

        # A GPU forward pass is seconds of blocking work, and the service is a
        # single event loop: doing it inline stalls every other connection,
        # including the health check the backend routes on.
        waveform = await asyncio.to_thread(speak)

        if waveform is None or len(waveform) == 0:
            raise RuntimeError(
                f"the synthesiser returned no audio for {spoken[:20]!r}; a silent "
                "sample would be scored as the model having said nothing"
            )
        pcm = (numpy.clip(numpy.asarray(waveform, dtype="float32"), -1.0, 1.0) * 32767).astype(
            "int16"
        )
        return resample(pcm.tobytes(), self.sample_rate, OUTPUT_SAMPLE_RATE)

    async def stream(self, utterance: Utterance) -> AsyncIterator[bytes]:
        """The same audio, handed over as the model produces it.

        Worth the extra path only because the cost here is compute rather than
        a round trip: the first samples exist long before the clause is done.
        Measured on a laptop card at 93 ms to the first chunk against 6405 ms
        for the whole clause, and the deployment card puts the whole clause at
        2133.9 ms -- so this is where the cascade's P95 overrun lives.

        The generator is synchronous and blocking, so it is drained on a worker
        thread one piece at a time. Draining it into a list first would restore
        exactly the wait this exists to remove.
        """
        import asyncio
        import queue

        import numpy

        spoken = clean_for_speech(utterance.text)
        if not spoken:
            return

        pieces: queue.Queue[Any] = queue.Queue()
        done = object()

        def produce() -> None:
            try:
                for piece in self.load().generate_streaming(
                    text=spoken,
                    prompt_wav_path=self.prompt_wav,
                    prompt_text=self.prompt_text,
                    normalize=self.normalise_text,
                ):
                    pieces.put(piece)
            except BaseException as error:  # surfaced on the consumer side
                pieces.put(error)
            finally:
                pieces.put(done)

        worker = asyncio.create_task(asyncio.to_thread(produce))
        spoke = False
        try:
            while True:
                piece = await asyncio.to_thread(pieces.get)
                if piece is done:
                    break
                if isinstance(piece, BaseException):
                    raise piece
                if piece is None or len(piece) == 0:
                    continue
                samples = (
                    numpy.clip(numpy.asarray(piece, dtype="float32"), -1.0, 1.0) * 32767
                ).astype("int16")
                spoke = True
                yield resample(samples.tobytes(), self.sample_rate, OUTPUT_SAMPLE_RATE)
        finally:
            await worker

        if not spoke:
            raise RuntimeError(
                f"the synthesiser streamed no audio for {spoken[:20]!r}; a silent "
                "sample would be scored as the model having said nothing"
            )


def instruction_leaked(spoken_text: str, transcript: str) -> bool:
    """Did the synthesiser read the delivery instruction aloud?

    Compared against the instruction rather than the text, because the failure
    is additive: the reply is still there, with the instruction in front of it.
    A length or similarity check on the reply alone would not see it.

    What actually leaks is not always the whole instruction. The sister project
    that hit this reproducibly saw only its *tail* -- "…的语气说吗？" -- survive
    into the audio, because the template's instruction and text are separated
    by a marker the synthesiser sometimes starts reading just after. Matching
    the instruction's first characters therefore misses the observed failure,
    so any suffix of the instruction counts.

    Still anchored at the start rather than containment: whatever fragment
    leaks, it is read *before* the reply. A reply that merely discusses tone
    contains those words in the middle, and flagging it would train everyone to
    ignore the screen.
    """
    if not transcript:
        return False
    normalised = re.sub(r"[^一-鿿 a-zA-Z]", "", transcript).lower()
    for instruction in EMOTION_INSTRUCTIONS.values():
        stripped = re.sub(r"[^一-鿿 a-zA-Z]", "", instruction).lower()
        if not stripped:
            continue
        # Every suffix long enough not to fire on ordinary speech. Five
        # characters of an instruction ending is already "的语气说吗", which is
        # not a way a reply begins.
        for start in range(len(stripped) - 4):
            if normalised.startswith(stripped[start:][: min(len(stripped) - start, 8)]):
                return True
    return False


def screen_batch(
    utterances: list[Utterance], transcripts: list[str]
) -> list[tuple[Utterance, str]]:
    """Return the utterances whose audio does not match what was asked for.

    Run over every synthesised batch, not a sample of it. The leak is
    intermittent, so a spot check finds it in the batches it did not ruin.

    Necessary and not sufficient. A recogniser only reports words, so this
    screen is blind to anything that is not speech: the sister project chased
    intermittent pops through 38 clips of acoustic screening and clean
    spectrograms before concluding they were random and unreproducible. Their
    fix was at playback -- a limiter after the analyser -- not a better screen.
    A batch that passes here has been shown to say the right words, not to
    sound right.
    """
    if len(utterances) != len(transcripts):
        raise ValueError(
            f"{len(utterances)} utterances against {len(transcripts)} transcripts; "
            "a misaligned pair would blame the wrong sample"
        )
    suspect = []
    for utterance, transcript in zip(utterances, transcripts, strict=True):
        if instruction_leaked(utterance.text, transcript) or not transcript.strip():
            suspect.append((utterance, transcript))
    return suspect
