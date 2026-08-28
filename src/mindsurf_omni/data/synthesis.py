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
from collections.abc import AsyncIterator
from dataclasses import dataclass
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


class SynthesiserUnavailable(RuntimeError):
    """The synthesiser was asked for audio and produced none.

    A RuntimeError still, so existing callers keep working, but named so the
    service can answer 502 rather than 500: an operator acts differently on "this
    service has a bug" and "the thing it depends on did not answer".
    """


_MARKDOWN = re.compile(r"[*#`_>~]+")
_SHORT_ASIDE = re.compile(r"[（(]([^）)]{1,20})[）)]")
_LONG_ASIDE = re.compile(r"[（(][^）)]{21,}[）)]")
_REPEATED_PUNCTUATION = re.compile(r"[。，]{2,}")
_WHITESPACE_RUN = re.compile(r"\s*\n+\s*")
# A character a speaker would voice. Punctuation is not one: it shapes how the
# rest is said, and on its own there is nothing to say.
_SPEAKABLE = re.compile(r"[\w一-鿿]")


def clean_for_speech(text: str) -> str:
    """Remove what reads badly aloud while keeping what was meant.

    A short parenthetical becomes an aside between commas, because a speaker would
    say it; a long one is dropped, because reading a paragraph inside brackets
    buries the sentence it interrupts.

    Text with nothing speakable left comes back empty rather than as bare
    punctuation: "。。。？！" is ordinary output for a chat model, and the hosted
    synthesiser answers it with no audio at all.
    """
    cleaned = _MARKDOWN.sub("", text)
    cleaned = cleaned.replace("——", "，").replace("—", "，")
    cleaned = _LONG_ASIDE.sub("", cleaned)
    cleaned = _SHORT_ASIDE.sub(r"，\1，", cleaned)
    cleaned = _WHITESPACE_RUN.sub("。", cleaned)
    cleaned = _REPEATED_PUNCTUATION.sub("。", cleaned)
    cleaned = cleaned.strip("，。 \t\n")
    return cleaned if _SPEAKABLE.search(cleaned) else ""


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

    Time to first audio needs the first samples, not the finished clause. Not every
    synthesiser can do that: a hosted one pays a network round trip that cannot be
    divided, while a local model pays compute, which can be handed over as it is
    produced. So this asks, and falls back to one whole piece -- a caller written
    against it does not branch.
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
                    raise SynthesiserUnavailable(
                        f"the synthesiser returned no audio for {spoken[:20]!r} twice "
                        f"({type(error).__name__}); a silent sample would be scored as "
                        "the model having said nothing"
                    ) from error
                continue
            if encoded:
                break
            if attempt == 2:
                raise SynthesiserUnavailable(
                    f"the synthesiser returned no audio for {spoken[:20]!r} twice; "
                    "a silent sample would be scored as the model having said nothing"
                )

        audio, rate = soundfile.read(io.BytesIO(bytes(encoded)), dtype="int16")
        return resample(audio.tobytes(), rate, OUTPUT_SAMPLE_RATE)
