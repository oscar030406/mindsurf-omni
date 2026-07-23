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
from dataclasses import dataclass
from typing import Protocol

# Delivery instructions, kept beside the text rather than inside it.
EMOTION_INSTRUCTIONS = {
    "neutral": "用亲切自然、娓娓道来的语气说",
    "happy": "用开心热情的语气说",
    "care": "用温柔关切、放慢语速的语气说",
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
    def synthesise(self, utterance: Utterance) -> bytes: ...


def instruction_leaked(spoken_text: str, transcript: str) -> bool:
    """Did the synthesiser read the delivery instruction aloud?

    Compared against the instruction rather than the text, because the failure
    is additive: the reply is still there, with the instruction in front of it.
    A length or similarity check on the reply alone would not see it.
    """
    if not transcript:
        return False
    normalised = re.sub(r"[^一-鿿 a-zA-Z]", "", transcript).lower()
    for instruction in EMOTION_INSTRUCTIONS.values():
        stripped = re.sub(r"[^一-鿿 a-zA-Z]", "", instruction).lower()
        # A prefix match rather than containment: the instruction is prepended,
        # and its words could legitimately appear inside a reply about tone.
        if stripped and normalised.startswith(stripped[: min(len(stripped), 8)]):
            return True
    return False


def screen_batch(
    utterances: list[Utterance], transcripts: list[str]
) -> list[tuple[Utterance, str]]:
    """Return the utterances whose audio does not match what was asked for.

    Run over every synthesised batch, not a sample of it. The leak is
    intermittent, so a spot check finds it in the batches it did not ruin.
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
