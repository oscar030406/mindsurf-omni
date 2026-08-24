"""Speech recognition, in two roles that must not be filled by one model.

SenseVoice serves the product: it is the cascade path's recogniser, and the
native path's audio encoder. It is fast, it is already on disk, and it is
frozen.

It must not be the judge. CER scores the speech our model produced; if the
recogniser scoring that speech shares lineage with the model producing it,
their failure modes cancel and the number is highest exactly where it should
warn. So evaluation uses an independent recogniser, and this module refuses to
hand out the service one for that purpose rather than trusting a convention
nobody remembers in three weeks.
"""

from __future__ import annotations

import re
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mindsurf_omni.service.engine import TooLongForModel
from mindsurf_omni.service.vad import (
    SEGMENT_ABOVE_SECONDS,
    SEGMENT_HARD_SECONDS,
    segments,
    voiced_seconds,
)

# SenseVoice emits inline tags for language, emotion and audio events. They are
# not speech, and leaving them in would count them as characters against the
# reference text.
_TAG = re.compile(r"<\|[^|]*\|>")
_LANGUAGE_TAG = re.compile(r"<\|(zh|en|yue|ja|ko|nospeech)\|>")


def strip_tags(raw: str) -> tuple[str, str | None]:
    """Return the spoken text and the language SenseVoice reported."""
    match = _LANGUAGE_TAG.search(raw)
    return _TAG.sub("", raw).strip(), match.group(1) if match else None


class Recogniser(Protocol):
    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]: ...


# Below this the buffer holds no speech: digital silence is 0.0 and a quiet
# room reads around 0.001. Ordinary speech sits two orders of magnitude above
# it, so this drops the button pressed by mistake without dropping a whisper.
SILENCE_RMS = 0.002

# The seven values SenseVoice knows (funasr's `lid_dict`). Anything else is
# taken silently as "auto", so a typo would look configured and behave
# unconfigured. Checked at startup instead.
SPOKEN_LANGUAGES = ("auto", "zh", "en", "yue", "ja", "ko", "nospeech")

# Below this the recogniser's own language detection stops being worth
# trusting: short particles come back as another language (嗯 as うん) and the
# detector is 0.99 confident when they do, so no threshold on its posterior
# separates them. Measured on two corpora; from 1.5 s up neither misfires.
SHORT_AUDIO_SECONDS = 1.5

# What SenseVoice's frontend runs at; the service resamples to it before here.
RECOGNISER_RATE = 16_000

# One fbank frame. Shorter than this the model only ever writes punctuation,
# and a single sample makes the frontend assert rather than return.
SHORTEST_SAMPLES = 400

# The longest recording accepted in one request. Not a memory limit: past
# SEGMENT_ABOVE_SECONDS no single forward pass sees more than one piece. What
# this bounds is the request body -- an hour of 16 kHz PCM16 is 115 MB.
LONGEST_SECONDS = 3600.0

# What to cut a long recording into when the pause detector finds no pauses at
# all. Bytes, at SEGMENT_HARD_SECONDS -- the same bound a forced cut uses.
FALLBACK_WINDOW = int(SEGMENT_HARD_SECONDS * RECOGNISER_RATE) * 2


# What each spoken language is written in. Latin letters and digits are in
# every one of them, so they are not listed: a transcript of 咖啡, demo or B203
# has to survive whatever the deployment declared.
SCRIPTS: dict[str, tuple[tuple[str, str], ...]] = {
    "zh": (("\u3400", "\u9fff"), ("\uf900", "\ufaff")),
    "yue": (("\u3400", "\u9fff"), ("\uf900", "\ufaff")),
    "ja": (("\u3040", "\u30ff"), ("\u3400", "\u9fff"), ("\uff66", "\uff9d")),
    "ko": (("\uac00", "\ud7a3"), ("\u1100", "\u11ff"), ("\u3130", "\u318f")),
    "en": (),
}


def spoken_scripts(language: str) -> tuple[tuple[str, str], ...]:
    """The character ranges a deployment declaring this language could produce.

    ``auto`` means the operator did not narrow it, so every script the
    recogniser knows counts. Anything else narrows the gate below to that one --
    which is what the gate is for, and also why it has to be asked rather than
    assumed.
    """
    if language == "auto":
        return tuple({span for spans in SCRIPTS.values() for span in spans})
    return SCRIPTS.get(language, SCRIPTS["zh"])


def writes_something(text: str, language: str = "zh") -> bool:
    """Whether the transcript is made of characters this deployment could have said.

    Asked to read a room with a fan in it SenseVoice does not answer "I do not
    know", it writes 그. or うん。. Gate on what it wrote, not on the language it
    reported: the reported language is wrong for 8% of ordinary short Mandarin,
    so a rule keyed on it deletes 咖啡 and 猫 along with the noise.

    Which characters count comes from what the deployment declared, and that is
    not decoration. Hardcoded to Chinese, this returned False for every Korean
    transcript and for every Japanese one written without kanji -- somebody
    spoke, the recogniser heard them, and the text box stayed empty. A Chinese
    deployment still refuses 그., because for it that really is invention.

    English inventions (I., The.) get through here; ``said_enough`` takes most
    of what is left. Readings in headline_numbers.json under dictation_entry_gate.
    """
    return bool(spoken_body(text, language))


def spoken_body(text: str, language: str = "zh") -> str:
    """The transcript with the punctuation a listener would not have said stripped."""
    body = text.strip(" .。,，!！?？、;；:：\n")
    spans = spoken_scripts(language)
    if not any(
        (char.isascii() and char.isalnum()) or any(low <= char <= high for low, high in spans)
        for char in body
    ):
        return ""
    return body


def said_enough(text: str, seconds: float, language: str = "zh") -> bool:
    """Whether this many characters in this much speech is a person talking.

    Steady noise makes the recogniser write one or two Chinese characters
    however long the buffer is, so ``writes_something`` passes it and speaking
    rate is what separates them. The floor is a third below the slowest real
    utterance measured.

    ``seconds`` must be how long somebody was talking, not how long the buffer
    is -- pass the buffer and every short command with a held key is deleted.
    Under a second is exempt: the rate of a fraction of a second is not a rate.
    """
    if seconds < 1.0:
        return True
    return len(spoken_body(text, language)) / seconds >= 0.5


@dataclass(slots=True)
class SenseVoiceRecogniser:
    """The product's recogniser. Not eligible to score the product."""

    model_dir: Path
    device: str = "cpu"
    # What this deployment is spoken in, not what to pass the model. Reaches
    # the model only below SHORT_AUDIO_SECONDS: the language slot is an
    # embedding prepended to the encoder input, so it perturbs decoding even
    # when the detector already agreed, and on long audio that costs more than
    # it buys.
    language: str = "zh"
    _model: Any = None
    # How many forward passes may be in flight. Not one: "the driver serialises
    # the kernels so the second buys nothing" was measured and is wrong -- on an
    # idle 4090, going from one to unlimited took throughput up 27% at two
    # concurrent and 17% at four, because feature extraction and the Python
    # around the pass do overlap even when the kernels do not.
    #
    # What the bound buys is the peak: unlimited reached 2356 MiB against 1438,
    # and the tail at eight concurrent was worse (P95 1046 ms against 944).
    # Two is where the throughput is won and the memory is not: 22-second audio
    # went 1086 to 1172 MiB. Held around the model call rather than the whole
    # request, so a short dictation waits for one piece of a long one.
    in_flight: int = 2
    _passes: threading.Semaphore = field(init=False, default=None)  # type: ignore[assignment]

    # Read by the evaluation harness, which refuses a recogniser that shares
    # lineage with the model under test.
    lineage: str = "sensevoice-small"
    eligible_as_judge: bool = False

    def __post_init__(self) -> None:
        self._passes = threading.Semaphore(self.in_flight)

    def load(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel

        # Kaldi-style feature extraction adds dither -- gaussian noise on the
        # waveform, which exists so a digitally silent frame does not take
        # log(0). At inference it makes the recogniser answer differently every
        # time. Driving the running service, the same 74-second recording came
        # back as six different transcripts in six requests, and at 164 seconds
        # as six again; under 25 seconds it looks stable only because a shorter
        # recording gives the noise fewer frames in which to flip a decision.
        # A dictation tool that returns different text for the same recording
        # is a bug the user sees and cannot explain.
        #
        # Set at construction, not after: funasr builds the frontend from this
        # config inside AutoModel, and assigning to the built object (or
        # passing dither to generate) is silently replaced. Measured both ways
        # -- fbank still received 1.0 -- so this is the only place it takes.
        conf = dict(self._frontend_conf())
        conf["dither"] = 0.0
        self._model = AutoModel(
            model=str(self.model_dir),
            device=self.device,
            disable_update=True,
            frontend_conf=conf,
        )

    def _frontend_conf(self) -> dict[str, object]:
        """The model's own frontend settings, read from the checkpoint's config.

        Read rather than restated so that a checkpoint with different framing
        keeps it. Only ``dither`` is ours to change, and the fallback below is
        the SenseVoice default for the case where the config cannot be parsed.
        """
        import yaml

        for name in ("config.yaml", "configuration.json"):
            path = self.model_dir / name
            if not path.is_file():
                continue
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a config we cannot read is not fatal
                continue
            conf = (loaded or {}).get("frontend_conf")
            if isinstance(conf, dict):
                conf = dict(conf)
                # Overwritten, not defaulted: the config records the path on
                # the machine that trained it, which is not this one.
                cmvn = self.model_dir / "am.mvn"
                conf["cmvn_file"] = str(cmvn) if cmvn.is_file() else conf.get("cmvn_file")
                return conf
        return {"fs": 16000, "window": "hamming", "n_mels": 80, "lfr_m": 7, "lfr_n": 6}

    def _read_once(self, audio: Any, language: str) -> Any:
        """One call into funasr, giving the card back if the call runs out of room.

        Out of memory is not the end of it: the exception keeps a traceback,
        the traceback keeps every frame's locals, and the locals keep the
        encoder's activations, so the block stays reserved after the request
        that failed has gone. Measured: reserved sat at 1782 MiB against a 986
        MiB baseline, and ``empty_cache`` on its own returned none of it,
        because the traceback still held the tensors. Dropping the traceback
        first is what makes the release work -- and it costs the caller nothing,
        since the message rather than the frames is what reaches them.
        """
        try:
            return self._model.generate(input=audio, cache={}, language=language, use_itn=True)
        except Exception as error:  # noqa: BLE001 - re-raised; this only hands the card back
            import torch

            if isinstance(error, torch.OutOfMemoryError):
                error.__traceback__ = None
                torch.cuda.empty_cache()
            raise

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        # On a thread: funasr is synchronous and holds the GIL for the length
        # of the utterance, so run inline it stops every other session and the
        # health check along with them.
        import asyncio

        return await asyncio.to_thread(self._transcribe, pcm, sample_rate)

    def _transcribe(self, pcm: bytes, sample_rate: int = RECOGNISER_RATE) -> tuple[str, str | None]:
        from mindsurf_omni.service.audio import resample, unwrap_wav, whole_samples
        from mindsurf_omni.service.vad import frame_energy

        # A rate declared in the body's own header beats one the caller passes
        # out of band: a client that gets one wrong usually gets both wrong, and
        # the wrong rate produces a transcript that reads like Chinese and says
        # something else.
        pcm, sample_rate = unwrap_wav(pcm, sample_rate)
        if sample_rate != RECOGNISER_RATE:
            pcm = resample(pcm, sample_rate, RECOGNISER_RATE)

        # Silence in, nothing out. Asked to read a room with nobody in it
        # SenseVoice invents rather than returning empty. Here rather than at
        # the endpoint because every path in has the same problem.
        if frame_energy(pcm) < SILENCE_RMS:
            return "", None

        samples = whole_samples(pcm)
        if len(samples) // 2 < SHORTEST_SAMPLES:
            return "", None

        # Below the silence check on purpose: telling somebody to split a
        # recording of nothing is advice they cannot use. The actual length
        # keeps a decimal so that a request just over the line does not read as
        # "this is 3600, the limit is 3600".
        seconds = len(samples) / 2 / RECOGNISER_RATE
        if seconds > LONGEST_SECONDS:
            raise TooLongForModel(
                f"this recording is {seconds:.1f} seconds, past the {LONGEST_SECONDS:.0f} "
                "this endpoint takes in one request; send it in pieces"
            )

        self.load()
        # Judged once over the whole recording, not once per piece. Splitting
        # moved this from one call to one per piece, and a piece that is a
        # pause for thought, a phone number or a URL is slow enough on its own
        # to be thrown away -- measured, 4 characters of 145 disappeared with
        # nothing in the log. The fast part of a dictation is what vouches for
        # the slow part, which is what it did before the split.
        voiced = voiced_seconds(samples, RECOGNISER_RATE)
        # Also once, for the same reason: the language prior exists for audio
        # too short for the detector, and a long recording whose last piece is
        # short is not that.
        short = seconds < SHORT_AUDIO_SECONDS

        if seconds <= SEGMENT_ABOVE_SECONDS:
            text, heard = self._read(samples, short)
            return (text, heard) if said_enough(text, voiced, self.language) else ("", None)

        # Cut at the pauses rather than fed whole. The encoder's allocation
        # grows with the square of the input, and so does what it gets wrong:
        # fifteen minutes in one pass wrote 3409 characters where 4243 were
        # spoken. Per piece both are flat, and every piece but the last is work
        # a caller streaming its audio can finish before the key comes up.
        # The detector answering "no speech here" is not the same answer as the
        # silence check above, which this buffer already passed: a quiet
        # recording with a flat envelope reads as nothing to a detector working
        # off a percentile of its own frames. Falling back to fixed windows
        # gives up the good seams rather than the recording.
        cuts = segments(samples, RECOGNISER_RATE) or [
            (start, min(start + FALLBACK_WINDOW, len(samples)))
            for start in range(0, len(samples), FALLBACK_WINDOW)
        ]

        parts: list[str] = []
        languages: list[str] = []
        for start, end in cuts:
            text, heard = self._read(samples[start:end], short)
            if not text:
                continue
            parts.append(text)
            if heard:
                languages.append(heard)
        # Joined with nothing: use_itn ends each piece with its own punctuation,
        # and a separator here would show up as a space in the text box.
        joined = "".join(parts)
        if not said_enough(joined, voiced, self.language):
            return "", None
        return joined, max(set(languages), key=languages.count) if languages else None

    def _read(self, samples: bytes, short: bool) -> tuple[str, str | None]:
        """One forward pass over one piece, with the guards that piece has to pass.

        ``short`` is about the whole recording, not this piece. With too little
        evidence the model does not answer "I do not know", it invents, so
        below SHORT_AUDIO_SECONDS it is given the deployment's language instead
        of a guess -- and the last piece of a long recording being short is not
        the same situation.
        """
        import numpy as np

        if len(samples) // 2 < SHORTEST_SAMPLES:
            return "", None

        audio = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
        language = self.language if short else "auto"
        with self._passes:
            result = self._read_once(audio, language)
        if not result:
            return "", None
        text, heard = strip_tags(str(result[0].get("text", "")))
        if heard == "nospeech" or not writes_something(text, self.language):
            return "", None
        return text, heard


@dataclass(slots=True)
class WhisperRecogniser:
    """The judge. Independent lineage, so its errors do not cancel ours."""

    model_name: str = "large-v3"
    device: str = "cpu"
    _model: Any = None

    lineage: str = "openai-whisper"
    eligible_as_judge: bool = True

    def load(self) -> None:
        if self._model is not None:
            return
        import whisper

        self._model = whisper.load_model(self.model_name, device=self.device)

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        import numpy as np

        from mindsurf_omni.service.audio import whole_samples

        self.load()
        audio = np.frombuffer(whole_samples(pcm), dtype=np.int16).astype(np.float32) / 32768.0
        result = self._model.transcribe(audio, fp16=False)
        return str(result.get("text", "")).strip(), result.get("language")


def require_independent_judge(recogniser: Any, model_lineage: str) -> None:
    """Refuse a recogniser that shares lineage with what it is scoring.

    Enforced rather than documented: the pretraining phase learned that a rule
    only written down is a rule that gets forgotten under deadline.
    """
    if not getattr(recogniser, "eligible_as_judge", False):
        raise ValueError(
            f"{getattr(recogniser, 'lineage', recogniser)!r} is a service component, "
            "not an independent judge; scoring a model with its own parts is circular"
        )
    if getattr(recogniser, "lineage", None) == model_lineage:
        raise ValueError(
            f"the recogniser and the model under test share lineage ({model_lineage!r}), "
            "so their failure modes cancel"
        )


# FunASR's streaming configuration, and the one the field quotes: a 600 ms
# chunk with 300 ms of lookahead. The three numbers are chunks of 60 ms --
# [left context, chunk, right context] -- so [0, 10, 5] is no left context, a
# ten-chunk window, five chunks of the future.
STREAMING_CHUNK = (0, 10, 5)
# Samples per step at 16 kHz: 10 chunks of 60 ms.
STREAMING_STRIDE = 9600


@dataclass(slots=True)
class ParaformerStreamingRecogniser:
    """A recogniser that writes while the speaker is still talking.

    SenseVoice cannot do this and it is not a tuning problem: it is a
    non-autoregressive whole-segment model with no chunk parameter, and feeding
    it the audio in pieces as their pauses close was measured at 0.4546 CER
    against 0.0866 for the same audio in one pass -- one or two seconds is not
    enough context for it to read the speech at all.

    This one is built for it. Measured over 30 real dictations: the first word
    lands 0.6 s in, each 600 ms step costs 194 ms (RTF 0.323), and the whole
    utterance reads 0.2796 against SenseVoice's 0.1094.

    That gap is the reason this is a choice and not a replacement. Where the
    stream only drives the display and the release still goes through
    SenseVoice, the gap never reaches the user's text box; where a deployment
    picks this as its only recogniser, it does.
    """

    model_dir: Path | str = "paraformer-zh-streaming"
    device: str = "cpu"
    language: str = "zh"
    chunk: tuple[int, int, int] = STREAMING_CHUNK
    _model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel

        self._model = AutoModel(
            model=str(self.model_dir),
            device=self.device,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        """The whole utterance, so this satisfies the same protocol as the other one."""

        pieces = [text async for text in self.stream(pcm, sample_rate)]
        return "".join(pieces).strip(), (self.language if self.language != "auto" else None)

    async def stream(self, pcm: bytes, sample_rate: int) -> AsyncIterator[str]:
        """Text as it is decided, one 600 ms step at a time.

        Yields only what each step added, never a rewritten prefix: this model
        commits as it goes, so a caller can append rather than re-render. That
        is what makes it usable for a live display without the text jumping
        around -- the thing every revision policy in the streaming literature
        exists to manage, and which we get by not needing one.
        """
        import asyncio

        import numpy as np

        self.load()
        from mindsurf_omni.service.audio import resample, whole_samples

        if sample_rate != RECOGNISER_RATE:
            pcm = resample(whole_samples(pcm), sample_rate, RECOGNISER_RATE)
        audio = np.frombuffer(whole_samples(pcm), dtype=np.int16).astype(np.float32) / 32768.0
        if not audio.size:
            return

        cache: dict[str, Any] = {}
        steps = (audio.size - 1) // STREAMING_STRIDE + 1
        for index in range(steps):
            piece = audio[index * STREAMING_STRIDE : (index + 1) * STREAMING_STRIDE]
            if not piece.size:
                continue
            said = await asyncio.to_thread(self._step, piece, cache, index == steps - 1)
            if said:
                yield said

    def _step(self, piece: Any, cache: dict[str, Any], last: bool) -> str:
        out = self._model.generate(
            input=piece,
            cache=cache,
            is_final=last,
            chunk_size=list(self.chunk),
            encoder_chunk_look_back=self.chunk[0],
            decoder_chunk_look_back=self.chunk[2],
        )
        return str(out[0]["text"]) if out else ""
