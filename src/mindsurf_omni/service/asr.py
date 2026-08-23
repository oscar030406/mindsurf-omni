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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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

# One fbank frame. Below it the frontend asserts ("choose a window size 1"),
# which reached the caller as a bare 500, and between one frame and about a
# tenth of a second the model only ever writes punctuation.
SHORTEST_SAMPLES = 400


def writes_something(text: str) -> bool:
    """Whether the transcript is made of characters this deployment could have said.

    Asked to read a room with a fan in it, SenseVoice does not answer "I do not
    know" -- it writes 그. or うん。 and reports 0.99 confidence in Korean. The
    reported language is the wrong thing to gate on: measured over 48 short
    spoken commands it calls 8% of ordinary Mandarin something other than
    Chinese, so a rule keyed on it deletes 咖啡 and 猫 along with the noise.

    This asks instead what the model actually wrote. Hangul and kana carry no
    Chinese and no Latin word characters, so an output made only of them is the
    recogniser inventing; 咖啡, demo and B203 all survive. Measured on 36
    synthetic noise clips it catches 11 with no correct transcript lost.

    The English inventions (I., The.) survive this; ``said_enough`` catches most
    of what is left, and separating the rest needs the language tag.
    """
    return bool(spoken_body(text))


def spoken_body(text: str) -> str:
    """The transcript with the punctuation a listener would not have said stripped."""
    body = text.strip(" .。,，!！?？、;；:：\n")
    if not any("一" <= char <= "鿿" or (char.isascii() and char.isalnum()) for char in body):
        return ""
    return body


def said_enough(text: str, seconds: float) -> bool:
    """Whether this many characters in this many seconds is a person talking.

    Steady noise makes the recogniser write one or two characters however long
    the buffer is -- three seconds of mains hum comes back as 我. -- and the
    character is often Chinese, so ``writes_something`` passes it. Speaking rate
    separates the two where script and reported language do not: measured over
    60 noise clips, 48 short spoken commands and 40 sentences, noise sits at a
    median of 0.33 characters per second against 1.35 and 4.52, and the slowest
    real utterance is 0.67.

    The floor is half a character per second, which is a third below that
    slowest real one: it drops 33 of the 60 noise clips and none of the 88
    spoken ones. Buffers under a second are exempt -- a single word is over the
    line anyway, and the rate of a fraction of a second is not a rate.
    """
    if seconds < 1.0:
        return True
    return len(spoken_body(text)) / seconds >= 0.5


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

    # Read by the evaluation harness, which refuses a recogniser that shares
    # lineage with the model under test.
    lineage: str = "sensevoice-small"
    eligible_as_judge: bool = False

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

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        # On a thread, because funasr is synchronous and holds the GIL for the
        # length of the utterance. Run inline it stops the event loop, which on
        # a live service is not "this request is slow": every other session
        # stops receiving, the heartbeat stops ticking, and health checks stop
        # answering. Measured at 94 ms for a short turn and minutes for a long
        # one, and the long one is what a stuck microphone sends.
        import asyncio

        return await asyncio.to_thread(self._transcribe, pcm)

    def _transcribe(self, pcm: bytes) -> tuple[str, str | None]:
        import numpy as np

        from mindsurf_omni.service.audio import whole_samples
        from mindsurf_omni.service.vad import frame_energy

        # Silence in, nothing out. Asked to read a room with nobody in it,
        # SenseVoice does not return empty -- it invents. Measured on three
        # seconds of digital silence: it returned the Korean "그.", the model
        # answered that in English, and the caller heard a 9.6-second reply to
        # a button they pressed by accident. The recogniser is where this
        # belongs: every path into it has the same problem.
        if frame_energy(pcm) < SILENCE_RMS:
            return "", None

        samples = whole_samples(pcm)
        if len(samples) // 2 < SHORTEST_SAMPLES:
            return "", None

        self.load()
        audio = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
        # Same shape as the silence check above: with too little evidence the
        # model does not answer "I do not know", it invents. Silence has a
        # right answer, so that one returns empty; this one does not, so it is
        # given the deployment's language instead of a guess.
        short = len(audio) < SHORT_AUDIO_SECONDS * RECOGNISER_RATE
        language = self.language if short else "auto"
        result = self._model.generate(input=audio, cache={}, language=language, use_itn=True)
        if not result:
            return "", None
        text, heard = strip_tags(str(result[0].get("text", "")))
        seconds = len(audio) / RECOGNISER_RATE
        if heard == "nospeech" or not writes_something(text) or not said_enough(text, seconds):
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
