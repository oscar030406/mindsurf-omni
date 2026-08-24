"""Build the configured engine, or say precisely why it cannot be built.

The container starts with an environment and a mounted weights directory, and
one of three things is true: the native path is ready, the cascade path is
ready, or neither is. The first two must work; the third must produce a message
someone can act on without reading this file.

So every failure names the variable that was missing or the path that did not
exist. "Engine unavailable" tells an operator nothing at three in the morning.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mindsurf_omni.contract import ComponentInfo, TokenSpec

_log = logging.getLogger("mindsurf.config")

PathName = Literal["native", "cascade"]

# Ids the tokenizer already reserves, so the client never has to be told them
# separately and they cannot drift from the weights.
SPECIAL_TOKENS = {
    "endoftext": 0,
    "im_start": 1,
    "im_end": 2,
    "image_pad": 12,
    "audio_start": 14,
    "audio_end": 15,
    "audio_pad": 16,
}
# Above the 2048 Mimi codes, in the space the codebook leaves free.
AUDIO_SPECIAL_TOKENS = {"pad": 2049, "stop": 2050, "spk": 2051}


class ConfigurationError(RuntimeError):
    """Raised with the specific missing thing, never a generic failure."""


@dataclass(frozen=True, slots=True)
class Paths:
    weights: Path
    tokenizer: Path
    audio_encoder: Path
    codec: Path
    speaker: Path

    def missing(self) -> list[tuple[str, Path]]:
        """The variables whose path is not there, each with the path it pointed at.

        Both, because the two halves go to different readers: the operator
        needs the path, and the message that reaches an unauthenticated caller
        must not carry it. See ``Settings.verify``.
        """
        return [
            (name, value)
            for name, value in (
                ("MINDSURF_WEIGHTS", self.weights),
                ("MINDSURF_TOKENIZER", self.tokenizer),
                ("MINDSURF_ASR", self.audio_encoder),
                ("MINDSURF_CODEC", self.codec),
                ("MINDSURF_SPEAKER", self.speaker),
            )
            if not value.exists()
        ]


def _spoken_language(value: str) -> str:
    """The declared language, refused at startup if the model cannot read it.

    funasr takes an unknown language silently as "auto" -- no warning, no
    error -- so MINDSURF_ASR_LANGUAGE=Chinese would look configured and behave
    unconfigured. Same rule as a weights path that does not exist: say so now
    rather than at the first request.
    """
    from mindsurf_omni.service.asr import SPOKEN_LANGUAGES

    declared = value.strip()
    if declared not in SPOKEN_LANGUAGES:
        raise ConfigurationError(
            f"MINDSURF_ASR_LANGUAGE={value!r} is not one of {', '.join(SPOKEN_LANGUAGES)}"
        )
    return declared


@dataclass(frozen=True, slots=True)
class Settings:
    path: PathName
    paths: Paths
    device: str = "cpu"
    chunk_frames: int = 4
    # Which synthesiser the cascade speaks with. Empty is the honest default:
    # the path answers with the reason it cannot speak rather than quietly
    # reaching a hosted endpoint nobody asked it to reach.
    tts: str = ""
    # The Thinker checkpoint the cascade answers from, and the MiniMind-O
    # checkout its model class is built from. Both unset until a checkpoint
    # exists, at which point the text stage stops being the unwired one.
    thinker: Path | None = None
    minimind_root: Path | None = None
    # The clip the local synthesiser clones, and what it says. Unset, VoxCPM
    # draws a speaker per call, so a reply changes voice between clauses -- the
    # "怪声" the fourth meeting raised. The knob exists in the synthesiser
    # already; what was missing was a way for an operator to reach it.
    tts_prompt_wav: Path | None = None
    tts_prompt_text: str | None = None
    # The dictation path's polish stage: a Thinker fine-tuned to delete spoken
    # filler. Unset means the service transcribes and stops there, which is what
    # the conversation product wants.
    polish: Path | None = None
    # The dictation path's second arm and the backbone it probes. Unset is the
    # shape this had before 2026-08-16: generate and tidy. Set, the two arms are
    # merged by veto, which on 986 held-out transcripts reads better than the
    # generator alone on four of five numbers and level on the fifth.
    polish_tagger: Path | None = None
    polish_tagger_backbone: Path | None = None
    # How sure the second arm has to be before it asks for a character.
    #
    # 0.5 was the value it was first measured at, never the value it was chosen
    # at. Swept on the deployment card over 986 held-out transcripts, every
    # step below it reads better and the curve is flat where it matters:
    #
    #   0.30  CER 0.0275  clearance 0.9733  retention 0.9796  invention 0.0197
    #   0.35  CER 0.0272  clearance 0.9693  retention 0.9800  invention 0.0200
    #   0.40  CER 0.0273  clearance 0.9625  retention 0.9805  invention 0.0204
    #   0.45  CER 0.0274  clearance 0.9574  retention 0.9808  invention 0.0209
    #   0.50  CER 0.0282  clearance 0.9517  retention 0.9809  invention 0.0218
    #   0.60  CER 0.0289  clearance 0.9353  retention 0.9814  invention 0.0230
    #
    # 0.35 to 0.45 are one number apart on CER, which is noise; the real signal
    # is that anything under 0.5 beats 0.5. Within the flat stretch the choice
    # is clearance against retention, and this project's standing rule is that
    # damaged content costs more than leftover filler -- 0.35 puts retention on
    # its 0.98 line and invention on its 0.02 line, 0.40 leaves both a margin
    # for the same CER. Paired bootstrap against 0.5, 4000 draws: CER -0.0009
    # [-0.0016, -0.0002], clearance +0.0108 [+0.0062, +0.0161], invention
    # -0.0013 [-0.0020, -0.0007], retention -0.0004 [-0.0008, -0.0001] -- the
    # loss is real and is a fiftieth of that criterion's margin.
    polish_tagger_threshold: float = 0.4
    # What this deployment is spoken in. Reaches the recogniser only for audio
    # too short for its own detector -- see asr.SHORT_AUDIO_SECONDS. "auto"
    # restores the behaviour this had before 2026-08-22 at every length.
    asr_language: str = "zh"
    # Which recogniser this deployment serves. "sensevoice" is the one the
    # polisher was trained against and the one the released numbers describe;
    # "paraformer-streaming" writes while the speaker is still talking, which
    # SenseVoice structurally cannot, and reads 0.2796 against its 0.1094.
    #
    # A choice rather than a replacement, because which one is right depends on
    # what the deployment is for: a live display wants the words early, a text
    # box wants them right.
    asr_model: str = "sensevoice"
    # Proper nouns this deployment says and the recogniser does not know, put
    # back into the transcript by sound. Global rather than per request: the
    # dictation endpoint takes a bare PCM body with no field to carry a table,
    # and one deployment serves one vocabulary. See hotwords.py for what an
    # entry has to survive to be taken.
    hotwords: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls, environment: dict[str, str] | None = None) -> Settings | None:
        """Read settings, or None when no engine was requested.

        None is not an error. A service started without an engine is a valid
        state -- it answers 503 with a reason, which is what lets the backend
        integrate before the model exists.
        """
        source = environment if environment is not None else dict(os.environ)
        requested = source.get("MINDSURF_ENGINE", "").strip().lower()
        if not requested:
            return None
        if requested not in {"native", "cascade"}:
            raise ConfigurationError(f"MINDSURF_ENGINE={requested!r} is not 'native' or 'cascade'")

        root = Path(source.get("MINDSURF_WEIGHTS", "/app/weights"))
        return cls(
            path=requested,  # type: ignore[arg-type]
            paths=Paths(
                weights=root,
                tokenizer=Path(source.get("MINDSURF_TOKENIZER", root / "tokenizer")),
                audio_encoder=Path(source.get("MINDSURF_ASR", root / "SenseVoiceSmall")),
                codec=Path(source.get("MINDSURF_CODEC", root / "mimi")),
                speaker=Path(source.get("MINDSURF_SPEAKER", root / "campplus")),
            ),
            device=source.get("MINDSURF_DEVICE", "cpu"),
            chunk_frames=int(source.get("MINDSURF_CHUNK_FRAMES", "4")),
            tts=source.get("MINDSURF_TTS", "").strip().lower(),
            thinker=Path(source["MINDSURF_THINKER"]) if source.get("MINDSURF_THINKER") else None,
            minimind_root=(
                Path(source["MINIMIND_O_ROOT"]) if source.get("MINIMIND_O_ROOT") else None
            ),
            tts_prompt_wav=(
                Path(source["MINDSURF_TTS_PROMPT_WAV"])
                if source.get("MINDSURF_TTS_PROMPT_WAV")
                else None
            ),
            tts_prompt_text=(source.get("MINDSURF_TTS_PROMPT_TEXT") or "").strip() or None,
            polish=Path(source["MINDSURF_POLISH"]) if source.get("MINDSURF_POLISH") else None,
            polish_tagger=(
                Path(source["MINDSURF_POLISH_TAGGER"])
                if source.get("MINDSURF_POLISH_TAGGER")
                else None
            ),
            polish_tagger_backbone=(
                Path(source["MINDSURF_POLISH_TAGGER_BACKBONE"])
                if source.get("MINDSURF_POLISH_TAGGER_BACKBONE")
                else None
            ),
            polish_tagger_threshold=float(source.get("MINDSURF_POLISH_TAGGER_THRESHOLD", "0.4")),
            asr_language=_spoken_language(source.get("MINDSURF_ASR_LANGUAGE", "zh")),
            asr_model=source.get("MINDSURF_ASR_MODEL", "sensevoice").strip().lower(),
            hotwords=tuple(
                word.strip()
                for word in source.get("MINDSURF_HOTWORDS", "").split(",")
                if word.strip()
            ),
        )

    #: The recognisers this build can serve, for /v1/models and for refusing a
    #: name nobody has wired rather than starting and failing on first speech.
    RECOGNISERS = ("sensevoice", "paraformer-streaming")

    def verify(self) -> None:
        if self.asr_model not in self.RECOGNISERS:
            raise ConfigurationError(
                f"MINDSURF_ASR_MODEL={self.asr_model!r} names no recogniser this build has; "
                f"it has {', '.join(self.RECOGNISERS)}"
            )
        missing = self.paths.missing()
        # A checkpoint that was named and is not there used to pass this check:
        # assembly only asked whether the variable was set. The service then
        # started, reported ready, and raised FileNotFoundError on the first
        # thing a user said -- which reads as the model failing rather than as
        # a typo in a path.
        # is_file, not exists: a directory at the checkpoint's path passes the
        # weaker check, and then the digest in /v1/models is null -- which is
        # the field that says which weights a measurement was taken on.
        if self.thinker is not None and not self.thinker.is_file():
            missing = [*missing, ("MINDSURF_THINKER", self.thinker)]
        # The cascade's Thinker loads lazily, so a mistyped checkout used to
        # surface as the connection dropping on the first thing a caller said.
        # The native path already refuses this at assembly, with a message
        # about the checkout specifically -- leave that one to it.
        if self.path == "cascade" and self.minimind_root is not None:
            definition = self.minimind_root / "model" / "model_minimind.py"
            if not definition.is_file():
                missing = [*missing, (f"MINIMIND_O_ROOT (no {definition.name})", definition)]
        # Both or neither. VoxCPM rejects a clip without its text, and it is
        # right to -- the text is what tells it which sounds belong to which
        # characters. Refused here rather than at the first request, because
        # half a reference is silently ignored by the model that receives it
        # and the service would then run the very failure this fixes.
        if (self.tts_prompt_wav is None) != (self.tts_prompt_text is None):
            raise ConfigurationError(
                "MINDSURF_TTS_PROMPT_WAV and MINDSURF_TTS_PROMPT_TEXT go together; "
                f"only {'the clip' if self.tts_prompt_wav else 'the text'} was set, and a "
                "clip without its text clones nothing"
            )
        if self.tts_prompt_wav is not None and not self.tts_prompt_wav.is_file():
            missing = [*missing, ("MINDSURF_TTS_PROMPT_WAV", self.tts_prompt_wav)]
        # Same reason as the Thinker's: a mistyped path would otherwise surface
        # as the first dictation coming back unpolished, which reads as the
        # model doing nothing rather than as a path that is not there.
        if self.polish is not None and not self.polish.is_file():
            missing = [*missing, ("MINDSURF_POLISH", self.polish)]
        # Both or neither, and only alongside a polisher. The head is a probe of
        # the blocks tuned with it, so half the pair measures nothing -- and a
        # tagger without a generator has no arm to be merged with.
        if (self.polish_tagger is None) != (self.polish_tagger_backbone is None):
            raise ConfigurationError(
                "MINDSURF_POLISH_TAGGER and MINDSURF_POLISH_TAGGER_BACKBONE go together; "
                f"only {'the head' if self.polish_tagger else 'the backbone'} was set, and "
                "the head is a probe of the blocks that were tuned with it"
            )
        if self.polish_tagger is not None and self.polish is None:
            raise ConfigurationError(
                "MINDSURF_POLISH_TAGGER is set but MINDSURF_POLISH is not; the tagger is "
                "the second arm of a merge and has no first arm to be merged with"
            )
        for name, value in (
            ("MINDSURF_POLISH_TAGGER", self.polish_tagger),
            ("MINDSURF_POLISH_TAGGER_BACKBONE", self.polish_tagger_backbone),
        ):
            if value is not None and not value.is_file():
                missing = [*missing, (name, value)]
        # Only the cascade builds one. Set on the native path it did nothing at
        # all, and worse, it did nothing loudly: /health answered
        # "polisher: ready" and /v1/models carried its sha256, while
        # /v1/audio/transcriptions returned polished=null on every request. An
        # operator reading the health check had no way to see that the stage
        # they configured was not there.
        # Only the cascade transcribes through a stage that can be wrapped.
        # Same failure as the polisher's below: set on the native path it would
        # be configured, reported, and inert.
        if self.path == "native" and self.hotwords:
            raise ConfigurationError(
                "MINDSURF_HOTWORDS is set but MINDSURF_ENGINE=native, and the native path "
                "transcribes inside the model rather than through a stage this can correct; "
                "run the dictation product on MINDSURF_ENGINE=cascade"
            )
        if self.path == "native" and self.polish is not None:
            raise ConfigurationError(
                "MINDSURF_POLISH is set but MINDSURF_ENGINE=native, and the native path has "
                "no polish stage; run the dictation product on MINDSURF_ENGINE=cascade, or "
                "unset MINDSURF_POLISH to serve conversation from this checkpoint"
            )
        if missing:
            # The variables, not the paths they hold. This message is the body
            # of a 503 and the detail on /health, and neither endpoint asks for
            # a credential -- so a deployment whose weights live under a
            # customer's name was handing that name to anyone who could reach
            # the port. The variable is what an operator has to change anyway;
            # the value they set it to is something they already know.
            _log.error(
                "%s path is missing files: %s",
                self.path,
                ", ".join(f"{name}={value}" for name, value in missing),
            )
            raise ConfigurationError(
                f"the {self.path} path needs these, and the file each names is not on "
                "disk: " + ", ".join(name for name, _ in missing) + " -- mount the weights "
                "directory or set the matching variable; the paths themselves are in the log"
            )


def token_spec(vocab_size: int = 6400) -> TokenSpec:
    """The spec served to clients, built from the ids the tokenizer reserves."""
    return TokenSpec(
        text_vocab_size=vocab_size,
        audio_codebooks=8,
        audio_codebook_size=2048,
        audio_frame_rate_hz=12.5,
        special_tokens=dict(SPECIAL_TOKENS),
        audio_special_tokens=dict(AUDIO_SPECIAL_TOKENS),
    )


def checkpoint_digest(path: Path | None) -> str | None:
    """SHA-256 of the weights, or None when there are none to name.

    Read in chunks because the file is hundreds of megabytes, and computed once
    at assembly rather than per request. None means no checkpoint is
    configured, which is different from a checkpoint whose digest is unknown --
    the same distinction the licence record makes with null.
    """
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_components(settings: Settings) -> list[ComponentInfo]:
    """What is loaded, with the frozen pieces marked as such.

    Which components are frozen decides what a result can be attributed to, so
    it belongs in the response rather than in a document beside it.
    """
    components = []
    # Listed only when it exists, the same rule the polisher below follows.
    # Listing it unconditionally made /health answer "thinker: ready" beside
    # "generator: not wired" -- two statements about one stage, contradicting
    # each other, in one payload an operator is meant to act on. Nothing was
    # checking readiness; the entry was a constant.
    if settings.thinker is not None:
        components.append(
            ComponentInfo(
                name="thinker",
                parameters=89_864_448,
                # Which weights spoke. Without it two evaluation runs -- one on
                # the pretrained base, one on the fine-tuned checkpoint --
                # produce byte-identical provenance, and the report comparing
                # them is comparing two things it cannot name.
                sha256=checkpoint_digest(settings.thinker),
                frozen=False,
            )
        )
    if settings.path == "native":
        components += [
            ComponentInfo(name="talker", frozen=False),
            ComponentInfo(name="mimi-codec", frozen=True),
        ]
    components.append(ComponentInfo(name="sensevoice-small", parameters=234_000_000, frozen=True))
    if settings.polish is not None:
        # Named and hashed like the Thinker: the polish stage rewrites what the
        # user sees, so which weights did it belongs in the same list.
        components.append(
            ComponentInfo(
                name="polisher",
                parameters=89_864_448,
                sha256=checkpoint_digest(settings.polish),
                frozen=False,
            )
        )
    if settings.path == "cascade" and settings.tts:
        # Named, because CER measures whether the synthesiser said the reply.
        # A report that does not say which one spoke has not measured anything
        # that can be compared to the next report. The digest is the reference
        # clip when there is one: with a clone prompt the voice in every sample
        # comes from that file, so swapping it changes the audio a report is
        # about as surely as swapping the synthesiser does. Null means no
        # prompt, which for VoxCPM means a different speaker each call.
        components.append(
            ComponentInfo(
                name=f"tts-{settings.tts}",
                sha256=checkpoint_digest(settings.tts_prompt_wav),
                frozen=True,
            )
        )
    return components
