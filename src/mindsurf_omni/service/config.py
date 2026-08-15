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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mindsurf_omni.contract import ComponentInfo, TokenSpec

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

    def missing(self) -> list[str]:
        return [
            f"{name}={value}"
            for name, value in (
                ("weights", self.weights),
                ("tokenizer", self.tokenizer),
                ("audio_encoder", self.audio_encoder),
                ("codec", self.codec),
                ("speaker", self.speaker),
            )
            if not value.exists()
        ]


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
        )

    def verify(self) -> None:
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
            missing = [*missing, f"thinker={self.thinker}"]
        # The cascade's Thinker loads lazily, so a mistyped checkout used to
        # surface as the connection dropping on the first thing a caller said.
        # The native path already refuses this at assembly, with a message
        # about the checkout specifically -- leave that one to it.
        if self.path == "cascade" and self.minimind_root is not None:
            definition = self.minimind_root / "model" / "model_minimind.py"
            if not definition.is_file():
                missing = [*missing, f"minimind_root={self.minimind_root} (no {definition.name})"]
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
            missing = [*missing, f"tts_prompt_wav={self.tts_prompt_wav}"]
        # Same reason as the Thinker's: a mistyped path would otherwise surface
        # as the first dictation coming back unpolished, which reads as the
        # model doing nothing rather than as a path that is not there.
        if self.polish is not None and not self.polish.is_file():
            missing = [*missing, f"polish={self.polish}"]
        if missing:
            raise ConfigurationError(
                f"the {self.path} path needs these, and they are not on disk: "
                + ", ".join(missing)
                + " -- mount the weights directory or set the matching variable"
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
