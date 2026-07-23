"""Build the configured engine, or say precisely why it cannot be built.

The container starts with an environment and a mounted weights directory, and
one of three things is true: the native path is ready, the cascade path is
ready, or neither is. The first two must work; the third must produce a message
someone can act on without reading this file.

So every failure names the variable that was missing or the path that did not
exist. "Engine unavailable" tells an operator nothing at three in the morning.
"""

from __future__ import annotations

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
        )

    def verify(self) -> None:
        missing = self.paths.missing()
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


def describe_components(settings: Settings) -> list[ComponentInfo]:
    """What is loaded, with the frozen pieces marked as such.

    Which components are frozen decides what a result can be attributed to, so
    it belongs in the response rather than in a document beside it.
    """
    components = [
        ComponentInfo(name="thinker", parameters=89_864_448, frozen=False),
    ]
    if settings.path == "native":
        components += [
            ComponentInfo(name="talker", frozen=False),
            ComponentInfo(name="mimi-codec", frozen=True),
        ]
    components.append(ComponentInfo(name="sensevoice-small", parameters=234_000_000, frozen=True))
    return components
