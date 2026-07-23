"""Assemble an engine from settings, or explain what is missing.

This is the one place that knows how the pieces fit together, so the app stays
ignorant of which path it is serving and the engines stay ignorant of where
their weights came from.

Loading is deferred until the first request rather than done at import: a
container that cannot reach its weights should start, answer 503 with the
reason, and stay up for a health check -- not crash-loop while an operator
tries to read the log.
"""

from __future__ import annotations

from typing import Any

from mindsurf_omni.service.config import (
    ConfigurationError,
    Settings,
    describe_components,
    token_spec,
)
from mindsurf_omni.service.engine import SpeechEngine


def build(settings: Settings | None) -> SpeechEngine | None:
    """Build the configured engine, or None when none was requested."""
    if settings is None:
        return None
    settings.verify()

    if settings.path == "cascade":
        return _build_cascade(settings)
    return _build_native(settings)


def _build_cascade(settings: Settings) -> SpeechEngine:
    from mindsurf_omni.service.asr import SenseVoiceRecogniser
    from mindsurf_omni.service.cascade import CascadeEngine

    recogniser = SenseVoiceRecogniser(
        model_dir=settings.paths.audio_encoder, device=settings.device
    )

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return await recogniser.transcribe(pcm, sample_rate)

    async def generate(messages: list[dict[str, str]], _: Any) -> Any:
        raise ConfigurationError(
            "the cascade path has no text generator wired yet; "
            "the Thinker checkpoint is still training"
        )
        yield ""  # pragma: no cover - makes this an async generator

    async def synthesise(text: str, _: Any) -> bytes:
        raise ConfigurationError(
            "the cascade path has no synthesiser wired yet; set MINDSURF_TTS once one is available"
        )

    return CascadeEngine(
        transcriber=transcribe,
        generator=generate,  # type: ignore[arg-type]
        synthesiser=synthesise,
        components=describe_components(settings),
        token_spec=token_spec(),
    )


def _build_native(settings: Settings) -> SpeechEngine:
    raise ConfigurationError(
        "the native path needs the Thinker-Talker checkpoint, which is still "
        "training; use MINDSURF_ENGINE=cascade until it lands"
    )
