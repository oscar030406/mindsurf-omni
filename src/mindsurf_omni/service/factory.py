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

    synthesise = _build_synthesiser(settings)

    # Named here because assembly is the only place that knows which stages
    # will refuse. "generator" is unconditional until the Thinker lands.
    unwired = ["generator"] if settings.tts else ["generator", "synthesiser"]

    return CascadeEngine(
        transcriber=transcribe,
        generator=generate,  # type: ignore[arg-type]
        synthesiser=synthesise,
        components=describe_components(settings),
        token_spec=token_spec(),
        unwired=tuple(unwired),
    )


def _build_synthesiser(settings: Settings) -> Any:
    """The callable the cascade speaks with, or one that says why it cannot.

    Nothing is chosen by default. A synthesiser decides what the audio in every
    evaluation report actually is, so it is named by an operator rather than
    picked here -- and an unconfigured path that refuses is safer than one that
    silently reaches a hosted endpoint.
    """
    from mindsurf_omni.service.engine import GenerationSettings
    from mindsurf_omni.service.tts import EdgeSynthesiser, Utterance

    if not settings.tts:

        async def unwired(text: str, _: Any) -> bytes:
            raise ConfigurationError(
                "the cascade path has no synthesiser wired; set MINDSURF_TTS=edge for the "
                "hosted one, which reaches the network and is named in /v1/models"
            )

        return unwired

    if settings.tts != "edge":
        raise ConfigurationError(
            f"MINDSURF_TTS={settings.tts!r} names no synthesiser this build has; "
            "'edge' is wired, CosyVoice2 is not yet"
        )

    # Checked here rather than on the first request. The container installs the
    # runtime set only, so this is the expected outcome inside one -- and a 503
    # naming the extra is a better answer than an ImportError raised halfway
    # through a turn, which /health cannot see.
    try:
        import edge_tts  # noqa: F401
    except ImportError as error:
        raise ConfigurationError(
            "MINDSURF_TTS=edge needs the edge-tts package, which the container does not "
            "carry: install the 'tts' extra, or leave MINDSURF_TTS unset"
        ) from error

    synthesiser = EdgeSynthesiser()

    async def speak(text: str, generation: GenerationSettings) -> bytes:
        return await synthesiser.synthesise(
            Utterance(text=text, voice=generation.voice, emotion=generation.emotion)
        )

    return speak


def _build_native(settings: Settings) -> SpeechEngine:
    raise ConfigurationError(
        "the native path needs the Thinker-Talker checkpoint, which is still "
        "training; use MINDSURF_ENGINE=cascade until it lands"
    )
