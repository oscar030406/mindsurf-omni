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
    # The package, not the weights. Loading stays deferred to the first request
    # so a container that cannot reach its weights still starts and explains
    # itself, but a package the image never installed is knowable now -- and
    # discovering it inside a request produces a 500 with an ImportError, which
    # is neither the documented 503 nor visible to /health.
    recogniser_available = _importable("funasr")

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        if not recogniser_available:
            raise ConfigurationError(
                "the recogniser needs the funasr package, which this image does not "
                "carry: install the 'asr' extra"
            )
        return await recogniser.transcribe(pcm, sample_rate)

    generate = _build_generator(settings)
    synthesise = _build_synthesiser(settings)

    # Named here because assembly is the only place that knows which stages
    # will refuse.
    unwired = []
    if settings.thinker is None:
        unwired.append("generator")
    if not settings.tts:
        unwired.append("synthesiser")
    if not recogniser_available:
        unwired.append("transcriber")

    return CascadeEngine(
        transcriber=transcribe,
        generator=generate,  # type: ignore[arg-type]
        synthesiser=synthesise,
        components=describe_components(settings),
        token_spec=token_spec(),
        unwired=tuple(unwired),
    )


def _build_generator(settings: Settings) -> Any:
    """The cascade's text stage, or one that says what it is waiting for."""
    if settings.thinker is None:

        async def unwired(messages: list[dict[str, str]], _: Any) -> Any:
            raise ConfigurationError(
                "the cascade path has no text generator wired; point MINDSURF_THINKER at a "
                "Thinker checkpoint and MINIMIND_O_ROOT at a MiniMind-O checkout"
            )
            yield ""  # pragma: no cover - makes this an async generator

        return unwired

    if settings.minimind_root is None:
        raise ConfigurationError(
            "MINDSURF_THINKER is set but MINIMIND_O_ROOT is not; the Thinker is built from "
            "MiniMind's own model class rather than a copy of it, so the checkout is needed"
        )
    if not _importable("torch"):
        raise ConfigurationError(
            "MINDSURF_THINKER is set but torch is not installed; the image carries the "
            "runtime set only, so the text stage runs on a host with the 'train' extra"
        )

    from mindsurf_omni.service.thinker import ThinkerGenerator

    thinker = ThinkerGenerator(
        checkpoint=settings.thinker,
        tokenizer_dir=settings.paths.tokenizer,
        minimind_root=settings.minimind_root,
        device=settings.device,
    )
    return thinker.generate


def _importable(module: str) -> bool:
    """Is the package present, without paying to import it.

    find_spec rather than import: the answer is needed at assembly, and funasr
    pulls in torch, which is seconds and hundreds of megabytes to learn
    something the file system already knows.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


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
    if not _importable("edge_tts"):
        raise ConfigurationError(
            "MINDSURF_TTS=edge needs the edge-tts package, which the container does not "
            "carry: install the 'tts' extra, or leave MINDSURF_TTS unset"
        )

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
