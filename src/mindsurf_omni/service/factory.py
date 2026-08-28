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

from collections.abc import Callable
from pathlib import Path
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
    return _build_cascade(settings)


def _build_cascade(settings: Settings) -> SpeechEngine:
    from mindsurf_omni.service.asr import SenseVoiceRecogniser
    from mindsurf_omni.service.cascade import CascadeEngine

    if settings.asr_model == "paraformer-streaming":
        from mindsurf_omni.service.asr import ParaformerStreamingRecogniser

        # No path of its own: this one is named, not mounted, and FunASR
        # resolves it against the same cache the others come from.
        recogniser: Any = ParaformerStreamingRecogniser(
            device=settings.device,
            language=settings.asr_language,
        )
    else:
        recogniser = SenseVoiceRecogniser(
            model_dir=settings.paths.audio_encoder,
            device=settings.device,
            language=settings.asr_language,
            preview_seconds=settings.preview_seconds,
        )
    # The packages, not the weights. Loading stays deferred to the first request
    # so a container that cannot reach its weights still starts and explains
    # itself, but a package the image never installed is knowable now -- and
    # discovering it inside a request produces a 500 with an ImportError, which
    # is neither the documented 503 nor visible to /health.
    #
    # All three, not just funasr. `import funasr` succeeds on a machine where
    # `from funasr import AutoModel` does not: its __init__ resolves submodules
    # lazily, and the submodule reaches torchaudio, which funasr imports and
    # does not declare (nor torch). Checking the top-level name alone is
    # checking that a directory exists. Found by running the built image: the
    # container started, /health said the recogniser was ready, and the first
    # dictation came back 500 with ModuleNotFoundError: torchaudio.
    missing = [name for name in ("funasr", "torch", "torchaudio") if not _importable(name)]
    repair = _build_hotword_repair(settings)

    async def transcribe(pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        if missing:
            raise ConfigurationError(
                f"the recogniser needs {', '.join(missing)}, which this image does not "
                "carry: install the 'asr' extra"
            )
        text, language = await recogniser.transcribe(pcm, sample_rate)
        return repair(text), language

    generate = _build_generator(settings)
    synthesise = _build_synthesiser(settings)

    # Named here because assembly is the only place that knows which stages
    # will refuse.
    unwired = []
    if settings.thinker is None:
        unwired.append("generator")
    if missing:
        unwired.append("transcriber")

    engine = CascadeEngine(
        polisher=_build_polisher(settings),
        recogniser=recogniser,
        realtime=settings.realtime,
        transcriber=transcribe,
        generator=generate,  # type: ignore[arg-type]
        synthesiser=synthesise,
        components=describe_components(settings),
        token_spec=token_spec(),
        unwired=tuple(unwired),
    )
    # Handed to the engine rather than called here: assembly must not block on
    # a network-mounted checkpoint, and the app warms it at startup instead.
    engine._warm_recogniser = recogniser.load if not missing else None
    return engine


def _build_hotword_repair(settings: Settings) -> Callable[[str], str]:
    """The pass that puts the operator's proper nouns back, or one that does not.

    Wrapped around the recogniser rather than bolted onto the dictation
    endpoint so that conversation gets it too: the cascade answers what it
    heard, and hearing a colleague's name wrong is the same defect on either
    path. It sits before the polisher, which the endpoint calls on this
    stage's output.

    The table is built here, at assembly, because building it is where a bad
    entry is refused -- see hotwords.build_table.
    """
    if not settings.hotwords:
        return lambda text: text
    from mindsurf_omni.service.hotwords import build_table, correct

    table = build_table(settings.hotwords)
    return lambda text: correct(text, table)


def _build_polisher(settings: Settings) -> Any:
    """The dictation path's second stage, or None when none was named.

    Loaded at assembly rather than on the first request: unlike the
    synthesiser, this one is not optional once configured -- a dictation turn
    that arrives before the weights are resident would pay the whole load, and
    the operator asked for it by naming a checkpoint.
    """
    if settings.polish is None:
        return None
    if settings.minimind_root is None:
        raise ConfigurationError(
            "MINDSURF_POLISH is set but MINIMIND_O_ROOT is not; the polisher is built from "
            "MiniMind's own model class rather than a copy of it, so the checkout is needed"
        )
    _require_minimind_packages("MINDSURF_POLISH", "the polish stage")

    from mindsurf_omni.service.polish import Polisher

    polisher = Polisher(
        checkpoint=settings.polish,
        tokenizer_dir=settings.paths.tokenizer,
        minimind_root=settings.minimind_root,
        device=settings.device,
        tagger=settings.polish_tagger,
        tagger_backbone=settings.polish_tagger_backbone,
        tagger_threshold=settings.polish_tagger_threshold,
    )
    polisher.load()
    return polisher


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
    _require_minimind_packages("MINDSURF_THINKER", "the text stage")

    from mindsurf_omni.service.thinker import ThinkerGenerator

    thinker = ThinkerGenerator(
        checkpoint=settings.thinker,
        tokenizer_dir=settings.paths.tokenizer,
        minimind_root=settings.minimind_root,
        device=settings.device,
    )
    # Loaded here rather than on the first request. Lazily, the service reports
    # ready while the weights are still on disk, and the first caller pays for
    # the load: measured at 32 s on a laptop card against 31 ms warm, and the
    # caller reads that as the model being slow. Startup is the honest place to
    # spend it -- an orchestrator waits for ready, a user does not.
    thinker.load()
    return thinker.generate


def _require_audio_loader(prompt_wav: Path) -> None:
    """Can VoxCPM actually read the clone clip, and if not, say why.

    VoxCPM opens the prompt through ``torchaudio.load``, which in torchaudio 2.x
    delegates to torchcodec, which needs FFmpeg's *shared* libraries. A machine
    with a static FFmpeg build has the ``ffmpeg`` command and none of the DLLs,
    and the failure is a hundred lines of library paths raised inside the first
    request -- measured here as 40 utterances out of 40, with nothing in the
    message naming FFmpeg as the thing to install.

    Installed is not the same as loadable, and the second shape cost an hour on
    top of the first. This machine had the shared build -- the right one, from
    winget -- with its DLLs sitting in the package directory and that directory
    on nobody's PATH, while ``ffmpeg`` on PATH resolved to an npm shim. So
    ``ffmpeg -version`` answered, every DLL was present, and torchcodec still
    loaded none of them. The message says both, because "install a shared
    build" sends someone to reinstall what they already have.

    Checked by loading the configured clip rather than by probing torchcodec:
    the question is whether this file can be read on this machine, and the file
    is five seconds long.

    Only when a prompt is configured. Without one VoxCPM opens nothing and this
    dependency does not exist -- which is exactly why the no-prompt arm kept
    working while every prompted one died.
    """
    try:
        import torchaudio
    except ImportError as error:  # pragma: no cover - torch is checked above
        raise ConfigurationError(
            "MINDSURF_TTS_PROMPT_WAV is set but torchaudio is not installed; "
            "VoxCPM reads the clone clip through it"
        ) from error
    try:
        torchaudio.load(str(prompt_wav))
    except Exception as error:
        raise ConfigurationError(
            f"the clone clip at {prompt_wav} cannot be read on this machine: "
            f"{type(error).__name__}. VoxCPM opens it through torchaudio, which needs "
            "FFmpeg's shared libraries. Two ways this goes wrong and they look the "
            "same: a static ffmpeg build has the command and none of the DLLs, or a "
            "shared build is installed and its DLL directory is not on PATH -- check "
            "for avcodec-*.dll and check that the directory holding them is on PATH, "
            "because `ffmpeg -version` answers in both cases and can even be answering "
            "from a different build entirely. Otherwise leave MINDSURF_TTS_PROMPT_WAV "
            "unset -- without a clip VoxCPM draws a speaker per call, so the voice "
            "changes between sentences"
        ) from error


def _require_minimind_packages(setting: str, stage: str) -> None:
    """Refuse at assembly for anything the MiniMind-backed stages import.

    torch was guarded and transformers was not, so a host carrying one and not
    the other got a bare ModuleNotFoundError out of `from transformers import
    AutoTokenizer` (thinker.py) -- raised through the polisher, through the
    engine, and out of a request as a 500 that /health could not see. The
    guarded case and the unguarded one are the same case; the difference was
    only which package somebody remembered.
    """
    for package, extra in (("torch", "dictation"), ("transformers", "dictation")):
        if not _importable(package):
            raise ConfigurationError(
                f"{setting} is set but {package} is not installed; the base image carries "
                f"the runtime set only, so {stage} needs the '{extra}' extra "
                f"(pip install 'mindsurf-omni[{extra}]')"
            )


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
    """What reads the finished text back, or a callable that says why it cannot.

    Off unless an operator asks for it. Read-aloud is a button in the client --
    somebody taps the speaker on a note they already have -- so nothing about a
    dictation turn depends on this being wired, and a deployment that never
    reaches the network is the default rather than a special case.

    Hosted only. The local synthesiser went with the assistant line, and on the
    material that matters it was the worse reader: over four real dictation
    notes edge holds 4.71 to 5.01 characters a second, a 6% spread, against
    4.64 to 6.12 for VoxCPM, which is 32%. On a 40-sentence benchmark the two
    differed by 2.6%, and that benchmark was long enumerations and written
    sentences -- the wrong material, again.

    The voice is chosen rather than inherited: `zh-CN-XiaoxiaoNeural`, picked by
    ear from six candidates on the same dictated note. Two of the rejected ones
    read 下载 with the wrong tone on 载, which no ruler here can catch -- read
    zǎi or zài it is the same character, so character error rate is exactly
    zero. See DECISIONS §29.
    """
    from mindsurf_omni.data.synthesis import EdgeSynthesiser, Utterance
    from mindsurf_omni.service.engine import GenerationSettings

    if not settings.tts:

        async def unwired(text: str, _: Any) -> bytes:
            raise ConfigurationError(
                "this deployment does not read text back; set MINDSURF_TTS=edge to turn "
                "it on. It reaches the network, and it is named in /v1/models when it is on"
            )

        return unwired

    # Checked here rather than at the first request. The base image carries the
    # runtime set only, so a 503 naming the extra beats an ImportError raised
    # halfway through a request, which /health cannot see.
    if not _importable("edge_tts"):
        raise ConfigurationError(
            "MINDSURF_TTS=edge needs the edge_tts package, which the container does not "
            "carry: install the 'tts' extra, or leave MINDSURF_TTS unset"
        )

    synthesiser = EdgeSynthesiser()

    async def speak(text: str, generation: GenerationSettings) -> bytes:
        return await synthesiser.synthesise(
            Utterance(text=text, voice=generation.voice, emotion=generation.emotion)
        )

    return speak
