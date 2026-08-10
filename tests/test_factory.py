"""Assembly, and the difference between "not configured" and "broken"."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindsurf_omni.service.config import ConfigurationError, Settings
from mindsurf_omni.service.factory import build


def _ready(tmp_path: Path) -> Settings:
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(tmp_path)}
    )
    assert settings is not None
    return settings


def test_no_settings_builds_no_engine_and_does_not_raise() -> None:
    """A service with no engine is a valid state, not a failure."""
    assert build(None) is None


def test_missing_weights_name_themselves(tmp_path: Path) -> None:
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(tmp_path / "absent")}
    )

    with pytest.raises(ConfigurationError, match="tokenizer="):
        build(settings)


def test_the_cascade_engine_builds_once_its_files_are_present(tmp_path: Path) -> None:
    engine = build(_ready(tmp_path))

    assert engine is not None
    assert engine.describe().path == "cascade"


def test_a_built_engine_still_reports_the_licence(tmp_path: Path) -> None:
    """It travels with every response, so it must survive assembly."""
    description = build(_ready(tmp_path)).describe()  # type: ignore[union-attr]

    assert description.commercial_use_permitted is False
    assert description.licence == "CC-BY-NC-4.0"


def test_an_unwired_stage_says_which_stage_and_why(tmp_path: Path) -> None:
    """ "Not implemented" would send the reader to the source; this does not."""
    engine = build(_ready(tmp_path))
    assert engine is not None

    with pytest.raises(ConfigurationError, match="synthesiser"):
        import asyncio

        asyncio.run(_first_chunk(engine))


def test_the_unwired_generator_reaches_the_caller_as_503(tmp_path: Path) -> None:
    """Assembled for real, not with a fake: this is the stage that is actually unwired."""
    from fastapi.testclient import TestClient

    from mindsurf_omni.service.app import create_app

    client = TestClient(create_app(build(_ready(tmp_path))))

    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "你好"}]}
    )

    assert response.status_code == 503
    assert "text generator" in response.json()["detail"]


def test_a_missing_recogniser_package_is_503_not_500(tmp_path: Path) -> None:
    """The image installs the runtime set only, so this is the normal case inside one.

    Reaching the first request and raising ImportError gives a 500, a stack
    trace in a log the caller cannot read, and a /health that still says the
    recogniser is fine.
    """
    import asyncio

    from mindsurf_omni.service import factory

    # Forced rather than inferred from this machine: a test that quietly passes
    # wherever funasr happens to be installed is the check that never fails.
    absent = {"funasr"}
    real = factory._importable
    factory._importable = lambda module: module not in absent and real(module)  # type: ignore[assignment]
    try:
        engine = build(_ready(tmp_path))
    finally:
        factory._importable = real  # type: ignore[assignment]

    assert engine is not None
    assert "transcriber" in engine.unwired  # type: ignore[attr-defined]
    with pytest.raises(ConfigurationError, match="funasr"):
        asyncio.run(engine.transcribe(b"\x00\x01" * 8, 16_000))


def test_health_does_not_call_a_half_built_cascade_ready(tmp_path: Path) -> None:
    """It holds every component and still cannot answer a turn."""
    from mindsurf_omni.service.health import assess

    report = assess(build(_ready(tmp_path)))

    assert report.status == "degraded"
    assert "generator" in [component.name for component in report.components if not component.ready]


def test_no_synthesiser_is_chosen_without_being_asked_for(tmp_path: Path) -> None:
    """Defaulting to the hosted one would send replies off the machine unasked."""
    settings = _ready(tmp_path)

    assert settings.tts == ""
    assert not any("tts" in component.name for component in build(settings).describe().components)  # type: ignore[union-attr]


def test_an_unknown_synthesiser_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """Falling back silently would report audio produced by something else."""
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_TTS": "cosyvoice",
        }
    )
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)

    with pytest.raises(ConfigurationError, match="cosyvoice"):
        build(settings)


def test_the_wired_synthesiser_is_named_in_the_component_list(tmp_path: Path) -> None:
    """CER measures what the synthesiser said, so a report that omits it compares nothing."""
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_TTS": "edge",
        }
    )

    names = [component.name for component in build(settings).describe().components]  # type: ignore[union-attr]

    assert "tts-edge" in names


def test_the_local_synthesiser_is_named_too(tmp_path: Path) -> None:
    """Two synthesisers make the same audio into two different measurements."""
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_TTS": "voxcpm",
        }
    )
    from mindsurf_omni.service import factory

    real = factory._importable
    factory._importable = lambda module: module == "voxcpm" or real(module)
    try:
        names = [component.name for component in build(settings).describe().components]  # type: ignore[union-attr]
    finally:
        factory._importable = real

    assert "tts-voxcpm" in names


def test_the_local_synthesiser_names_its_own_extra_when_absent(tmp_path: Path) -> None:
    """ "install the tts extra" would install the hosted one, which is the opposite choice."""
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_TTS": "voxcpm",
        }
    )
    from mindsurf_omni.service import factory

    real = factory._importable
    factory._importable = lambda module: module != "voxcpm" and real(module)
    try:
        with pytest.raises(ConfigurationError, match="tts-local"):
            build(settings)
    finally:
        factory._importable = real


async def _first_chunk(engine: object) -> None:
    from mindsurf_omni.service.engine import GenerationSettings

    async for _ in engine.speak("你好", GenerationSettings()):  # type: ignore[attr-defined]
        break


def test_the_native_path_says_what_it_is_waiting_for(tmp_path: Path) -> None:
    """One checkpoint supplies both halves, so both variables are needed."""
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "native", "MINDSURF_WEIGHTS": str(tmp_path)}
    )

    with pytest.raises(ConfigurationError, match="MINDSURF_THINKER"):
        build(settings)


def test_the_native_path_refuses_a_checkout_it_cannot_find(tmp_path: Path) -> None:
    """Named before torch is imported: the image has no torch and would say so instead."""
    from mindsurf_omni.service import factory

    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    # The checkout is the thing missing here; the checkpoint has to exist or
    # assembly refuses for that instead.
    (tmp_path / "sft.pth").write_bytes(b"")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "native",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_THINKER": str(tmp_path / "sft.pth"),
            "MINIMIND_O_ROOT": str(tmp_path / "absent"),
        }
    )
    real = factory._importable
    factory._importable = lambda module: module == "torch" or real(module)  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigurationError, match="checkout is not at"):
            build(settings)
    finally:
        factory._importable = real  # type: ignore[assignment]


def test_the_engine_carries_the_token_spec_clients_need(tmp_path: Path) -> None:
    spec = build(_ready(tmp_path)).token_spec()  # type: ignore[union-attr]

    assert spec.text_vocab_size == 6400
    assert spec.audio_codebooks == 8
    assert spec.input_sample_rate == 16_000
    assert spec.output_sample_rate == 24_000
