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

    with pytest.raises(ConfigurationError, match="MINDSURF_TOKENIZER"):
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


def test_read_aloud_is_off_until_it_is_asked_for(tmp_path: Path) -> None:
    """Off is the default: read-aloud is a button in the client, and a
    deployment that never reaches the network should not have to opt out."""
    engine = build(_ready(tmp_path))
    assert engine is not None

    with pytest.raises(ConfigurationError, match="does not read text back"):
        import asyncio

        asyncio.run(_first_chunk(engine))


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


async def _first_chunk(engine: object) -> None:
    from mindsurf_omni.service.engine import GenerationSettings

    async for _ in engine.speak("你好", GenerationSettings()):  # type: ignore[attr-defined]
        break


def test_the_engine_carries_the_token_spec_clients_need(tmp_path: Path) -> None:
    spec = build(_ready(tmp_path)).token_spec()  # type: ignore[union-attr]

    assert spec.text_vocab_size == 6400
    assert spec.audio_codebooks == 8
    assert spec.input_sample_rate == 16_000
    assert spec.output_sample_rate == 24_000


def test_a_polisher_without_transformers_is_refused_by_name(tmp_path: Path) -> None:
    """torch and transformers are the same case. Guarding only one of them lets a
    bare ModuleNotFoundError out of a request as a 500 that /health cannot see."""
    from mindsurf_omni.service import factory

    checkpoint = tmp_path / "polish.pth"
    checkpoint.write_bytes(b"not a real checkpoint")
    root = tmp_path / "minimind"
    root.mkdir()
    (root / "model").mkdir()
    (root / "model" / "model_minimind.py").write_text("", encoding="utf-8")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(checkpoint),
            "MINIMIND_O_ROOT": str(tmp_path / "minimind"),
        }
    )
    assert settings is not None
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)

    real = factory._importable
    for absent in ("torch", "transformers"):
        # The fake answers for every package, not just the one faked absent.
        # Falling through to the real check made the test read the machine:
        # on a host without torch, faking transformers away still tripped the
        # torch refusal first, and the message named the wrong package.
        factory._importable = (  # type: ignore[assignment]
            lambda module, missing=absent: module != missing
        )
        try:
            with pytest.raises(ConfigurationError, match=absent):
                build(settings)
        finally:
            factory._importable = real  # type: ignore[assignment]


def test_the_refusal_names_the_extra_that_fixes_it(tmp_path: Path) -> None:
    """ "install the train extra" was the old answer and it was wrong: that set's
    own comment says it never runs in a container."""
    from mindsurf_omni.service import factory

    real = factory._importable
    factory._importable = lambda module: module != "torch" and real(module)  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigurationError, match=r"mindsurf-omni\[dictation\]"):
            factory._require_minimind_packages("MINDSURF_POLISH", "the polish stage")
    finally:
        factory._importable = real  # type: ignore[assignment]


def test_a_recogniser_missing_torchaudio_refuses_before_the_request(tmp_path: Path) -> None:
    """``import funasr`` succeeds where ``from funasr import AutoModel`` does not:
    funasr resolves submodules lazily and reaches torchaudio, which it imports and
    does not declare. Checking the top-level name alone checks that a directory
    exists."""
    import asyncio

    from mindsurf_omni.service import factory

    real = factory._importable
    factory._importable = lambda module: module != "torchaudio" and real(module)  # type: ignore[assignment]
    try:
        engine = build(_ready(tmp_path))
    finally:
        factory._importable = real  # type: ignore[assignment]

    assert engine is not None
    assert "transcriber" in engine.unwired  # type: ignore[attr-defined]
    with pytest.raises(ConfigurationError, match="torchaudio"):
        asyncio.run(engine.transcribe(b"\x00\x01" * 8, 16_000))


def test_the_asr_extra_declares_what_funasr_imports_and_does_not() -> None:
    """The upstream package lists 27 dependencies and no torch of any kind."""
    import tomllib

    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_bytes()
    asr = tomllib.loads(text.decode("utf-8"))["project"]["optional-dependencies"]["asr"]

    for package in ("torch", "torchaudio"):
        assert any(entry.startswith(package) for entry in asr), asr
