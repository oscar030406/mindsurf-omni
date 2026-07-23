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


async def _first_chunk(engine: object) -> None:
    from mindsurf_omni.service.engine import GenerationSettings

    async for _ in engine.speak("你好", GenerationSettings()):  # type: ignore[attr-defined]
        break


def test_the_native_path_says_what_it_is_waiting_for(tmp_path: Path) -> None:
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "native", "MINDSURF_WEIGHTS": str(tmp_path)}
    )

    with pytest.raises(ConfigurationError, match="still\ntraining|still training"):
        build(settings)


def test_the_engine_carries_the_token_spec_clients_need(tmp_path: Path) -> None:
    spec = build(_ready(tmp_path)).token_spec()  # type: ignore[union-attr]

    assert spec.text_vocab_size == 6400
    assert spec.audio_codebooks == 8
    assert spec.input_sample_rate == 16_000
    assert spec.output_sample_rate == 24_000
