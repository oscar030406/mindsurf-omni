"""Configuration, and the quality of its failure messages.

An operator reading a startup failure at three in the morning should learn
which variable was missing, not that something was unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mindsurf_omni.service.config import (
    AUDIO_SPECIAL_TOKENS,
    SPECIAL_TOKENS,
    ConfigurationError,
    Settings,
    describe_components,
    token_spec,
)


def test_no_engine_requested_is_a_valid_state_not_an_error() -> None:
    """It is what lets the backend integrate before the model exists."""
    assert Settings.from_environment({}) is None
    assert Settings.from_environment({"MINDSURF_ENGINE": "  "}) is None


def test_an_unrecognised_engine_names_what_was_given() -> None:
    with pytest.raises(ConfigurationError, match="'nativ'"):
        Settings.from_environment({"MINDSURF_ENGINE": "nativ"})


@pytest.mark.parametrize("requested", ["native", "cascade", "NATIVE", " Cascade "])
def test_both_paths_are_accepted_case_and_space_insensitively(requested: str) -> None:
    settings = Settings.from_environment({"MINDSURF_ENGINE": requested})

    assert settings is not None
    assert settings.path == requested.strip().lower()


def test_missing_files_are_listed_individually_not_summarised(tmp_path: Path) -> None:
    """ "Engine unavailable" tells an operator nothing."""
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(tmp_path / "absent")}
    )
    assert settings is not None

    with pytest.raises(ConfigurationError) as error:
        settings.verify()

    message = str(error.value)
    assert "tokenizer=" in message
    assert "codec=" in message
    assert "mount the weights directory" in message


def test_verification_passes_once_everything_is_present(tmp_path: Path) -> None:
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir()
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "native", "MINDSURF_WEIGHTS": str(tmp_path)}
    )
    assert settings is not None

    settings.verify()  # must not raise


def test_individual_paths_can_be_overridden(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_CODEC": str(elsewhere),
        }
    )

    assert settings is not None
    assert settings.paths.codec == elsewhere


def test_the_token_spec_matches_the_ids_the_tokenizer_reserves() -> None:
    """These are read from the vocabulary, not chosen here."""
    spec = token_spec()

    assert spec.special_tokens["audio_start"] == SPECIAL_TOKENS["audio_start"] == 14
    assert spec.special_tokens["image_pad"] == 12  # kept for a later vision path
    assert spec.audio_codebooks == 8
    assert spec.audio_frame_rate_hz == 12.5


def test_audio_special_ids_sit_above_the_codebook() -> None:
    """Overlapping the 2048 Mimi codes would make a control token decode to sound."""
    assert min(AUDIO_SPECIAL_TOKENS.values()) >= 2048


def test_components_mark_what_is_frozen() -> None:
    """Which pieces are frozen decides what a result can be attributed to."""
    settings = Settings.from_environment({"MINDSURF_ENGINE": "native"})
    assert settings is not None

    components = {c.name: c for c in describe_components(settings)}

    assert components["thinker"].frozen is False
    assert components["mimi-codec"].frozen is True
    assert components["sensevoice-small"].frozen is True


def test_the_cascade_path_does_not_claim_a_talker() -> None:
    """It has none, and listing one would misattribute its output."""
    settings = Settings.from_environment({"MINDSURF_ENGINE": "cascade"})
    assert settings is not None

    names = {c.name for c in describe_components(settings)}

    assert "talker" not in names
    assert "sensevoice-small" in names


def test_the_thinker_component_names_which_weights_spoke(tmp_path: Path) -> None:
    """Two runs on two checkpoints would otherwise carry identical provenance.

    That is the whole comparison the evaluation exists to make -- base against
    fine-tuned -- and nothing else in the manifest distinguishes them.
    """
    from mindsurf_omni.service.config import checkpoint_digest, describe_components

    checkpoint = tmp_path / "sft.pth"
    checkpoint.write_bytes(b"weights")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_THINKER": str(checkpoint),
        }
    )
    assert settings is not None

    thinker = describe_components(settings)[0]

    assert thinker.name == "thinker"
    assert thinker.sha256 == checkpoint_digest(checkpoint)
    assert thinker.sha256 is not None and len(thinker.sha256) == 64


def test_no_checkpoint_reports_no_digest_rather_than_a_wrong_one(tmp_path: Path) -> None:
    """None is "nothing configured", not "a checkpoint we did not hash"."""
    from mindsurf_omni.service.config import checkpoint_digest, describe_components

    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(tmp_path)}
    )
    assert settings is not None

    assert describe_components(settings)[0].sha256 is None
    assert checkpoint_digest(tmp_path / "absent.pth") is None
