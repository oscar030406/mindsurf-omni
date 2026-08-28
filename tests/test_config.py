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


@pytest.mark.parametrize("requested", ["cascade", "CASCADE", " Cascade "])
def test_the_path_is_accepted_case_and_space_insensitively(requested: str) -> None:
    settings = Settings.from_environment({"MINDSURF_ENGINE": requested})

    assert settings is not None
    assert settings.path == requested.strip().lower()


def test_missing_files_are_listed_individually_not_summarised(tmp_path: Path) -> None:
    """ "Engine unavailable" tells an operator nothing.

    The codec used to be asserted here, which is how the requirement to mount
    it got in: the dictation path never opened Mimi, and demanding it kept a
    transcribe-and-polish deployment at 503 until somebody mounted gigabytes it
    would not read. Both went with the assistant line.
    """
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(tmp_path / "absent")}
    )
    assert settings is not None

    with pytest.raises(ConfigurationError) as error:
        settings.verify()

    message = str(error.value)
    assert "MINDSURF_TOKENIZER" in message
    assert "mount the weights directory" in message


def test_the_missing_file_message_names_variables_and_not_paths(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """This message is the body of a 503 and the detail on /health, and neither
    endpoint asks for a credential -- so a weights directory named after the
    customer it belongs to was being handed to anyone who could reach the port.

    The operator still needs the path, so it goes to the log instead.
    """
    root = tmp_path / "customer-a" / "weights"
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(root)}
    )
    assert settings is not None

    with (
        caplog.at_level("ERROR", logger="mindsurf.config"),
        pytest.raises(ConfigurationError) as error,
    ):
        settings.verify()

    assert str(root) not in str(error.value)
    assert "customer-a" not in str(error.value)
    assert str(root) in caplog.text


def test_verification_passes_once_everything_is_present(tmp_path: Path) -> None:
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir()
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(tmp_path)}
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
            "MINDSURF_TOKENIZER": str(elsewhere),
        }
    )

    assert settings is not None


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


def test_components_mark_what_is_frozen(tmp_path: Path) -> None:
    """Which pieces are frozen decides what a result can be attributed to."""
    checkpoint = tmp_path / "thinker.pth"
    checkpoint.write_bytes(b"weights")
    settings = Settings.from_environment(
        {"MINDSURF_ENGINE": "cascade", "MINDSURF_THINKER": str(checkpoint)}
    )
    assert settings is not None

    components = {c.name: c for c in describe_components(settings)}

    assert components["thinker"].frozen is False
    assert components["sensevoice-small"].frozen is True


def test_a_thinker_that_was_never_configured_is_not_listed() -> None:
    """Listing it unconditionally made /health answer "thinker: ready" beside
    "generator: not wired" -- two statements about one stage, contradicting
    each other, in one payload an operator is meant to act on."""
    settings = Settings.from_environment({"MINDSURF_ENGINE": "cascade"})
    assert settings is not None

    assert "thinker" not in {c.name for c in describe_components(settings)}


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


def test_a_checkpoint_that_is_not_on_disk_is_refused_at_startup(tmp_path: Path) -> None:
    """A typo in the path used to surface as the model failing mid-conversation.

    verify() only asked whether the variable was set, so a mistyped or
    unmounted checkpoint started the service, passed the health check and
    raised FileNotFoundError on the first thing a user said -- by which point
    the caller is looking at the model rather than at the path.
    """
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_THINKER": str(tmp_path / "typo.pth"),
        }
    )
    assert settings is not None

    with pytest.raises(ConfigurationError, match="MINDSURF_THINKER"):
        settings.verify()

    (tmp_path / "typo.pth").write_bytes(b"")
    settings.verify()


def test_the_same_polisher_is_fine_on_the_cascade(tmp_path: Path) -> None:
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir()
    checkpoint = tmp_path / "sft_polish.pth"
    checkpoint.write_bytes(b"weights")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(checkpoint),
        }
    )
    assert settings is not None

    settings.verify()  # must not raise


def test_a_polish_tagger_needs_its_backbone(tmp_path: Path) -> None:
    """The head is a probe of the blocks tuned with it; reading it off other
    weights measures nothing."""
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    for name in ("sft_polish.pth", "tagger.pt"):
        (tmp_path / name).write_bytes(b"weights")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(tmp_path / "sft_polish.pth"),
            "MINDSURF_POLISH_TAGGER": str(tmp_path / "tagger.pt"),
        }
    )
    assert settings is not None

    with pytest.raises(ConfigurationError, match="go together"):
        settings.verify()


def test_a_polish_tagger_without_a_polisher_has_nothing_to_merge_with(tmp_path: Path) -> None:
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    for name in ("tagger.pt", "backbone.pth"):
        (tmp_path / name).write_bytes(b"weights")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH_TAGGER": str(tmp_path / "tagger.pt"),
            "MINDSURF_POLISH_TAGGER_BACKBONE": str(tmp_path / "backbone.pth"),
        }
    )
    assert settings is not None

    with pytest.raises(ConfigurationError, match="no first arm"):
        settings.verify()


def test_the_whole_pair_verifies_when_it_is_all_there(tmp_path: Path) -> None:
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    for name in ("sft_polish.pth", "tagger.pt", "backbone.pth"):
        (tmp_path / name).write_bytes(b"weights")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(tmp_path / "sft_polish.pth"),
            "MINDSURF_POLISH_TAGGER": str(tmp_path / "tagger.pt"),
            "MINDSURF_POLISH_TAGGER_BACKBONE": str(tmp_path / "backbone.pth"),
            "MINDSURF_POLISH_TAGGER_THRESHOLD": "0.5",
        }
    )
    assert settings is not None

    settings.verify()  # must not raise

    assert settings.polish_tagger_threshold == 0.5


def test_the_second_arm_defaults_to_the_swept_threshold_not_the_first_one_tried() -> None:
    """0.5 is the value the arm was first measured at; 0.4 is the one the sweep
    chose."""
    settings = Settings.from_environment({"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": "/w"})

    assert settings is not None
    assert settings.polish_tagger_threshold == 0.4


def test_the_preview_interval_is_refused_now_rather_than_at_the_first_frame() -> None:
    """A float() that raises inside a websocket handler is a deployment that
    looked configured and was not."""
    from mindsurf_omni.service.config import ConfigurationError, Settings

    base = {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": "/w"}

    with pytest.raises(ConfigurationError, match="MINDSURF_PREVIEW_SECONDS"):
        Settings.from_environment({**base, "MINDSURF_PREVIEW_SECONDS": "一秒"})
    with pytest.raises(ConfigurationError, match="turns the preview off"):
        Settings.from_environment({**base, "MINDSURF_PREVIEW_SECONDS": "-1"})

    assert Settings.from_environment(base).preview_seconds == 1.0
    off = Settings.from_environment({**base, "MINDSURF_PREVIEW_SECONDS": "0"})
    assert off is not None and off.preview_seconds == 0


def test_turning_the_preview_off_means_the_recogniser_offers_none() -> None:
    """The websocket asks the recogniser to open a turn and takes None as "no
    preview". A setting that only changed a number would leave the card paying
    for readings nobody wanted."""
    from mindsurf_omni.service.asr import SenseVoiceRecogniser

    quiet = SenseVoiceRecogniser(model_dir=Path("x"), preview_seconds=0)
    assert quiet.open(16_000) is None

    talking = SenseVoiceRecogniser(model_dir=Path("x"), preview_seconds=2.5)
    live = talking.open(16_000)
    assert live is not None
    assert live.every == 2.5


def test_the_second_polish_arm_is_named_because_it_is_load_bearing(tmp_path) -> None:
    """Merged by veto the two arms remove 0.9902 of the unambiguous fillers on
    230 real long dictations; the generator alone removes 0.7781. A deployment
    that forgot the tagger used to report the same components as one that had
    it, and ran a quarter worse without saying so."""
    from mindsurf_omni.service.config import Settings, describe_components

    for name in ("out", "tok", "asr"):
        (tmp_path / name).mkdir()
    (tmp_path / "polish.pth").write_bytes(b"x")
    (tmp_path / "tagger.pt").write_bytes(b"y")
    base = {
        "MINDSURF_ENGINE": "cascade",
        "MINDSURF_WEIGHTS": str(tmp_path / "out"),
        "MINDSURF_TOKENIZER": str(tmp_path / "tok"),
        "MINDSURF_ASR": str(tmp_path / "asr"),
        "MINDSURF_POLISH": str(tmp_path / "polish.pth"),
    }

    one_arm = Settings.from_environment(base)
    assert one_arm is not None
    named = [c.name for c in describe_components(one_arm)]
    assert "polisher" in named
    assert "polish-tagger" not in named

    two_arms = Settings.from_environment(
        {**base, "MINDSURF_POLISH_TAGGER": str(tmp_path / "tagger.pt")}
    )
    assert two_arms is not None
    assert "polish-tagger" in [c.name for c in describe_components(two_arms)]
