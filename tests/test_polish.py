"""The dictation path's second stage, wired into the service."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mindsurf_omni.service.cascade import CascadeEngine
from mindsurf_omni.service.config import ConfigurationError, Settings, describe_components
from mindsurf_omni.service.polish import Polisher, build_prompt, reachable, subsequence_pointer


def _cascade(polisher: object = None) -> CascadeEngine:
    async def transcribe(pcm: bytes, rate: int) -> tuple[str, str | None]:
        return "那个今天天气怎么样", "zh"

    async def speak(text: str, settings: object) -> bytes:
        return b""

    return CascadeEngine(
        transcriber=transcribe,
        generator=lambda *_: None,  # type: ignore[arg-type]
        synthesiser=speak,
        components=[],
        token_spec=None,  # type: ignore[arg-type]
        polisher=polisher,
    )


def test_no_polisher_answers_none_rather_than_the_transcript() -> None:
    """A caller must tell "this service does not polish" from "nothing to polish"."""
    engine = _cascade()

    assert asyncio.run(engine.polish("那个今天天气怎么样")) is None


def test_a_wired_polisher_is_used() -> None:
    class _Fake:
        async def polish(self, transcript: str) -> str:
            return transcript.replace("那个", "")

    assert asyncio.run(_cascade(_Fake()).polish("那个今天天气怎么样")) == "今天天气怎么样"


def test_a_missing_polish_checkpoint_is_refused_at_startup(tmp_path: Path) -> None:
    """Otherwise the first dictation comes back unpolished and reads as a model doing nothing."""
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(tmp_path / "typo.pth"),
        }
    )
    assert settings is not None

    with pytest.raises(ConfigurationError, match="typo.pth"):
        settings.verify()

    (tmp_path / "typo.pth").write_bytes(b"")
    settings.verify()


def test_the_polisher_is_named_and_hashed_in_the_component_list(tmp_path: Path) -> None:
    """It rewrites what the user sees, so which weights did it belongs in the report."""
    checkpoint = tmp_path / "polish.pth"
    checkpoint.write_bytes(b"weights")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(checkpoint),
        }
    )
    assert settings is not None

    named = [c for c in describe_components(settings) if c.name == "polisher"]

    assert len(named) == 1
    assert named[0].sha256 is not None and len(named[0].sha256) == 64


def test_the_decode_can_only_walk_forward_through_the_transcript() -> None:
    """Same rule the measurement round settled on, now in the service."""
    source = [10, 11, 12, 13, 14]

    assert subsequence_pointer(source, [10, 13]) == 4
    assert reachable(source, 1, lookahead=2) == [11, 12]
    assert reachable(source, 1, lookahead=0) == [11, 12, 13, 14]


def test_empty_input_is_returned_untouched(tmp_path: Path) -> None:
    """No forward pass for nothing, and no chance to invent a sentence from silence."""
    polisher = Polisher(
        checkpoint=tmp_path / "unused.pth",
        tokenizer_dir=tmp_path,
        minimind_root=tmp_path,
    )

    assert asyncio.run(polisher.polish("   ")) == "   "


def test_the_instruction_is_the_one_the_model_was_trained_on() -> None:
    """Train and serve share these words; a flag would let them drift apart."""
    from scripts.train_polish import INSTRUCTION as trained

    assert build_prompt("你好") == f"{trained}\n\n你好"


def test_projection_keeps_the_deletions_and_drops_the_inventions() -> None:
    """The model's text intersected with the transcript, in order."""
    from mindsurf_omni.service.polish import project_onto

    transcript = "那个，今天天气怎么样？"

    # Filler removed and nothing added: projection changes nothing.
    assert project_onto(transcript, "今天天气怎么样？") == "今天天气怎么样？"
    # The model answered instead: only what it echoed survives.
    assert project_onto(transcript, "今天天气很好，适合出门") == "今天天气"
    # Untouched output projects back to itself.
    assert project_onto(transcript, transcript) == transcript


def test_the_first_word_is_protected_unless_it_is_a_filler() -> None:
    """Measured end to end: 你想看什么类型的电影 came back as 想看什么类型的电影."""
    from mindsurf_omni.service.polish import reachable

    source = [40, 41, 50, 51, 52]  # 40,41 are a filler; 50.. is the sentence

    free = reachable(source, 0, lookahead=3, protect_head=False)
    guarded = reachable(source, 0, lookahead=3, protect_head=True)
    with_door = reachable(source, 0, lookahead=3, fillers=((40, 41),), protect_head=True)

    assert 50 in free  # unguarded, the opening word can be skipped
    assert guarded == [40]  # guarded, only the first token
    assert 50 in with_door  # unless the skip is the filler itself


def test_a_projected_target_is_reachable_by_a_deletion_only_decoder() -> None:
    """What --project-targets buys: every training target is one the decoder can produce.

    45% of the pairs carry a clean text that is not a subsequence of the
    transcript -- the recogniser misheard a character, and no amount of
    deleting recovers it. Reaching for it is what makes the decoder skip.
    """
    from mindsurf_omni.service.polish import project_onto

    source, target = "那个今天天汽怎么样", "今天天气怎么样"  # 气 heard as 汽

    projected = project_onto(source, target)

    remaining = iter(source)
    assert all(character in remaining for character in projected)
    assert "气" not in projected  # unreachable, and no longer asked for
