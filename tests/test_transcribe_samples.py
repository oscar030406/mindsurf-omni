"""The independence check, which is the only reason this step is separate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.transcribe_samples import (
    JudgeError,
    load_manifest,
    transcribe_all,
    verify_independent,
)


def test_our_own_recogniser_is_refused_as_judge() -> None:
    """Shared failure modes cancel, so the score is best where it should warn."""
    with pytest.raises(JudgeError, match="circular"):
        verify_independent("sensevoice-small", "thinker talker")


def test_an_independent_recogniser_is_accepted() -> None:
    verify_independent("whisper", "thinker talker sensevoice-small")


def test_a_judge_that_is_part_of_the_model_under_test_is_refused() -> None:
    """Even an otherwise independent one, if the model embeds it."""
    with pytest.raises(JudgeError, match="would cancel"):
        verify_independent("whisper", "thinker whisper-encoder")


def _manifest(tmp_path: Path, audio: list[bool]) -> dict:
    samples = []
    for index, present in enumerate(audio):
        path = tmp_path / f"zh{index:03d}.wav"
        if present:
            path.write_bytes(b"RIFF")
        samples.append(
            {
                "id": f"zh{index:03d}",
                "prompt": f"问题{index}",
                "reference_text": f"回答{index}",
                "audio_path": str(path) if present else None,
            }
        )
    return {
        "generated_by": {"components": [{"name": "thinker"}, {"name": "sensevoice-small"}]},
        "samples": samples,
    }


def test_missing_audio_is_recorded_as_a_failure_to_measure(tmp_path: Path) -> None:
    """Omitting it would shrink the sample size and every noise floor with it."""
    rows = transcribe_all(_manifest(tmp_path, [True, False, True]), lambda p: "话", "whisper")

    assert len(rows) == 3
    assert [row["transcribed"] for row in rows] == [True, False, True]
    assert rows[1]["transcript"] == ""


def test_one_unreadable_file_does_not_end_the_run(tmp_path: Path) -> None:
    def flaky(path: Path) -> str:
        if "zh001" in str(path):
            raise OSError("corrupt")
        return "话"

    rows = transcribe_all(_manifest(tmp_path, [True, True, True]), flaky, "whisper")

    assert len(rows) == 3
    assert "corrupt" in rows[1]["error"]
    assert rows[2]["transcribed"] is True


def test_the_reference_text_survives_transcription(tmp_path: Path) -> None:
    """CER needs both sides; losing one would score against nothing."""
    rows = transcribe_all(_manifest(tmp_path, [True]), lambda p: "听到的", "whisper")

    assert rows[0]["reference_text"] == "回答0"
    assert rows[0]["transcript"] == "听到的"


def test_the_judge_is_stamped_on_every_row(tmp_path: Path) -> None:
    """Including the rows that failed, which is where provenance is easiest to lose."""
    rows = transcribe_all(
        _manifest(tmp_path, [True, False]), lambda p: "话", "paraformer", judge="paraformer-zh"
    )

    assert [row["judge"] for row in rows] == ["paraformer-zh", "paraformer-zh"]


def test_an_empty_manifest_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": []}), encoding="utf-8")

    with pytest.raises(SystemExit, match="no samples"):
        load_manifest(path)
