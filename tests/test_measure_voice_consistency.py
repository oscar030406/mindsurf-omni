"""One voice or several, which is the thing the reference clip was supposed to fix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.measure_voice_consistency import (
    FIXED_VOICE,
    RATE_CEILING,
    SCATTERED_VOICE,
    load_arm,
    read_punctuation,
    speaking_rate,
    spread,
    verdict,
)


def _vectors(*rows: tuple[float, ...]):
    """Stand-ins for embeddings, so the arithmetic is checkable without CAM++."""
    import torch

    return [torch.tensor(row) for row in rows]


def test_one_voice_reads_high_and_several_read_low() -> None:
    """The whole comparison rests on this being the direction it is."""
    same = spread(_vectors((1.0, 0.0), (0.99, 0.14), (0.98, 0.20)), draws=50)
    different = spread(_vectors((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)), draws=50)

    assert same["median"] > FIXED_VOICE
    assert different["median"] < SCATTERED_VOICE
    assert same["pairs"] == 3


def test_the_floor_comes_from_resampling_utterances() -> None:
    """Pairs share terms; bootstrapping them would report a firmness that is not there."""
    measured = spread(_vectors((1.0, 0.0), (0.6, 0.8), (0.0, 1.0), (0.8, 0.6)), draws=200)

    assert measured["pairs"] == 6
    assert measured["noise_floor"] is not None and measured["noise_floor"] > 0.0


def test_a_stretched_clip_is_flagged_and_an_ordinary_one_is_not() -> None:
    """A squeal or a drone shows up as characters per second nowhere near the rest."""
    rows = [
        {"id": "zh000", "reference_text": "今天天气怎么样", "audio_seconds": 1.5, "path": "a"},
        {"id": "zh001", "reference_text": "今天天气怎么样", "audio_seconds": 60.0, "path": "b"},
        {"id": "zh002", "reference_text": "今天天气怎么样", "audio_seconds": 0.3, "path": "c"},
    ]

    measured = speaking_rate(rows)

    assert [item["id"] for item in measured["outliers"]] == ["zh001", "zh002"]
    assert measured["outlier_rate"] == pytest.approx(2 / 3)
    assert measured["maximum"] > RATE_CEILING


def test_a_clip_that_never_spoke_is_counted_apart_from_a_slow_one() -> None:
    """Silence is a different fault from a drone and would otherwise vanish."""
    rows = [
        {"id": "zh000", "reference_text": "今天天气怎么样", "audio_seconds": 1.5, "path": "a"},
        {"id": "zh001", "reference_text": "今天天气怎么样", "audio_seconds": 0.0, "path": None},
    ]

    measured = speaking_rate(rows)

    assert measured["silent"] == ["zh001"]
    assert measured["n"] == 1


def test_the_verdict_needs_both_arms_to_call_the_cause_confirmed() -> None:
    """A tight prompted arm alone does not show the unprompted one was the fault."""
    tight = {"timbre": {"median": 0.95}, "rate": {"outlier_rate": 0.0}}
    scattered = {"timbre": {"median": 0.35}, "rate": {"outlier_rate": 0.0}}

    both = verdict({"voxcpm_ref": tight, "voxcpm_noref": scattered})
    only_tight = verdict({"voxcpm_ref": tight, "voxcpm_noref": tight})
    unfixed = verdict({"voxcpm_ref": scattered, "voxcpm_noref": scattered})

    assert "修好了" in both["音色"]
    assert "没坐实" in only_tight["音色"]
    assert "没修好" in unfixed["音色"]
    assert "判不了" in verdict({"voxcpm_ref": tight})["音色"]


def test_a_directory_without_a_manifest_is_refused(tmp_path: Path) -> None:
    """The manifest is what says which text each clip was asked to say."""
    with pytest.raises(SystemExit, match="manifest"):
        load_arm(tmp_path)

    (tmp_path / "manifest.json").write_text(json.dumps({"samples": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="没有样本"):
        load_arm(tmp_path)


def test_a_synthesiser_reading_a_mark_aloud_is_caught() -> None:
    """The third symptom from the meeting: 逗号 spoken instead of paused."""
    rows = [
        {"id": "a", "reference_text": "今天天气怎么样", "audio_seconds": 1.5, "path": "a"},
        {"id": "b", "reference_text": "逗号怎么用", "audio_seconds": 1.5, "path": "b"},
    ]
    transcripts = {"a": "今天天气怎么样逗号", "b": "逗号怎么用"}

    caught = read_punctuation(rows, transcripts)

    # b asked about commas, so hearing one back is not a defect.
    assert [clip["id"] for clip in caught["clips"]] == ["a"]
    assert caught["rate"] == 0.5


def test_an_arm_without_transcripts_says_so_rather_than_reporting_clean() -> None:
    """Zero found and nothing checked must not look alike in the report."""
    assert read_punctuation([{"id": "a", "reference_text": "你好"}], {})["clips"] == []
