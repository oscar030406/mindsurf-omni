"""The floor the polish training target is supposed to come from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.measure_polish_floor import (
    arm,
    error_shapes,
    filler_retention,
    load_rows,
    punctuated,
    punctuation_agreement,
    verdicts,
)


def _row(id: str, spoken: str, heard: str, clean: str = "", injections: list | None = None) -> dict:
    return {
        "id": id,
        "reference_text": spoken,
        "asr_transcript": heard,
        "clean_text": clean or spoken,
        "injections": injections or [],
    }


def test_filler_already_in_the_sentence_is_not_counted_as_survived() -> None:
    """就是 and 然后 are ordinary words, and the written text is full of them.

    Counting occurrences rather than the excess reports filler the injection
    never added, which is the difference between "the recogniser keeps filler"
    and "the sentence contained that word already".
    """
    rows = [
        _row(
            "zh000",
            "就是，我就是这么想的",
            "我就是这么想的",  # the injected 就是 is gone; the original one is not
            clean="我就是这么想的",
            injections=[{"kind": "filler", "token": "就是", "clause": 0}],
        )
    ]

    result = filler_retention(rows)

    assert result == {
        "injected": 1,
        "kept": 0,
        "retention": 0.0,
        "by_kind": {"filler": {"injected": 1, "kept": 0, "retention": 0.0}},
        "by_token": {"就是": {"injected": 1, "kept": 0}},
    }


def test_filler_that_survived_is_counted() -> None:
    rows = [
        _row(
            "zh000",
            "那个，今天天气怎么样",
            "那个今天天气怎么样",
            clean="今天天气怎么样",
            injections=[{"kind": "filler", "token": "那个", "clause": 0}],
        ),
        _row(
            "zh001",
            "我我觉得可以",
            "我觉得可以",
            clean="我觉得可以",
            injections=[{"kind": "repetition", "token": "我", "clause": 0}],
        ),
    ]

    result = filler_retention(rows)

    assert result["retention"] == 0.5
    assert result["by_kind"]["filler"]["kept"] == 1
    assert result["by_kind"]["repetition"]["kept"] == 0


def test_the_error_split_names_substitutions_apart_from_insertions() -> None:
    """ "Which errors" decides what the polisher is trained to do; "how many" does not."""
    rows = [_row("zh000", "今天天气怎么样", "今天天汽怎么样")]

    shapes = error_shapes(rows, "reference_text")

    assert shapes["substituted"] == 1
    assert shapes["inserted"] == 0 and shapes["deleted"] == 0
    assert shapes["top_confusions"][0] == {"reference": "气", "heard": "汽", "n": 1}


def test_punctuation_is_read_off_the_raw_transcript() -> None:
    """The CER normaliser strips punctuation, so asking it this question always answers no."""
    rows = [_row("a", "你好", "你好。"), _row("b", "你好", "你好")]

    assert punctuated(rows) == 0.5


def test_an_untranscribed_row_stops_the_run(tmp_path: Path) -> None:
    """Scoring a missing transcript as an empty string reads as a perfect failure."""
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "zh000", "reference_text": "你好"}) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="asr_transcript"):
        load_rows(path)


def test_the_verdicts_follow_the_written_lines_not_the_numbers() -> None:
    """The thresholds are fixed before the run; this is what reads them back."""
    clean = arm([_row("a", "今天天气怎么样", "今天天气怎么样")], "reference_text", "clean")

    assert clean["cer_folded"] == 0.0
    assert clean["exact"] == 1

    kept = verdicts(clean, {"retention": 0.9})
    dropped = verdicts(clean, {"retention": 0.1})
    partial = verdicts(clean, {"retention": 0.5})

    assert "真活" in kept["口语词"]
    assert "基本不存在" in dropped["口语词"]
    assert "部分保留" in partial["口语词"]
    assert "纠错收益小" in kept["纠错"]
    assert "断句是主要收益" in kept["断句"]


def test_punctuation_agreement_asks_where_not_whether() -> None:
    """"Came back punctuated" and "broke in the right places" are different claims."""
    rows = [
        _row("a", "你好，今天天气怎么样？", "你好，今天天气怎么样？"),
        _row("b", "你好，今天天气怎么样？", "你好今天，天气怎么样？"),
    ]

    agreement = punctuation_agreement(rows, "reference_text")

    assert agreement["reference_breaks"] == 4
    assert agreement["heard_breaks"] == 4
    # Row a matches both marks; row b keeps the final one and moves the comma.
    assert agreement["matched"] == 3
    assert agreement["recall"] == 0.75
    assert agreement["precision"] == 0.75


def test_a_misheard_character_does_not_shift_every_later_break() -> None:
    """Positions are matched through the alignment, not by raw index."""
    rows = [_row("a", "你好，今天天气怎么样？", "你号，今天天气怎么样？")]

    agreement = punctuation_agreement(rows, "reference_text")

    assert agreement["matched"] == 2
