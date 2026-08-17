"""The frontier table's two jobs: aggregate honestly, and say what an interval means."""

from __future__ import annotations

import json

from scripts.polish_frontier import load_arm, numbers, verdict


def _row(**fields: float) -> dict:
    base = {
        "cer_after": 0.0,
        "content_kept": 1.0,
        "invented": 0.0,
        "filler_arrived": 1,
        "filler_removed": 1,
    }
    return {**base, **fields}


def test_the_stored_cer_is_re_derived_not_believed(tmp_path) -> None:
    """A file written before the numeral fold reached CER carries the unfolded
    number. Believing it puts two rulers in one column and ranks the arms by
    which day they were run on."""
    written = tmp_path / "arm.jsonl"
    written.write_text(
        json.dumps(
            {
                "id": "x",
                "source": "会议改到下午六点",
                "target": "会议改到下午六点",
                "polished": "会议改到下午6点",
                "cer_after": 0.25,  # what the old build wrote
                "content_kept": 1.0,
                "invented": 0.0,
                "filler_arrived": 0,
                "filler_removed": 0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_arm(written)[0]["cer_after"] == 0.0


def test_a_number_whose_interval_straddles_the_line_has_not_passed() -> None:
    """The whole reason this file exists: 0.9805 on 156 sentences did not pass."""
    assert verdict("内容保留", 0.9805, 0.9705, 0.9886) == "跨线"
    assert verdict("内容保留", 0.9850, 0.9810, 0.9890) == "过"
    assert verdict("内容保留", 0.9600, 0.9550, 0.9650) == "不过"


def test_the_lines_that_want_a_small_number_read_the_other_way() -> None:
    assert verdict("编造", 0.0100, 0.0060, 0.0150) == "过"
    assert verdict("编造", 0.0184, 0.0126, 0.0252) == "跨线"
    assert verdict("CER", 0.0496, 0.0450, 0.0540) == "不过"


def test_filler_clearance_weighs_by_how_much_filler_arrived() -> None:
    """Three fillers in one sentence are three chances, not one."""
    heavy = _row(filler_arrived=3, filler_removed=0)
    light = _row(filler_arrived=1, filler_removed=1)

    assert numbers([heavy, light])["口语词清除"] == 0.25


def test_a_set_with_no_filler_reads_zero_rather_than_dividing_by_it() -> None:
    assert numbers([_row(filler_arrived=0, filler_removed=0)])["口语词清除"] == 0.0


def test_punctuation_kept_is_measured_against_the_ceiling_not_the_corpus() -> None:
    """The punctuation a polisher should keep is the recogniser's, minus
    whatever arrived attached to an injected filler."""
    from scripts.polish_frontier import punctuation_kept, surviving_punctuation

    # The ceiling drops 然后， entirely and keeps the rest.
    ceiling = {
        "a": {"id": "a", "source": "然后，今天天气好，我出门", "polished": "今天天气好，我出门"}
    }
    # This arm also ate the second comma.
    arm = [{"id": "a", "source": "然后，今天天气好，我出门", "polished": "今天天气好我出门"}]

    assert surviving_punctuation("然后，今天天气好，我出门", "今天天气好，我出门") == {8}
    assert punctuation_kept(arm, ceiling) == 0.0


def test_an_arm_that_keeps_every_comma_scores_one() -> None:
    from scripts.polish_frontier import punctuation_kept

    ceiling = {"a": {"id": "a", "source": "嗯，今天好，我出门", "polished": "今天好，我出门"}}
    arm = [{"id": "a", "source": "嗯，今天好，我出门", "polished": "今天好，我出门"}]

    assert punctuation_kept(arm, ceiling) == 1.0
