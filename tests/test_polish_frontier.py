"""The frontier table's two jobs: aggregate honestly, and say what an interval means."""

from __future__ import annotations

from scripts.polish_frontier import numbers, verdict


def _row(**fields: float) -> dict:
    base = {
        "cer_after": 0.0,
        "content_kept": 1.0,
        "invented": 0.0,
        "filler_arrived": 1,
        "filler_removed": 1,
    }
    return {**base, **fields}


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
