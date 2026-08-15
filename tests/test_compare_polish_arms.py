"""Two arms compared on the same sentences, and the pairing that makes it honest."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_polish_arms import load, means


def _rows(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _row(identifier: str, **fields: float) -> dict:
    base = {
        "id": identifier,
        "cer_after": 0.0,
        "content_kept": 1.0,
        "invented": 0.0,
        "filler_arrived": 1,
        "filler_removed": 1,
    }
    return {**base, **fields}


def test_only_the_sentences_both_arms_ran_are_compared(tmp_path: Path) -> None:
    """One arm measured on 155 and the other on 156 is not a paired comparison."""
    left = load(_rows(tmp_path / "a.jsonl", [_row("x"), _row("y")]))
    right = load(_rows(tmp_path / "b.jsonl", [_row("y"), _row("z")]))

    assert sorted(set(left) & set(right)) == ["y"]


def test_filler_clearance_weighs_sentences_by_how_much_filler_they_carried() -> None:
    """A sentence with three fillers is three chances, not one."""
    heavy = _row("a", filler_arrived=3, filler_removed=0)
    light = _row("b", filler_arrived=1, filler_removed=1)

    # 1 of 4 removed, not the mean of 0.0 and 1.0.
    assert means([(heavy, heavy), (light, light)], 0)["口语词清除"] == 0.25


def test_a_set_with_no_filler_at_all_reads_zero_not_a_crash() -> None:
    """The clean fraction of the pool can dominate a small resample."""
    empty = _row("a", filler_arrived=0, filler_removed=0)

    assert means([(empty, empty)], 0)["口语词清除"] == 0.0
