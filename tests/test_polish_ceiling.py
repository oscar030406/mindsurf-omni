"""The ceiling row, and why the floor row belongs next to it."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.polish_ceiling import main


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _run(tmp_path: Path, mode: str) -> list[dict]:
    pool = _write(tmp_path / "pool.jsonl", [{"id": "a", "text": "白衬衫领子发黄怎么办"}])
    pairs = _write(
        tmp_path / "pairs.jsonl",
        [
            {
                "id": "a",
                "source": "然后，白衬衫领子发黄怎么办？",
                "target": "白衬衫领子发黄怎么办？",
                "spoken": "然后，白衬衫领子发黄怎么办",
                "injections": [{"kind": "filler", "token": "然后", "clause": 0}],
                "split": "val",
            }
        ],
    )
    output = tmp_path / "out.jsonl"
    import sys

    argv = sys.argv
    sys.argv = [
        "polish_ceiling.py",
        "--pairs",
        str(pairs),
        "--pool",
        str(pool),
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    try:
        main()
    finally:
        sys.argv = argv
    lines = output.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_the_ceiling_deletes_exactly_the_injected_filler(tmp_path: Path) -> None:
    rows = _run(tmp_path, "perfect")

    assert rows[0]["polished"] == "白衬衫领子发黄怎么办？"
    assert rows[0]["filler_removed"] == rows[0]["filler_arrived"]


def test_the_floor_touches_nothing(tmp_path: Path) -> None:
    """It belongs in the same table: content retention is maximised by doing
    nothing, so clearing that line alone does not mean any work happened."""
    rows = _run(tmp_path, "nothing")

    assert rows[0]["polished"] == rows[0]["source"]
    assert rows[0]["filler_removed"] == 0


def test_both_modes_cover_every_sentence(tmp_path: Path) -> None:
    """A ceiling measured on a different set than the arms is not a ceiling."""
    assert len(_run(tmp_path, "perfect")) == len(_run(tmp_path, "nothing")) == 1
