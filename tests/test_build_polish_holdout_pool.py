"""The held-out pool, and the one thing that would silently ruin it."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_polish_holdout_pool import LONGEST, SHORTEST, harvest


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_a_sentence_the_training_pool_already_has_is_not_held_out(tmp_path: Path) -> None:
    """Deduplicated by the sentence, not the id.

    The same sentence reached several probe files under different ids. Held out
    under one of them, it would be a sentence the checkpoint trained on, and
    every number read on it would be optimistic for a reason nobody could see.
    """
    trained_on = {"今天天气怎么样"}
    path = _jsonl(tmp_path / "a.jsonl", [{"text": "今天天气怎么样"}, {"text": "帮我推荐电影"}])
    found = harvest(path)

    fresh = [text for text in found if text not in trained_on]

    assert fresh == ["帮我推荐电影"]


def test_a_sentence_is_taken_from_whichever_field_holds_it(tmp_path: Path) -> None:
    """The probe files and the reply archives disagree on the key name."""
    found = harvest(
        _jsonl(
            tmp_path / "b.jsonl",
            [{"prompt": "怎么煮面"}, {"reply": "水要宽，水开再下面"}, {"speaker": "张三"}],
        )
    )

    assert set(found) == {"怎么煮面", "水要宽，水开再下面"}


def test_the_length_bounds_match_the_training_pool(tmp_path: Path) -> None:
    """Over 160 characters is not one breath, and the context has to hold both sides."""
    found = harvest(
        _jsonl(
            tmp_path / "c.jsonl",
            [{"text": "短"}, {"text": "刚好够长的一句话"}, {"text": "长" * (LONGEST + 1)}],
        )
    )

    assert found == ["刚好够长的一句话"]
    assert SHORTEST < len("刚好够长的一句话") <= LONGEST


def test_a_line_that_is_not_json_does_not_stop_the_harvest(tmp_path: Path) -> None:
    """These files are archives written by several scripts over several weeks."""
    path = tmp_path / "d.jsonl"
    path.write_text('{"text": "能读的这条"}\n不是 json 的一行\n', encoding="utf-8")

    assert harvest(path) == ["能读的这条"]
