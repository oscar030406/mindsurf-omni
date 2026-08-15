"""The pairs the polisher trains on, and the two ways they go wrong."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from scripts.build_polish_pairs import GARBLED, load_texts, speak_many


class _Synthesiser:
    """Speaks everything except the text it was told to refuse."""

    def __init__(self, refuse: str = "") -> None:
        self.said: list[str] = []
        self.refuse = refuse

    async def synthesise(self, utterance) -> bytes:  # type: ignore[no-untyped-def]
        self.said.append(utterance.text)
        if self.refuse and self.refuse in utterance.text:
            raise RuntimeError("the endpoint returned no audio")
        return b"\x00\x01" * 100


def _pool(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pool.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_the_same_sentence_twice_is_one_target(tmp_path: Path) -> None:
    """The pool is assembled from files that overlap; a duplicate target is free noise."""
    path = _pool(
        tmp_path,
        [
            {"id": "a", "text": "今天天气怎么样"},
            {"id": "b", "text": "今天天气怎么样"},
            {"id": "c", "text": "帮我推荐一部电影"},
            {"id": "d", "text": "   "},
        ],
    )

    rows = load_texts([path], limit=None)

    assert [row["id"] for row in rows] == ["a", "c"]


def test_a_row_without_an_id_still_gets_one(tmp_path: Path) -> None:
    path = _pool(tmp_path, [{"text": "今天天气怎么样"}])

    assert load_texts([path], limit=None)[0]["id"] == "t000000"


def test_a_refused_clip_comes_back_empty_rather_than_killing_the_batch() -> None:
    """One bad clip is a dropped pair; raising would lose the whole batch."""
    synthesiser = _Synthesiser(refuse="坏")

    audio = asyncio.run(speak_many(synthesiser, ["好句子", "坏句子", "另一句"], concurrency=2))

    assert [bool(piece) for piece in audio] == [True, False, True]
    assert len(synthesiser.said) == 3


def test_the_garbled_line_is_far_above_the_measured_floor() -> None:
    """0.30 is a synthesis failure; the floor run's median was 0.008."""
    assert GARBLED > 0.1


def test_the_target_gets_the_final_mark_the_transcript_already_wrote() -> None:
    """Half the pool is prompts written without one, and deleting it is not the task."""
    from scripts.build_polish_pairs import align_final_punctuation

    assert (
        align_final_punctuation("白衬衫领子发黄怎么办", "白衬衫领子发黄怎么办？")
        == "白衬衫领子发黄怎么办？"
    )
    # Already punctuated: untouched, including when the two disagree -- the
    # target is the sentence a person meant and this is not a rewrite.
    assert align_final_punctuation("今天天气怎么样。", "今天天气怎么样？") == "今天天气怎么样。"
    assert align_final_punctuation("今天天气怎么样", "今天天气怎么样") == "今天天气怎么样"
