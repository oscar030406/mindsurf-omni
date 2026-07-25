"""The Talker-isolated protocol: fixed text, forced, paired."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from scripts.evaluate_talker import EOS_TOKEN, SHAPES, forced_text_plan, load_texts

ROOT = Path(__file__).resolve().parent.parent


def test_the_plan_is_the_target_then_eos() -> None:
    """The whole point is that the text axis carries no sampling at all."""
    assert forced_text_plan([5, 6, 7]) == [5, 6, 7, EOS_TOKEN]
    assert forced_text_plan([]) == [EOS_TOKEN]


def test_both_shapes_are_stated_not_defaulted() -> None:
    """A shape left to defaults cost one training run and one wrong conclusion.

    "mindsurf" must carry our overrides explicitly; "upstream-default" being
    empty is itself a statement -- the trainer defaults ARE upstream's shape.
    """
    assert SHAPES["mindsurf"] == {"intermediate_size": 3584, "num_key_value_heads": 8}
    assert SHAPES["upstream-default"] == {}


def test_the_frozen_text_set_holds_what_the_protocol_needs() -> None:
    """160 texts long enough that one wrong character is 1%, not 11%.

    The floor-run lesson: CER normalises by reference length, so short texts
    make every error enormous and nothing is comparable across sets. This set
    exists to hold the text axis still; it must not quietly change shape.
    """
    rows = load_texts(ROOT / "configs" / "talker_texts_zh_v1.jsonl")

    assert len(rows) == 160
    assert len({row["id"] for row in rows}) == 160, "duplicate ids would double-count pairs"
    lengths = [len(row["text"]) for row in rows]
    assert statistics.median(lengths) > 60, "short texts inflate per-character error"
    assert all(row["prompt"].strip() for row in rows), "the forced turn needs its context"


def test_an_empty_target_is_refused_up_front(tmp_path: Path) -> None:
    """An empty text would be scored as the Talker producing silence."""
    bad = tmp_path / "texts.jsonl"
    bad.write_text(
        json.dumps({"id": "a", "prompt": "x", "text": " "}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(SystemExit, match="no text"):
        load_texts(bad)


def test_one_temperature_or_eight_but_not_four() -> None:
    """Four values have no meaning for eight codebooks, and broadcasting would invent one."""
    import pytest
    from scripts.evaluate_talker import parse_temperature

    assert parse_temperature("0.2") == 0.2
    assert parse_temperature("0.2,0.2,0.15,0.15,0.1,0.1,0.05,0.05") == [
        0.2,
        0.2,
        0.15,
        0.15,
        0.1,
        0.1,
        0.05,
        0.05,
    ]
    with pytest.raises(SystemExit, match="1 or 8"):
        parse_temperature("0.2,0.3,0.4,0.5")


def test_a_per_codebook_temperature_reaches_the_right_codebook() -> None:
    """The list is indexed by codebook, so an off-by-one would sample the wrong stack."""
    from scripts.evaluate_talker import AudioSampling

    sampling = AudioSampling(temperature=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    assert sampling.for_codebook(0) == 0.1
    assert sampling.for_codebook(7) == 0.8
    assert AudioSampling(temperature=0.2).for_codebook(5) == 0.2
