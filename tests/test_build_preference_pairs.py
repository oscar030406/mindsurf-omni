"""The three ways a preference set silently stops being about preference.

Mixing checkpoints turns it into a ranking of models rather than of responses;
keeping ties trains toward a coin flip; and letting a looping draft into the
pool gets it ranked confidently for reasons unrelated to whether it is better.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pytest
from scripts.build_preference_pairs import build, draw_pairs, resolve, usable


def _report(tmp_path: Path, name: str, checkpoint: str, replies: dict[str, str]) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "replies": [
                    {"id": key, "prompt": f"问{key}", "reply": value}
                    for key, value in replies.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_samples_from_two_checkpoints_are_refused(tmp_path: Path) -> None:
    """Otherwise the pairs rank checkpoints, and DPO learns to imitate the winner."""
    one = _report(tmp_path, "a.json", "sft_graft", {"x": "今天天气不错，适合出门散步走走。"})
    two = _report(tmp_path, "b.json", "t2a_graft", {"x": "天气挺好的，可以出去转转。"})

    args = argparse.Namespace(samples=[one, two], pairs_per_prompt=1, seed=0)
    with pytest.raises(SystemExit, match="rank checkpoints"):
        build(args)


def test_a_looping_draft_never_reaches_a_judge() -> None:
    ok, why = usable("好的好的好的好的好的好的好的好的好的好的好的")
    assert not ok and why == "looping"

    empty_ok, empty_why = usable("   ")
    assert not empty_ok and empty_why == "empty"

    fine, _ = usable("今天天气晴朗，气温大约二十度，很适合出门散步。")
    assert fine


def test_a_prompt_with_one_usable_draft_is_dropped_not_paired(tmp_path: Path) -> None:
    one = _report(tmp_path, "a.json", "m", {"x": "这是一条正常的回答，长度也够，可以拿来比较。"})
    two = _report(tmp_path, "b.json", "m", {"x": "好的好的好的好的好的好的好的好的好的好的"})

    report = build(argparse.Namespace(samples=[one, two], pairs_per_prompt=2, seed=0))

    assert report["pairs"] == []
    assert report["dropped_drafts"]["looping"] == 1
    assert report["dropped_drafts"]["too few usable drafts"] == 1


def test_ties_are_dropped_rather_than_counted_as_half(tmp_path: Path) -> None:
    """A pair the judge could not separate has no direction to train toward."""
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            {
                "pairs": [
                    {"id": "a", "prompt": "问", "left": "甲的回答", "right": "乙的回答"},
                    {"id": "b", "prompt": "问", "left": "丙的回答", "right": "丁的回答"},
                ]
            }
        ),
        encoding="utf-8",
    )
    judged = tmp_path / "judged.json"
    judged.write_text(
        json.dumps([{"id": "a", "winner": "left"}, {"id": "b", "winner": "tie"}]),
        encoding="utf-8",
    )

    out = resolve(argparse.Namespace(pairs=pairs, judged=judged))

    assert out["dropped_ties"] == 1
    assert out["triples"] == [{"prompt": "问", "chosen": "甲的回答", "rejected": "乙的回答"}]


def test_pair_sampling_stays_linear_in_the_number_of_prompts() -> None:
    """All-pairs is quadratic and buys little; the cap is what keeps it affordable."""
    drafts = ["a", "b", "c", "d", "e"]

    drawn = draw_pairs(drafts, 2, random.Random(0))

    assert len(drawn) == 2
    assert len(set(drawn)) == 2
    assert draw_pairs(["only-one"], 2, random.Random(0)) == []
