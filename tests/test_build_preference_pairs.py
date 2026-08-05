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


def test_several_pairs_from_one_prompt_keep_separate_keys(tmp_path: Path) -> None:
    """The bug this replaced: keying by prompt id collapsed them.

    A prompt contributes several pairs. Keyed by prompt id, the last one wins
    the dictionary slot and every judgement for that prompt attaches to it --
    so half the labels describe text the judge never saw, and nothing about the
    resulting file looks wrong.
    """
    drafts = {
        "x": "第一条回答，长度足够拿来比较，不会被短句筛掉。",
        "y": "第二条回答，同样够长，内容不一样。",
        "z": "第三条回答，也够长，还是不一样的内容。",
    }
    reports = [_report(tmp_path, f"{name}.json", "m", {"x": text}) for name, text in drafts.items()]

    report = build(argparse.Namespace(samples=reports, pairs_per_prompt=3, seed=1))

    keys = [item["key"] for item in report["pairs"]]
    assert len(report["pairs"]) == 3
    assert len(set(keys)) == 3, keys
    assert all(key.startswith("x#") for key in keys)


def test_ties_are_dropped_rather_than_counted_as_half(tmp_path: Path) -> None:
    """A pair the judge could not separate has no direction to train toward."""
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            {
                "checkpoint": "m",
                "pairs": [
                    {
                        "key": "a#0",
                        "id": "a",
                        "prompt": "问",
                        "left": "甲的回答",
                        "right": "乙的回答",
                    },
                    {
                        "key": "a#1",
                        "id": "a",
                        "prompt": "问",
                        "left": "丙的回答",
                        "right": "丁的回答",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    judged = tmp_path / "judged.json"
    judged.write_text(
        json.dumps([{"key": "a#0", "winner": "left"}, {"key": "a#1", "winner": "tie"}]),
        encoding="utf-8",
    )

    out = resolve(argparse.Namespace(pairs=pairs, judged=judged))

    assert out["dropped_ties"] == 1
    assert out["triples"] == [{"prompt": "问", "chosen": "甲的回答", "rejected": "乙的回答"}]
    # Provenance survives the stage, so the trainer can re-assert on-policy.
    assert out["checkpoint"] == "m"


def test_pair_sampling_stays_linear_in_the_number_of_prompts() -> None:
    """All-pairs is quadratic and buys little; the cap is what keeps it affordable."""
    drafts = ["a", "b", "c", "d", "e"]

    drawn = draw_pairs(drafts, 2, random.Random(0))

    assert len(drawn) == 2
    assert len(set(drawn)) == 2
    assert draw_pairs(["only-one"], 2, random.Random(0)) == []


def test_a_wide_length_gap_never_reaches_a_judge() -> None:
    """The fourth way a preference set stops being about preference.

    A judge shown a long draft against a short one is partly ranking length.
    That is measured here rather than assumed: the length-controlled diagnostic
    put the judge's own length bias at 0.5092, and it only read that neutral
    once the two arms were within a character of each other at the median.
    Leave the gap open and the gradient carries length whether or not anyone
    meant it -- and a length intervention is exactly what the conversation
    guardrail cannot judge.
    """
    drafts = ["短" * 10, "短" * 14, "长" * 200]

    within = draw_pairs(drafts, per_prompt=99, rng=random.Random(0), max_length_gap=10)

    assert (0, 1) in within or (1, 0) in within
    assert all(abs(len(drafts[i]) - len(drafts[j])) <= 10 for i, j in within)
    # Without the gap the long draft is paired with both short ones.
    assert len(draw_pairs(drafts, per_prompt=99, rng=random.Random(0))) == 3


def test_a_prompt_whose_drafts_are_all_far_apart_is_dropped_with_a_reason(
    tmp_path: Path,
) -> None:
    """Silently producing nothing for a prompt looks like the prompt was fine."""
    # Both drafts have to survive the repetition screen, or this would be
    # testing that screen instead of the length gap.
    report = _report(tmp_path, "s1.json", "sft_merge", {"a": "今天天气不错出门走走"})
    other = _report(
        tmp_path,
        "s2.json",
        "sft_merge",
        {
            "a": "关于天气这件事要看季节和地区，春天多雨秋天干燥，出门前查一下预报比较稳妥，"
            "带伞或者带外套都取决于当天的实际情况而不是季节的平均值"
        },
    )

    built = build(
        argparse.Namespace(samples=[report, other], pairs_per_prompt=2, seed=1, max_length_gap=20)
    )

    assert built["pairs"] == []
    assert built["dropped_drafts"]["no pair within the length gap"] == 1


def test_the_resolved_set_reports_the_length_signal_it_carries(tmp_path: Path) -> None:
    """A round meant to be about quality can still hand over a length signal.

    The judge has its own preference and the sampler its own spread, and the
    only place that shows up before training is here.
    """
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            {
                "checkpoint": "sft_merge",
                "pairs": [
                    {
                        "key": "a#0",
                        "id": "a",
                        "prompt": "问",
                        "left": "长" * 30,
                        "right": "短" * 10,
                    },
                    {
                        "key": "b#0",
                        "id": "b",
                        "prompt": "问",
                        "left": "长" * 30,
                        "right": "短" * 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    judged = tmp_path / "judged.json"
    judged.write_text(
        json.dumps([{"key": "a#0", "winner": "left"}, {"key": "b#0", "winner": "left"}]),
        encoding="utf-8",
    )

    out = resolve(argparse.Namespace(pairs=pairs, judged=judged, output=tmp_path / "o.json"))

    assert out["chosen_minus_rejected_chars"] == 20.0
    assert out["chosen_longer_share"] == 1.0
