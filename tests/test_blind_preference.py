"""The parts that decide the number: side assignment, ties, and position bias."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.blind_preference import build_pairs, load_prebuilt, tally


def arm(reply: str) -> dict[str, dict[str, str]]:
    return {
        f"zh{i:03d}": {"id": f"zh{i:03d}", "prompt": f"问题{i}", "reply": f"{reply}{i}"}
        for i in range(10)
    }


def test_sides_are_seeded_so_a_rerun_pairs_the_same_way() -> None:
    first = build_pairs(("base", arm("a")), ("cand", arm("b")), seed=7)
    again = build_pairs(("base", arm("a")), ("cand", arm("b")), seed=7)
    assert [p["second_side"] for p in first] == [p["second_side"] for p in again]


def test_both_sides_get_used() -> None:
    """A coin that never flips would hand the judge a fixed layout to learn."""
    sides = {p["second_side"] for p in build_pairs(("base", arm("a")), ("cand", arm("b")), 7)}
    assert sides == {"left", "right"}


def test_the_reply_actually_moves_with_the_side() -> None:
    for pair in build_pairs(("base", arm("a")), ("cand", arm("b")), seed=7):
        candidate = pair["left"] if pair["second_side"] == "left" else pair["right"]
        assert candidate.startswith("b"), "候选臂的回答没跟着 second_side 走"


def test_ties_are_counted_not_split() -> None:
    """Half a win is a preference the judge declined to express."""
    pairs = [
        {"winner": "left", "second_side": "left"},
        {"winner": "right", "second_side": "left"},
        {"winner": "tie", "second_side": "left"},
        {"winner": "tie", "second_side": "right"},
    ]
    result = tally(pairs, "cand")
    assert result["tie"] == 2
    assert result["n_decided"] == 2
    assert result["cand_share"] == 0.5


def test_a_judge_that_always_picks_the_second_slot_is_visible() -> None:
    pairs = [{"winner": "right", "second_side": s} for s in ("left", "right") * 5]
    assert tally(pairs, "cand")["position_right_share"] == 1.0


def test_a_win_has_to_clear_the_noise_floor() -> None:
    """Ten decided pairs cannot resolve a small edge, so it is not a win."""
    pairs = [{"winner": "left", "second_side": "left"} for _ in range(6)]
    pairs += [{"winner": "right", "second_side": "left"} for _ in range(4)]
    result = tally(pairs, "cand")
    assert result["cand_share"] == 0.6
    assert result["verdict"] == "未胜出"


def test_a_side_preference_that_randomisation_cancels_is_not_a_contaminated_estimate() -> None:
    """The first rule flagged the judge; the estimate it protects was fine.

    A judge that always picks left, with seats exactly even, leaves a pooled
    share of 0.5 -- no leak at all -- while share-of-right reads 0.0.
    """
    pairs = [{"winner": "left", "second_side": s} for s in ("left", "right") * 50]
    result = tally(pairs, "cand")

    assert result["position_right_share"] == 0.0
    assert result["seating"]["side_bias"] == 0.5
    assert result["seating"]["residual_bias"] == 0.0
    assert result["cand_share"] == 0.5


def test_the_leak_is_the_side_bias_times_the_seat_imbalance() -> None:
    """Uneven seats are what turn a judge's preference into a wrong number."""
    pairs = [{"winner": "left", "second_side": "left"} for _ in range(75)]
    pairs += [{"winner": "left", "second_side": "right"} for _ in range(25)]
    seat = tally(pairs, "cand")["seating"]

    # Always-left judge, 75% of candidates seated left: the share is inflated
    # by the whole imbalance rather than by nothing.
    assert seat["side_bias"] == 0.5
    assert seat["left_seat_share"] == 0.75
    assert round(seat["residual_bias"], 4) == 0.25
    assert seat["seat_balanced_share"] == 0.5


def test_one_sided_seating_reports_that_it_cannot_be_estimated() -> None:
    """A coin that never varied cannot cancel anything, and says so."""
    seat = tally([{"winner": "left", "second_side": "left"} for _ in range(20)], "cand")["seating"]

    assert seat["side_bias"] is None
    assert "无法" in seat["note"]


def test_prebuilt_pairs_without_keys_are_refused(tmp_path: Path) -> None:
    """A judgement is attached by key, so a pair without one lands on the wrong text.

    One prompt contributes several pairs. Keying by prompt id silently collapses
    them and half the labels end up on a pair the judge never saw -- which is why
    build_preference_pairs makes the key unique per pair and refuses duplicates.
    """
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps({"pairs": [{"id": "a", "prompt": "问", "left": "甲", "right": "乙"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="key"):
        load_prebuilt(path)


def test_prebuilt_pairs_are_not_reshuffled(tmp_path: Path) -> None:
    """Sides were assigned when the pairs were formed, and the key kept apart.

    Re-randomising here would detach the judgement from the pair the judge was
    shown, which is the same failure as keying by prompt, arrived at differently.
    """
    pairs = [
        {"key": "a#0", "id": "a", "prompt": "问", "left": "甲", "right": "乙"},
        {"key": "a#1", "id": "a", "prompt": "问", "left": "丙", "right": "丁"},
    ]
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")

    loaded = load_prebuilt(path)

    assert [p["key"] for p in loaded] == ["a#0", "a#1"]
    assert [(p["left"], p["right"]) for p in loaded] == [("甲", "乙"), ("丙", "丁")]
