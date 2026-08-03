"""The parts that decide the number: side assignment, ties, and position bias."""

from __future__ import annotations

from scripts.blind_preference import build_pairs, tally


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
