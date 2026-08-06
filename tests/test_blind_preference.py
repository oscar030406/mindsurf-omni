"""The parts that decide the number: side assignment, ties, and position bias."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.blind_preference import (
    bin_tally,
    bin_verdict,
    build_pairs,
    load_partial,
    load_prebuilt,
    pair_key,
    tally,
)


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


def test_a_killed_run_gives_its_judgements_back(tmp_path: Path) -> None:
    """575 of 900 calls died with the session because nothing was written until the end.

    The rows are keyed, so resuming has to return them keyed -- a positional
    replay would hand the judge's verdict to whichever pair happened to sit at
    that index on the next run.
    """
    path = tmp_path / "judged.partial.jsonl"
    path.write_text(
        '{"key": "far#p1#0", "winner": "left"}\n{"key": "near#p2#0", "winner": "tie"}\n',
        encoding="utf-8",
    )

    assert load_partial(path) == {"far#p1#0": "left", "near#p2#0": "tie"}
    assert load_partial(tmp_path / "absent.jsonl") == {}


def test_a_half_written_last_line_is_dropped_not_repaired(tmp_path: Path) -> None:
    """A kill lands mid-write. Re-judging that pair costs a fraction of a cent;
    guessing at it attaches a wrong verdict to a real key."""
    path = tmp_path / "judged.partial.jsonl"
    path.write_text(
        '{"key": "a#0", "winner": "right"}\n{"key": "a#1", "winn',
        encoding="utf-8",
    )

    assert load_partial(path) == {"a#0": "right"}


def test_both_input_shapes_have_a_key_to_resume_on() -> None:
    """Prebuilt pairs carry ``key``; pairs built from two arms carry only ``id``."""
    assert pair_key({"key": "far#p1#0", "id": "p1"}) == "far#p1#0"
    assert pair_key({"id": "zh007"}) == "zh007"


def gap_bin(name: str, gap: int, n: int, longer_wins: int, ties: int = 0) -> list[dict]:
    """``longer_wins`` of ``n`` go to the longer draft, seats alternating."""
    rows = []
    for i in range(n):
        side = "left" if i % 2 else "right"
        other = "right" if side == "left" else "left"
        winner = "tie" if i < ties else side if i - ties < longer_wins else other
        rows.append(
            {"key": f"{name}#{i}", "bin": name, "gap": gap, "longer_side": side, "winner": winner}
        )
    return rows


def test_same_length_pairs_are_dropped_not_scored_against_the_longer_one() -> None:
    """A pair with a zero gap has no longer draft to win or lose."""
    rows = gap_bin("near", 4, 10, 5) + gap_bin("near", 0, 6, 0)
    for row in rows[10:]:
        row["key"] += "z"

    entry = bin_tally(rows)["bins"]["near"]

    assert entry["n"] == 10
    assert entry["same_length_dropped"] == 6
    assert entry["longer_share"] == 0.5


def test_ties_stay_out_of_the_share_and_get_their_own_rate() -> None:
    """Folding a tie in as half a win invents a preference; the rate is the point."""
    entry = bin_tally(gap_bin("near", 4, 100, 30, ties=40))["bins"]["near"]

    assert entry["n_decided"] == 60
    assert entry["tie_rate"] == 0.4
    assert entry["longer_share"] == 0.5


def test_L2_is_reachable_so_the_registered_falsifier_can_actually_fire() -> None:
    """The line the page most wants: three bins at chance, inference falsified."""
    rows = (
        gap_bin("near", 4, 300, 150) + gap_bin("mid", 33, 300, 150) + gap_bin("far", 75, 300, 152)
    )

    assert bin_verdict(bin_tally(rows))["line"] == "L2"


def test_L1_needs_both_the_level_and_the_spread() -> None:
    """0.60 in the far bin is not sensitivity if the near bin is already there."""
    high_both = (
        gap_bin("near", 4, 300, 195) + gap_bin("mid", 33, 300, 195) + gap_bin("far", 75, 300, 198)
    )
    assert bin_verdict(bin_tally(high_both))["line"] == "L3"

    spread = (
        gap_bin("near", 4, 300, 150) + gap_bin("mid", 33, 300, 175) + gap_bin("far", 75, 300, 210)
    )
    assert bin_verdict(bin_tally(spread))["line"] == "L1"


def test_the_registered_floor_is_three_sigma_not_the_one_point_nine_six() -> None:
    """n=300 was registered at ±0.087. 1.96σ would call 0.55 a result."""
    entry = bin_tally(gap_bin("far", 75, 300, 150))["bins"]["far"]

    assert entry["noise"] == pytest.approx(0.0866, abs=5e-4)


def test_a_verdict_that_flips_with_the_noise_floor_says_so() -> None:
    """Ties took half the near bin, so its floor is far wider than the registered
    one. L2 landing on the wider floor and not the planned one is the finding."""
    rows = (
        gap_bin("near", 4, 300, 59, ties=156)
        + gap_bin("mid", 33, 300, 101, ties=79)
        + gap_bin("far", 75, 300, 123, ties=30)
    )

    verdict = bin_verdict(bin_tally(rows))

    assert verdict["line"] == "L2"
    assert verdict["floor_sensitive"] is not None
    assert "near" not in verdict["floor_sensitive"]


def test_a_verdict_that_holds_on_both_floors_says_nothing() -> None:
    rows = (
        gap_bin("near", 4, 300, 150) + gap_bin("mid", 33, 300, 150) + gap_bin("far", 75, 300, 152)
    )

    assert bin_verdict(bin_tally(rows))["floor_sensitive"] is None
