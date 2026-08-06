"""The parts that decide the number: what counts as a flip, and what a flip means."""

from __future__ import annotations

from scripts.judge_reliability import compare, flip_shape, swap, verdict


def pair(key: str, bin_name: str, left: str, right: str, winner: str | None) -> dict:
    row = {
        "key": key,
        "bin": bin_name,
        "gap": 30,
        "longer_side": "left",
        "left": left,
        "right": right,
    }
    if winner:
        row["winner"] = winner
    return row


def swapped(origin: dict, winner: str | None) -> dict:
    row = swap([origin])[0]
    if winner:
        row["winner"] = winner
    return row


def test_a_verdict_that_follows_the_text_across_the_swap_is_not_a_flip() -> None:
    """The seats are what changed, so comparing seats calls every consistent
    judgement a flip -- the exact opposite of the reading."""
    origin = pair("a#0", "mid", "甲", "乙", "left")  # picked 甲
    after = swapped(origin, "right")  # 甲 now on the right

    report = compare([origin], [after])

    assert report["bins"]["mid"]["flips"] == 0
    assert report["bins"]["mid"]["decided_twice"] == 1


def test_picking_the_other_text_is_a_flip() -> None:
    origin = pair("a#0", "mid", "甲", "乙", "left")  # picked 甲
    after = swapped(origin, "left")  # left is now 乙

    assert compare([origin], [after])["bins"]["mid"]["flips"] == 1


def test_a_tie_on_either_round_leaves_the_flip_rate_alone() -> None:
    """A pair the judge would not separate says nothing about repeatability,
    but where the ties went is its own evidence and gets counted."""
    origin = pair("a#0", "mid", "甲", "乙", "left")
    after = swapped(origin, "tie")

    entry = compare([origin], [after])["bins"]["mid"]

    assert entry["decided_twice"] == 0
    assert "flip_rate" not in entry
    assert entry["transitions"]["decided_tie"] == 1


def test_the_swap_does_not_carry_the_old_verdict_into_the_new_file() -> None:
    """The judge's fresh answer is written into this file; a leftover winner
    would be indistinguishable from it."""
    origin = pair("a#0", "mid", "甲", "乙", "left")

    row = swap([origin])[0]

    assert "winner" not in row
    assert (row["left"], row["right"]) == ("乙", "甲")
    assert row["key"] == "a#0~swap" and row["of_key"] == "a#0"
    assert row["longer_side"] == "right"


def test_a_judge_that_always_takes_the_left_seat_is_told_apart_from_noise() -> None:
    """Every flip going left twice is a seat preference, not unrepeatability --
    and seat randomisation already cancels the first one."""
    matched = []
    for i in range(60):
        origin = pair(f"a#{i}", "mid", f"甲{i}", f"乙{i}", "left")
        matched.append((origin, swapped(origin, "left")))

    shape = flip_shape(matched)

    assert shape["flips"] == 60
    assert shape["left_share"] == 1.0
    assert shape["reads_as"] == "座位偏好"


def test_evenly_split_flips_read_as_noise() -> None:
    matched = []
    for i in range(200):
        origin = pair(f"a#{i}", "mid", f"甲{i}", f"乙{i}", "left" if i % 2 else "right")
        matched.append((origin, swapped(origin, "left" if i % 2 else "right")))

    shape = flip_shape(matched)

    assert shape["flips"] == 200
    assert shape["left_share"] == 0.5
    assert shape["reads_as"].startswith("对称")


def test_R2_needs_one_bin_at_thirty_percent_not_an_average() -> None:
    report = {
        "bins": {
            "near": {"flip_rate": 0.05},
            "mid": {"flip_rate": 0.05},
            "far": {"flip_rate": 0.31},
        }
    }
    assert verdict(report)["line"] == "R2"

    report = {
        "bins": {
            "near": {"flip_rate": 0.09},
            "mid": {"flip_rate": 0.10},
            "far": {"flip_rate": 0.08},
        }
    }
    assert verdict(report)["line"] == "R1"


def test_the_registered_prediction_is_scored_separately_from_the_line() -> None:
    """near > mid > far was registered as falsifiable and is not part of R1/R2."""
    held = {
        "bins": {
            "near": {"flip_rate": 0.30},
            "mid": {"flip_rate": 0.20},
            "far": {"flip_rate": 0.10},
        }
    }
    assert verdict(held)["prediction_near_gt_mid_gt_far"]["held"] is True

    broke = {
        "bins": {
            "near": {"flip_rate": 0.24},
            "mid": {"flip_rate": 0.21},
            "far": {"flip_rate": 0.26},
        }
    }
    assert verdict(broke)["prediction_near_gt_mid_gt_far"]["held"] is False
