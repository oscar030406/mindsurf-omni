"""Two things decide whether this pack can answer anything.

The sample must be stratified over the assigned label -- a proportional draw
spends two thirds of the budget on neutral and leaves too few of the rare
classes to say anything about them. And the token must not leak the label, or
a rater who notices the pattern stops judging by ear.
"""

from __future__ import annotations

import collections

from scripts.build_label_check import LABEL_TO_CHOICE, pick, token_for


def _rows() -> list[dict[str, object]]:
    rows = []
    for index in range(200):
        label = "中立/neutral" if index < 160 else "开心/happy"
        rows.append({"row_group": index // 50, "index": index, "label": label})
    for index in range(200, 205):
        rows.append({"row_group": 9, "index": index, "label": "难过/sad"})
    return rows


def test_rare_labels_are_not_crowded_out_by_the_common_one() -> None:
    picked = pick(_rows(), per_label=10, seed=1)

    counts = collections.Counter(row["label"].split("/")[-1] for row in picked)
    assert counts["neutral"] == 10
    assert counts["happy"] == 10
    # only five exist, and taking all five beats taking a proportional two
    assert counts["sad"] == 5


def test_labels_outside_the_closed_set_are_dropped() -> None:
    """厌恶 / 恐惧 / other were 7 clips in 1000 -- offering a rater an option
    that never appears makes them reach for it."""
    rows = _rows() + [{"row_group": 0, "index": 999, "label": "厌恶/disgusted"}]

    picked = pick(rows, per_label=10, seed=1)

    assert all(row["label"].split("/")[-1] in LABEL_TO_CHOICE for row in picked)


def test_the_token_does_not_encode_the_label() -> None:
    same_clip = {"row_group": 3, "index": 7}

    assert token_for(same_clip, "salt") == token_for({**same_clip, "label": "开心/happy"}, "salt")
    assert token_for(same_clip, "salt") != token_for({"row_group": 3, "index": 8}, "salt")
