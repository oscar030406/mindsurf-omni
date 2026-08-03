"""The parts that decide whether the instrument may gate, without a network."""

from __future__ import annotations

from scripts.answerable_rate import answered, shuffled

ROWS = [{"id": f"zh{i:03d}", "prompt": f"问题{i}", "reply": f"回答{i}"} for i in range(20)]


def test_only_an_explicit_yes_counts_as_answered() -> None:
    """Defaulting the malformed case to yes would inflate the rate."""
    assert answered("是")
    assert answered("是的")
    assert not answered("否")
    assert not answered("")
    assert not answered("无法判断")


def test_the_shuffled_control_gives_nobody_their_own_reply() -> None:
    control = shuffled(ROWS, seed=7)
    assert len(control) == len(ROWS)
    for original, moved in zip(ROWS, control, strict=True):
        assert moved["prompt"] == original["prompt"]
        assert moved["reply"] != original["reply"], "对照臂把原回答还给了原问题"


def test_the_shuffled_control_reuses_every_reply_exactly_once() -> None:
    """A control that drops or repeats replies is measuring something else."""
    control = shuffled(ROWS, seed=7)
    assert sorted(row["reply"] for row in control) == sorted(row["reply"] for row in ROWS)


def test_the_shuffle_is_seeded() -> None:
    assert [r["reply"] for r in shuffled(ROWS, 7)] == [r["reply"] for r in shuffled(ROWS, 7)]
