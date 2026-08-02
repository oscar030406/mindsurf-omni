"""The two numbers that would be wrong in a way nobody notices.

Occupancy divides by the prompt's length, so a promptless probe set would
either crash or silently produce infinities that poison the median. And
"unfinished at T" is the one a reader will quote, so its direction has to be
pinned: it counts replies still going, not replies finished.
"""

from __future__ import annotations

from scripts.measure_turn_taking import measure, seconds


def _reply(text: str, prompt: str = "问题") -> dict[str, str]:
    return {"reply": text, "prompt": prompt}


def test_unfinished_counts_the_ones_still_going() -> None:
    # 4.67 chars a second: 47 chars is about 10s, 5 chars is about 1s
    short, long = _reply("短" * 5), _reply("长" * 47)

    result = measure([short, long, short, long])

    assert result["unfinished_at"]["3.0"] == 0.5  # both long ones still going
    assert result["unfinished_at"]["30.0"] == 0.0  # nothing runs that long


def test_a_promptless_probe_does_not_poison_occupancy() -> None:
    """A zero-length prompt makes the ratio infinite, and one infinity in the
    list would make the median meaningless without raising anything."""
    result = measure([_reply("回答内容", prompt=""), _reply("回答内容", prompt="问")])

    assert result["occupancy"]["median"] == 4.0
    assert result["replies"] == 2


def test_seconds_scale_with_length() -> None:
    assert seconds("四个字符") < seconds("四个字符" * 3)
    assert seconds("") == 0.0
