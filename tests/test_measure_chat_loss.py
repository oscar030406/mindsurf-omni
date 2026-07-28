"""The boundary between the prompt and the reply, which is the whole instrument.

Scoring the prompt as well as the reply would not fail, it would just dilute
every difference toward zero and report "indistinguishable" -- the failure mode
this project has already paid for once, on a repetition metric that gave a
seven-character answer full marks.

The forward pass needs torch and a checkpoint. The offset arithmetic does not,
and it is the part that goes wrong silently.
"""

from __future__ import annotations

from scripts.measure_chat_loss import reply_span


def test_the_reply_starts_where_the_prompt_ends() -> None:
    assert reply_span([1, 2, 3], [1, 2, 3, 4, 5]) == slice(3, 5)


def test_a_merged_boundary_is_dropped_rather_than_scored_at_the_wrong_offset() -> None:
    """Tokenisers may merge across the join, and then len(prefix) is a lie.

    Scoring from there charges the model for a token it was handed, which
    lowers the loss of whichever arm happens to merge -- a difference between
    tokenisations reported as a difference between models.
    """
    assert reply_span([1, 2, 3], [1, 2, 9, 4, 5]) is None


def test_an_empty_reply_is_dropped() -> None:
    assert reply_span([1, 2, 3], [1, 2, 3]) is None
    assert reply_span([1, 2, 3], [1, 2]) is None
