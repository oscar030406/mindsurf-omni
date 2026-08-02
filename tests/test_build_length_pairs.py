"""What would silently produce a training set that teaches the wrong thing.

Two failures matter here. A pair whose two sides are nearly the same length
spends an update on whatever else differs between them while teaching nothing
about brevity -- and it looks like a perfectly good pair in the output file. And
a shortest draft that does not answer would teach the model that saying nothing
wins, which is precisely the collapse this round's guard exists to prevent.
"""

from __future__ import annotations

from scripts.build_length_pairs import MINIMUM_GAP_CHARS, pick


def _sample(reply: str) -> dict[str, str]:
    return {"reply": reply, "prompt": "问题"}


def test_takes_the_extremes_not_the_middle() -> None:
    samples = [_sample("短" * 10), _sample("中" * 60), _sample("长" * 200)]

    chosen = pick(samples)

    assert chosen is not None
    short, long = chosen
    assert len(short["reply"]) == 10
    assert len(long["reply"]) == 200


def test_a_pair_with_no_length_signal_is_dropped() -> None:
    gap = MINIMUM_GAP_CHARS - 1
    samples = [_sample("字" * 100), _sample("字" * (100 + gap))]

    assert pick(samples) is None


def test_an_empty_draft_is_not_eligible_to_be_the_short_side() -> None:
    """Otherwise the shortest is always the empty one and every pair teaches silence."""
    samples = [_sample("   "), _sample("有内容的回答" * 3), _sample("很长的回答" * 40)]

    chosen = pick(samples)

    assert chosen is not None
    assert chosen[0]["reply"].strip()


def test_one_usable_draft_is_not_a_pair() -> None:
    assert pick([_sample("只有一条")]) is None
    assert pick([_sample(""), _sample("只有一条")]) is None


def test_chosen_is_the_shortest_that_answers_not_the_shortest() -> None:
    """Screened over 771 prompts, only 191 shortest drafts still answered.

    Preferring the shortest outright would have taught the model that the
    broken-but-brief reply wins, which is the collapse this whole round guards
    against.
    """
    samples = [_sample("太短没答"), _sample("短而且答了" * 4), _sample("很长的回答" * 40)]

    chosen = pick(samples, answers={0: False, 1: True, 2: True})

    assert chosen is not None
    assert chosen[0]["reply"].startswith("短而且答了")


def test_no_draft_answers_means_no_pair() -> None:
    samples = [_sample("坏的短回答"), _sample("坏的长回答" * 40)]

    assert pick(samples, answers={0: False, 1: False}) is None


def test_the_only_answering_draft_being_the_longest_yields_no_pair() -> None:
    """chosen and rejected would be the same row, which is not a preference."""
    samples = [_sample("坏的短回答"), _sample("好的长回答" * 40)]

    assert pick(samples, answers={0: False, 1: True}) is None
