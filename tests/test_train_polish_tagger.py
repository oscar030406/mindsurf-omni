"""Per-token judgement instead of rewriting: the labels are the whole risk."""

from __future__ import annotations

from scripts.train_polish_tagger import label_tokens


def test_a_token_whose_characters_all_vanish_is_a_delete() -> None:
    source, target = "那个今天天气怎么样", "今天天气怎么样"
    spans = [(0, 2), (2, 4), (4, 6), (6, 9)]  # 那个 | 今天 | 天气 | 怎么样

    assert label_tokens(source, target, spans) == [1, 0, 0, 0]


def test_a_partly_surviving_token_is_kept() -> None:
    """Deleting it would take content with it, and content loss is the failure
    this round exists to fix."""
    source, target = "微波炉里", "微炉里"
    spans = [(0, 3), (3, 4)]  # 微波炉 | 里 -- 波 is gone, 微 and 炉 survive

    assert label_tokens(source, target, spans) == [0, 0]


def test_nothing_survives_means_everything_is_deleted() -> None:
    assert label_tokens("嗯呃", "", [(0, 1), (1, 2)]) == [1, 1]


def test_a_misrecognised_character_is_not_labelled_delete() -> None:
    """The 12.6% that made the first version unlearnable.

    汽 is what the recogniser wrote for 气. It is wrong, but nothing in the
    input says so, and deleting it loses the character rather than fixing it.
    """
    source, target = "今天天汽怎么样", "今天天气怎么样"
    spans = [(0, 2), (2, 4), (4, 7)]  # 今天 | 天汽 | 怎么样

    assert label_tokens(source, target, spans) == [0, 0, 0]


def test_filler_next_to_a_misrecognition_is_still_deleted() -> None:
    source, target = "那个今天天汽怎么样", "今天天气怎么样"
    spans = [(0, 2), (2, 4), (4, 6), (6, 9)]  # 那个 | 今天 | 天汽 | 怎么样

    assert label_tokens(source, target, spans) == [1, 0, 0, 0]
