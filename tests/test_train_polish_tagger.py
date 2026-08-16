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


def test_repetition_columns_mark_both_copies() -> None:
    """The head reads one position at a time, so 时间时间 looks like two
    ordinary words. It clears 0.437 of the injected repetition against the
    generative arm's 0.603, and this is the column that difference lives in."""
    import torch
    from scripts.train_polish_tagger import repetition_features

    # ids 1,2 repeated at 3,4 -> length-2 repetition opening at 1, closing at 3.
    out = repetition_features([9, 1, 2, 1, 2, 8], torch, "cpu", 2)

    opens, closes = out[:, 2], out[:, 3]  # length 2 columns
    assert opens.tolist() == [0, 1, 1, 0, 0, 0]
    assert closes.tolist() == [0, 0, 0, 1, 1, 0]


def test_a_sequence_with_no_repetition_is_all_zero() -> None:
    import torch
    from scripts.train_polish_tagger import repetition_features

    assert repetition_features([1, 2, 3, 4], torch, "cpu", 2).sum().item() == 0


def test_the_old_width_is_unchanged_when_the_flag_is_off() -> None:
    """A head trained before these columns existed still has to load."""
    import torch
    from scripts.train_polish_tagger import repetition_features

    assert repetition_features([1, 1, 2], torch, "cpu", 0).shape == (3, 0)


def test_the_head_width_matches_the_columns_that_arrive() -> None:
    """The invariant nothing was checking, and it was false.

    `--repetition 3` built the head from `embedding * (1 + lookahead)` with the
    six repetition columns left out of the arithmetic, while the unfrozen path
    assembled its rows without those columns at all -- so training ran, wrote
    `repetition: 3` into the checkpoint, and inference then handed a 2310-wide
    row to a 2304-wide head. Every threshold in the sweep died on the shape.

    The three tests above check the columns themselves. None of them compared
    the width the head is built to against the width the features arrive at,
    which is where the whole thing came apart.
    """
    import torch
    from scripts.train_polish_tagger import assemble, feature_width

    ids = [9, 1, 2, 1, 2, 8]
    hidden = torch.zeros((len(ids), 768))
    embeddings = torch.zeros((len(ids), 768))

    for repetition in (0, 1, 3):
        matrix = assemble(hidden, embeddings, ids, torch, "cpu", 2, repetition)

        assert matrix.shape[1] == feature_width(768, 2, repetition)


def test_the_repetition_columns_actually_reach_the_row() -> None:
    """Not just that the width grew -- that the marks are in it."""
    import torch
    from scripts.train_polish_tagger import assemble

    ids = [9, 1, 2, 1, 2, 8]
    hidden = torch.zeros((len(ids), 4))
    embeddings = torch.zeros((len(ids), 4))

    matrix = assemble(hidden, embeddings, ids, torch, "cpu", 1, 2)

    # Everything the model contributes is zero here, so anything non-zero is
    # the hand-crafted half.
    assert matrix.sum().item() == 4.0
