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


def _spans(text: str) -> list[tuple[int, int]]:
    """One span per character, for tests that are not about tokenisation."""
    return [(index, index + 1) for index in range(len(text))]


def test_repetition_columns_mark_both_copies() -> None:
    """The head reads one position at a time, so 时间时间 looks like two
    ordinary words. It clears 0.437 of the injected repetition against the
    generative arm's 0.603, and this is the column that difference lives in."""
    import torch
    from scripts.train_polish_tagger import repetition_features

    text = "他时间时间了"
    out = repetition_features(text, _spans(text), torch, "cpu", 2)

    opens, closes = out[:, 2], out[:, 3]  # length 2 columns
    assert opens.tolist() == [0, 1, 1, 0, 0, 0]
    assert closes.tolist() == [0, 0, 0, 1, 1, 0]


def test_a_repetition_the_tokenizer_cuts_through_is_still_seen() -> None:
    """The bug this file was rewritten for.

    地点还是还是在B203 tokenises to 地点 / 还是 / 还 / 是在: the second 还是 is
    split across a boundary with 在, so comparing token ids left the columns
    dark on a repetition that is plain in the text. Measured on four dictated
    notes -- 应该应该 and 这边这边 tokenise cleanly and were removed, this one
    did not and survived every arm.
    """
    import torch
    from scripts.train_polish_tagger import repetition_features

    text = "地点还是还是在"
    spans = [(0, 2), (2, 4), (4, 5), (5, 7)]  # 地点 | 还是 | 还 | 是在

    out = repetition_features(text, spans, torch, "cpu", 2)

    # Every token overlapping either copy is marked, including the one the
    # tokenizer glued to the following character.
    assert out[0].sum().item() == 0.0  # 地点, outside
    assert out[1].sum().item() > 0.0  # 还是, the first copy
    assert out[2].sum().item() > 0.0  # 还, half the second copy
    assert out[3].sum().item() > 0.0  # 是在, the other half plus 在


def test_a_sequence_with_no_repetition_is_all_zero() -> None:
    import torch
    from scripts.train_polish_tagger import repetition_features

    text = "客户那边催了"
    assert repetition_features(text, _spans(text), torch, "cpu", 2).sum().item() == 0


def test_the_old_width_is_unchanged_when_the_flag_is_off() -> None:
    """A head trained before these columns existed still has to load."""
    import torch
    from scripts.train_polish_tagger import repetition_features

    assert repetition_features("啊啊他", _spans("啊啊他"), torch, "cpu", 0).shape == (3, 0)


def test_the_head_width_matches_the_columns_that_arrive() -> None:
    """The invariant nothing was checking, and it was false.

    `--repetition 3` built the head from `embedding * (1 + lookahead)` with the
    six repetition columns left out of the arithmetic, while the unfrozen path
    assembled its rows without those columns at all -- so training ran, wrote
    `repetition: 3` into the checkpoint, and inference then handed a 2310-wide
    row to a 2304-wide head. Every threshold in the sweep died on the shape.
    """
    import torch
    from scripts.train_polish_tagger import assemble, feature_width

    text = "他时间时间了"
    spans = _spans(text)
    hidden = torch.zeros((len(spans), 768))
    embeddings = torch.zeros((len(spans), 768))

    for repetition in (0, 1, 3):
        matrix = assemble(hidden, embeddings, text, spans, torch, "cpu", 2, repetition)

        assert matrix.shape[1] == feature_width(768, 2, repetition)


def test_the_repetition_columns_actually_reach_the_row() -> None:
    """Not just that the width grew -- that the marks are in it."""
    import torch
    from scripts.train_polish_tagger import assemble

    text = "他时间时间了"
    spans = _spans(text)
    hidden = torch.zeros((len(spans), 4))
    embeddings = torch.zeros((len(spans), 4))

    matrix = assemble(hidden, embeddings, text, spans, torch, "cpu", 1, 2)

    # Everything the model contributes is zero here, so anything non-zero is
    # the hand-crafted half: four characters marked, one column each.
    assert matrix.sum().item() == 4.0


def test_the_unit_is_recorded_so_an_old_head_cannot_be_loaded_silently() -> None:
    """Token-id columns and character columns have the same width and different
    meanings; without this field nothing downstream could tell them apart."""
    from scripts.train_polish_tagger import REPETITION_UNIT

    assert REPETITION_UNIT == "character"
