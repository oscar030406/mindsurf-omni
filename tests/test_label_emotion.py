"""The de-interleave is the one place this script can be silently wrong.

Reading the corpus's flat token list as eight contiguous blocks instead of
eight interleaved streams decodes to noise -- and noise still gets a label, so
the pilot would report a distribution measured on garbage and nothing would
error. The training loader's convention is the authority, so this pins it.
"""

from __future__ import annotations

import pytest
from scripts.label_emotion import deinterleave


def test_frames_are_interleaved_not_blocked() -> None:
    """omni_dataset.py reads tokens[i + j] for j in 0..7 stepping by 8."""
    tokens = list(range(24))  # three frames of eight codebooks

    rows = deinterleave(tokens)

    assert len(rows) == 8
    assert rows[0] == [0, 8, 16]
    assert rows[7] == [7, 15, 23]


def test_a_trailing_partial_frame_is_dropped() -> None:
    """A ragged tail would make the rows unequal, and the codec needs a rectangle."""
    rows = deinterleave(list(range(19)))

    assert {len(row) for row in rows} == {2}


@pytest.mark.parametrize("length", [0, 7])
def test_too_short_to_hold_a_frame_yields_empty_rows(length: int) -> None:
    rows = deinterleave(list(range(length)))

    assert rows == [[] for _ in range(8)]


def test_truncation_bounds_the_decode_at_a_whole_frame() -> None:
    """A ragged tail would leave the codebook rows unequal and the codec needs a
    rectangle; the cap is applied before de-interleaving so it must divide."""
    from scripts.label_emotion import MAX_TOKENS

    assert MAX_TOKENS % 8 == 0
    rows = deinterleave(list(range(MAX_TOKENS)))
    assert {len(row) for row in rows} == {MAX_TOKENS // 8}
