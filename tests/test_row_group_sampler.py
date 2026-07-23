"""What the row-group sampler promises, and what it deliberately does not."""

from __future__ import annotations

from mindsurf_omni.data.row_group_sampler import RowGroupShuffleSampler

# Five groups of ten rows.
OFFSETS = [0, 10, 20, 30, 40]
TOTAL = 50


def test_every_row_appears_exactly_once() -> None:
    """Dropping or repeating rows would silently change what the model sees."""
    indices = list(RowGroupShuffleSampler(OFFSETS, TOTAL, seed=1))

    assert sorted(indices) == list(range(TOTAL))


def test_order_changes_between_epochs() -> None:
    sampler = RowGroupShuffleSampler(OFFSETS, TOTAL, seed=1)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert first != second
    assert sorted(second) == list(range(TOTAL))


def test_the_same_seed_and_epoch_reproduce_the_order() -> None:
    """A run that cannot be reproduced cannot be debugged."""
    a = RowGroupShuffleSampler(OFFSETS, TOTAL, seed=7)
    b = RowGroupShuffleSampler(OFFSETS, TOTAL, seed=7)
    a.set_epoch(3)
    b.set_epoch(3)

    assert list(a) == list(b)


def test_rows_stay_within_their_window_which_is_the_point_and_the_cost() -> None:
    """Locality is what makes reads cheap, and it is also the compromise.

    With one group per window, a row never appears outside its own group's
    stretch of the sequence. That is exactly why each group is read once per
    epoch -- and exactly why this is not a global shuffle.
    """
    indices = list(RowGroupShuffleSampler(OFFSETS, TOTAL, seed=1, buffer_groups=1))

    for start in range(0, TOTAL, 10):
        window = indices[start : start + 10]
        assert len({index // 10 for index in window}) == 1


def test_a_larger_buffer_mixes_across_groups() -> None:
    indices = list(RowGroupShuffleSampler(OFFSETS, TOTAL, seed=1, buffer_groups=2))
    first_window = indices[:20]

    assert len({index // 10 for index in first_window}) == 2


def test_a_ragged_final_group_is_not_truncated() -> None:
    """The last row group is usually short; losing its tail loses data."""
    offsets = [0, 10, 20]
    indices = list(RowGroupShuffleSampler(offsets, total_rows=23, seed=1))

    assert sorted(indices) == list(range(23))


def test_buffer_groups_must_be_positive() -> None:
    try:
        RowGroupShuffleSampler(OFFSETS, TOTAL, buffer_groups=0)
    except ValueError:
        return
    raise AssertionError("a zero-group buffer should be refused, not silently ignored")
