"""Thinning the training pairs for the data-volume curve."""

from __future__ import annotations

from scripts.subsample_polish_pairs import keeps


def test_the_subsets_nest() -> None:
    """A non-monotonic curve over disjoint subsets would look like a size effect."""
    identifiers = [f"row{index:04d}" for index in range(400)]
    quarter = {name for name in identifiers if keeps(name, 0.25, 1)}
    half = {name for name in identifiers if keeps(name, 0.5, 1)}

    assert quarter <= half


def test_the_fraction_is_roughly_what_was_asked_for() -> None:
    identifiers = [f"row{index:04d}" for index in range(2000)]

    kept = sum(1 for name in identifiers if keeps(name, 0.5, 1))

    assert 0.45 < kept / len(identifiers) < 0.55


def test_everything_survives_at_one() -> None:
    assert all(keeps(f"row{index}", 1.0, 1) for index in range(200))
