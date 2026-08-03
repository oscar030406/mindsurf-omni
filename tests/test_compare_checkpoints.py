"""The check has to fail when the frozen half moved, not only when nothing did."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.compare_checkpoints import compare  # noqa: E402


def checkpoint(thinker: float, talker: float) -> dict[str, object]:
    return {
        "model.layers.0.weight": torch.tensor([thinker, thinker]),
        "lm_head.weight": torch.tensor([thinker]),
        "talker.layers.0.weight": torch.tensor([talker]),
        "audio_proj.weight": torch.tensor([talker]),
    }


def test_frozen_half_identical_and_trained_half_moved() -> None:
    report = compare(checkpoint(1.0, 1.0), checkpoint(1.0, 2.0))
    assert report["groups"]["frozen"] == {
        "total": 2,
        "identical": 2,
        "max_abs_diff": 0.0,
        "moved": [],
    }
    assert report["groups"]["trained"]["identical"] == 0


def test_one_ulp_on_a_frozen_tensor_is_a_failure() -> None:
    """A tolerance would turn "no gradient reached this" into "not much did"."""
    after = checkpoint(1.0, 1.0)
    after["lm_head.weight"] = torch.tensor([1.0 + 2**-23])
    frozen = compare(checkpoint(1.0, 1.0), after)["groups"]["frozen"]
    assert frozen["identical"] == 1
    assert frozen["moved"] == ["lm_head.weight"]


def test_a_tensor_the_loader_dropped_is_reported_not_skipped() -> None:
    after = checkpoint(1.0, 2.0)
    del after["talker.layers.0.weight"]
    report = compare(checkpoint(1.0, 1.0), after)
    assert report["only_in_before"] == ["talker.layers.0.weight"]
    assert report["groups"]["trained"]["total"] == 1


def test_shape_change_counts_as_moved_rather_than_crashing() -> None:
    after = checkpoint(1.0, 1.0)
    after["model.layers.0.weight"] = torch.tensor([1.0, 1.0, 1.0])
    frozen = compare(checkpoint(1.0, 1.0), after)["groups"]["frozen"]
    assert frozen["identical"] == 1
    assert "shape" in frozen["moved"][0]
