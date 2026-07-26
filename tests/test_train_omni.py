"""The guard against a checkpoint that only half loads.

Upstream's loader drops every tensor whose shape disagrees with the model it
built, logs one line, and trains on. That has already cost this project one
full run, and the graft makes it likelier rather than rarer: our Thinker and
upstream's Talker have different shapes on purpose, so any caller who builds
both halves at one shape silently throws away the half that was the point.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeModel:
    def __init__(self, shapes: dict[str, tuple[int, ...]]) -> None:
        self._shapes = shapes

    def named_parameters(self) -> Any:
        return [(name, FakeTensor(shape)) for name, shape in self._shapes.items()]


def install_fake_torch(state: dict[str, FakeTensor], monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")
    torch.load = lambda *args, **kwargs: state  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)


# Ours is FFN 3584; upstream's Talker is 2432. The graft carries both.
OURS = (3584, 768)
UPSTREAM = (2432, 768)


def test_a_talker_that_would_be_dropped_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact seven-hour failure: the graft evaporates and nothing says so."""
    from scripts.train_omni import refuse_skipped_tensors

    install_fake_torch(
        {"talker.model.layers.0.mlp.gate_proj.weight": FakeTensor(UPSTREAM)}, monkeypatch
    )
    checkpoint = tmp_path / "t2a_graft_768.pth"
    checkpoint.write_bytes(b"")

    with pytest.raises(SystemExit) as raised:
        refuse_skipped_tensors(
            FakeModel({"talker.model.layers.0.mlp.gate_proj.weight": OURS}), checkpoint
        )

    message = str(raised.value)
    assert "talker.model.layers.0.mlp.gate_proj.weight" in message
    # The message has to carry the fix, not just the symptom: whoever reads it
    # is reading a log at the start of a run they meant to leave unattended.
    assert "MINDSURF_TALKER_SHAPE=upstream" in message


def test_a_checkpoint_that_fits_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.train_omni import refuse_skipped_tensors

    install_fake_torch(
        {"talker.model.layers.0.mlp.gate_proj.weight": FakeTensor(UPSTREAM)}, monkeypatch
    )
    checkpoint = tmp_path / "t2a_graft_768.pth"
    checkpoint.write_bytes(b"")

    refuse_skipped_tensors(
        FakeModel({"talker.model.layers.0.mlp.gate_proj.weight": UPSTREAM}), checkpoint
    )


def test_a_missing_talker_is_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T2A starts from a text-only base every time; absent is not wrong."""
    from scripts.train_omni import refuse_skipped_tensors

    install_fake_torch({"model.layers.0.mlp.gate_proj.weight": FakeTensor(OURS)}, monkeypatch)
    checkpoint = tmp_path / "llm_mindsurf_768.pth"
    checkpoint.write_bytes(b"")

    refuse_skipped_tensors(
        FakeModel(
            {
                "model.layers.0.mlp.gate_proj.weight": OURS,
                "talker.model.layers.0.mlp.gate_proj.weight": OURS,
            }
        ),
        checkpoint,
    )


def test_the_a2a_tail_sets_the_talker_shape_in_both_places() -> None:
    """One knob, used twice. Two knobs is two chances to set only one.

    Training with the Talker widened to our shape drops upstream's twenty
    tensors; reading the product at the wrong --shape scores a model that was
    never loaded. Either alone produces a confident wrong number.
    """
    script = (ROOT / "scripts" / "run_a2a_from.sh").read_text(encoding="utf-8")

    assert 'MINDSURF_TALKER_SHAPE="$TALKER_SHAPE"' in script
    assert '--shape "$SHAPE"' in script
    assert "--shape mindsurf" not in script  # the hardcoded one this replaced
