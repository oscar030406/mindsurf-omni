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


def test_the_t2a_probe_names_its_product_after_its_rate() -> None:
    """Two probes differing only in learning rate must not share a filename.

    The archived stage product is the entire reason for running one stage
    instead of a chain: the round before this one lost its T2A checkpoint to
    the A2A stages and could not attribute its own failure.
    """
    script = (ROOT / "scripts" / "run_t2a_probe.sh").read_text(encoding="utf-8")

    assert 'T2A_LR="${T2A_LR:-5e-5}"' in script
    assert '--learning_rate "$T2A_LR"' in script
    assert "t2a_lr5e5" not in script.split("set -u", 1)[1]  # no hardcoded name below the header
    assert '--checkpoint "$ROOT/out/${SAVE}_768.pth"' in script


def test_the_a2a_tail_runs_upstreams_three_passes_at_upstreams_length() -> None:
    """640 and 768 dropped the ends of the longest utterances, silently.

    Out-of-range target codes are neither written nor labelled, so the cap was
    deciding how much of each answer got supervised: 16.9% of A2A samples lost
    codes at 640 and 7.5% at 768, against none at upstream's 1024. Line 6 of
    that pipeline -- a third A2A pass at 5e-6 -- had never been run here.
    """
    script = (ROOT / "scripts" / "run_a2a_from.sh").read_text(encoding="utf-8")

    assert 'A2A_SEQ="${A2A_SEQ:-1024}"' in script
    assert "--max_seq_len 640" not in script
    assert "--max_seq_len 768" not in script

    projector = script.index('run "a2a_proj"')
    full = script.index('run "a2a_full"')
    tail = script.index('run "a2a_tail"')
    assert projector < full < tail
    assert "--learning_rate 5e-4" in script[projector:full]
    assert "--learning_rate 5e-5" in script[full:tail]
    assert "--learning_rate 5e-6" in script[tail:]
