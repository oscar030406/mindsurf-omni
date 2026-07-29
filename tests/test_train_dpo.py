"""The parts of a from-scratch DPO trainer that fail without saying anything.

A flipped sign still descends -- toward the wrong optimum. A reference model
that is not detached still trains. A prompt scored along with the reply still
produces a number. None of these raise, and the loss curve looks the same in
every case, which is why they are pinned here rather than watched for.
"""

from __future__ import annotations

import math

import pytest
from scripts.train_dpo import BETA, dpo_loss, encode_pair

pytest.importorskip("torch")


def _scalar(value: float):
    import torch

    return torch.tensor(value)


def test_the_loss_falls_when_the_policy_prefers_the_chosen_branch() -> None:
    """The direction the whole objective rests on.

    A sign error anywhere in the margin trains the model to prefer the rejected
    branch, and the loss still goes down while it does so.
    """
    reference = (_scalar(-10.0), _scalar(-10.0))

    better, better_margin = dpo_loss(_scalar(-8.0), _scalar(-12.0), *reference)
    worse, worse_margin = dpo_loss(_scalar(-12.0), _scalar(-8.0), *reference)

    assert float(better) < float(worse)
    assert float(better_margin) > 0 > float(worse_margin)


def test_matching_the_reference_exactly_costs_log_two() -> None:
    """With no margin the objective is -log sigmoid(0), which pins beta's place.

    If beta multiplied the wrong term or sat outside the sigmoid, this constant
    would move.
    """
    reference = (_scalar(-5.0), _scalar(-9.0))
    same, margin = dpo_loss(_scalar(-5.0), _scalar(-9.0), *reference)

    assert float(same) == pytest.approx(math.log(2.0), abs=1e-5)
    # Zero margin at step zero is not a coincidence: policy and reference are
    # the same weights, so a first screen showing anything else means they are
    # not, which is a cheap self-check the training loop leans on.
    assert float(margin) == pytest.approx(0.0, abs=1e-6)


def test_the_reference_cancels_rather_than_being_ignored() -> None:
    """Two pairs with the same policy gap but different reference gaps differ.

    A trainer that dropped the reference terms would score these identically,
    and would be doing plain likelihood training under a DPO name.
    """
    policy = (_scalar(-6.0), _scalar(-8.0))

    against_neutral, _ = dpo_loss(*policy, _scalar(-7.0), _scalar(-7.0))
    against_agreed, _ = dpo_loss(*policy, _scalar(-6.0), _scalar(-8.0))

    assert float(against_neutral) < float(against_agreed)


def test_beta_scales_the_margin_and_nothing_else() -> None:
    import torch

    margin = 4.0
    expected = -float(torch.nn.functional.logsigmoid(torch.tensor(BETA * margin)))

    got, _ = dpo_loss(_scalar(-6.0), _scalar(-10.0), _scalar(-7.0), _scalar(-7.0))

    assert float(got) == pytest.approx(expected, abs=1e-6)


def test_the_margin_cancels_the_length_bias_that_raw_sums_carry() -> None:
    """Why the ordering counter uses the margin and not the log-prob difference.

    Summed log-probability falls with length, so a longer chosen reply looks
    worse under a raw comparison however good it is. The reference carries the
    same bias, so subtracting it cancels -- and a counter built on the raw
    difference would have been reporting which branch was shorter.
    """
    # A long chosen reply and a short rejected one, both exactly as likely as
    # the reference finds them: no preference either way, margin zero.
    _, margin = dpo_loss(_scalar(-40.0), _scalar(-4.0), _scalar(-40.0), _scalar(-4.0))

    assert float(margin) == pytest.approx(0.0, abs=1e-6)
    # The raw difference, which the first version counted, is hugely negative.
    assert float(_scalar(-40.0) - _scalar(-4.0)) < -30


class _Tokeniser:
    """Character-level, and honest about the boundary the real one can blur."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<u>" + messages[0]["content"] + "<a>"

    def __call__(self, text: str):
        return type("_Ids", (), {"input_ids": [ord(c) for c in text]})()


def test_a_pair_encodes_to_two_spans_that_exclude_the_prompt() -> None:
    """Scoring the prompt adds the same constant to both branches and dilutes
    the only quantity the loss is built from."""
    encoded = encode_pair(_Tokeniser(), "问题", "好的回答", "差回答")

    assert encoded is not None
    chosen_ids, chosen_span, rejected_ids, rejected_span = encoded
    prefix_length = len("<u>问题<a>")
    assert chosen_span == slice(prefix_length, prefix_length + len("好的回答"))
    assert rejected_span == slice(prefix_length, prefix_length + len("差回答"))
    assert chosen_ids[chosen_span] == [ord(c) for c in "好的回答"]


def test_an_empty_branch_drops_the_pair_rather_than_scoring_an_empty_span() -> None:
    """A zero-length span sums to zero log-probability, which reads as a
    perfectly likely reply rather than as a missing one."""
    assert encode_pair(_Tokeniser(), "问题", "回答", "") is None
    assert encode_pair(_Tokeniser(), "问题", "", "回答") is None


def test_the_floor_tracks_the_checkpoint_scale_rather_than_being_a_constant() -> None:
    """An absolute threshold would be wrong for any other model.

    fp16's spacing is relative, so the same displacement is decisive on small
    weights and invisible on large ones. This project's Thinker sits near
    per-parameter RMS 0.13, where the spacing is about 1.2e-4 -- which is the
    number the first default's whole update budget fell under.
    """
    from scripts.train_dpo import storage_resolution

    ours = storage_resolution(0.1287)
    smaller = storage_resolution(0.001)

    assert ours == pytest.approx(1.22e-4, rel=0.05)
    assert smaller < ours
    # The old default's entire budget across the run.
    assert ours > 2.5e-5
