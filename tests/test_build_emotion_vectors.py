"""The pack is one line of arithmetic, and two things about it decide a run.

alpha=0 must return the originals -- that is the baseline arm of every emotion
comparison, and a builder that quietly perturbs it would make the baseline and
the emotional arm differ by something nobody declared. And the reference codes
must come through untouched: they are the model's other conditioning input, and
the cross-arm measurement showed they carry no prosody, so moving them would
cost identity and buy nothing.
"""

from __future__ import annotations

import pytest
from scripts.build_emotion_vectors import shift

pytest.importorskip("torch")


def _voices() -> dict[str, dict[str, object]]:
    import torch

    return {
        "dylan": {"ref_codes": torch.ones(8, 4), "spk_emb": torch.tensor([1.0, 0.0])},
        "moon": {"ref_codes": torch.zeros(8, 4), "spk_emb": torch.tensor([0.0, 1.0])},
    }


def test_alpha_zero_returns_the_originals() -> None:
    import torch

    voices = _voices()
    built = shift(voices, torch.tensor([0.5, 0.5]), 0.0)

    for name, entry in built.items():
        assert torch.equal(entry["spk_emb"], voices[name]["spk_emb"])


def test_the_reference_codes_are_carried_through_untouched() -> None:
    import torch

    voices = _voices()
    built = shift(voices, torch.tensor([0.5, 0.5]), 1.0)

    for name, entry in built.items():
        assert torch.equal(entry["ref_codes"], voices[name]["ref_codes"])
    assert torch.allclose(built["dylan"]["spk_emb"], torch.tensor([1.5, 0.5]))


def test_a_voice_that_is_not_in_the_pack_stops_the_build() -> None:
    """Silently building eleven voices for a twelve-voice protocol produces a
    result table with a missing row that reads like a failed generation."""
    import torch

    with pytest.raises(SystemExit, match="没有"):
        shift(_voices(), torch.tensor([0.5, 0.5]), 0.5, only=["dylan", "nobody"])
