"""A pack from outside upstream's speaker directory must not join a certified band.

The seen and unseen numbers this project quotes were measured on upstream's
twelve voices. Harvested references are three corpus speakers, and averaging
them into ``unseen`` would move a published band without anyone deciding to --
the same silent-merge failure as scoring an arm against a reference set it
wrote itself. So they load into their own split, and that is what is checked
here along with the two ways the flag can be typed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.measure_voice_clone import SEEN_PACK, identify, load_voices

# torch is in the train extra, not the test environment; this runs where the
# packs are.
pytest.importorskip("torch")


def _pack(path: Path, names: list[str]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: {"ref_codes": torch.zeros(8, 4), "spk_emb": torch.zeros(192)} for name in names},
        str(path),
    )


def test_an_external_pack_lands_in_its_own_split(tmp_path: Path) -> None:
    _pack(tmp_path / "model" / "speaker" / SEEN_PACK, ["dylan"])
    _pack(tmp_path / "harvest" / SEEN_PACK, ["speaker1_calm", "speaker1_lively"])

    voices = load_voices(tmp_path, (tmp_path / "harvest",))

    assert voices["dylan"]["split"] == "seen"
    assert voices["speaker1_calm"]["split"] == "external"
    assert voices["speaker1_lively"]["split"] == "external"


def test_the_pack_may_be_named_as_a_file_or_as_its_directory(tmp_path: Path) -> None:
    _pack(tmp_path / "model" / "speaker" / SEEN_PACK, ["dylan"])
    _pack(tmp_path / "harvest" / SEEN_PACK, ["speaker1_calm"])

    by_directory = load_voices(tmp_path, (tmp_path / "harvest",))
    by_file = load_voices(tmp_path, (tmp_path / "harvest" / SEEN_PACK,))

    assert sorted(by_directory) == sorted(by_file)


def test_a_missing_external_pack_stops_rather_than_scoring_without_it(tmp_path: Path) -> None:
    """Continuing would score whatever voices happen to be loaded, under a name
    the caller believes came from the pack they asked for."""
    _pack(tmp_path / "model" / "speaker" / SEEN_PACK, ["dylan"])

    with pytest.raises(SystemExit, match="不存在"):
        load_voices(tmp_path, (tmp_path / "nowhere",))


def _voices(
    vectors: dict[str, list[float]], splits: dict[str, str]
) -> dict[str, dict[str, object]]:
    import torch

    return {
        name: {"split": splits[name], "spk_emb": torch.tensor(value), "ref_codes": None}
        for name, value in vectors.items()
    }


def test_identification_names_the_nearest_voice_and_prices_the_margin() -> None:
    import torch

    voices = _voices({"dylan": [1.0, 0.0], "moon": [0.0, 1.0]}, {"dylan": "seen", "moon": "unseen"})

    own = identify(torch.tensor([0.9, 0.1]), "dylan", voices)
    assert own is not None and own["hit"] and own["nearest"] == "dylan"
    assert own["margin"] > 0

    swapped = identify(torch.tensor([0.1, 0.9]), "dylan", voices)
    assert swapped is not None and not swapped["hit"] and swapped["nearest"] == "moon"
    assert swapped["margin"] < 0


def test_an_external_pack_does_not_join_the_line_up() -> None:
    """Scoring a harvested pack must not silently make the task twelve-plus-way.

    The criterion is "closest among upstream's twelve"; adding candidates
    lowers the hit rate without anyone choosing a harder test, and a clip
    conditioned on a voice outside that line-up has no verdict to give.
    """
    import torch

    voices = _voices(
        {"dylan": [1.0, 0.0], "moon": [0.0, 1.0], "harvested": [0.95, 0.05]},
        {"dylan": "seen", "moon": "unseen", "harvested": "external"},
    )

    own = identify(torch.tensor([0.9, 0.1]), "dylan", voices)
    assert own is not None and own["candidates"] == 2 and own["hit"]

    assert identify(torch.tensor([0.9, 0.1]), "harvested", voices) is None
