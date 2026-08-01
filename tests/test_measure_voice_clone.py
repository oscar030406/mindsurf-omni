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
from scripts.measure_voice_clone import SEEN_PACK, load_voices

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
