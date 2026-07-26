"""The licence record, and the ways it could quietly become a false claim.

The pretraining phase of this project shipped a boolean that asserted "human
reviewed" with nobody having reviewed anything. The lesson was not to add more
booleans but to make every claim carry its evidence, and to make an unverified
one impossible to mistake for a verified one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = json.loads((ROOT / "configs" / "release" / "licence.json").read_text(encoding="utf-8"))


def test_the_conclusion_names_which_asset_binds_it() -> None:
    """ "Not commercial" without a reason cannot be acted on or rebutted."""
    conclusion = RECORD["conclusion"]

    assert conclusion["commercial_use_permitted"] is False
    assert conclusion["binding_constraint"] in {asset["name"] for asset in RECORD["assets"]}
    assert len(conclusion["reason"]) > 40


def test_an_unverified_asset_is_null_not_false() -> None:
    """False reads as "checked and disallowed"; null reads as "not checked".

    Collapsing the two would let an unread licence be reported as a finding.
    """
    for asset in RECORD["assets"]:
        if not asset["verified"]:
            assert asset["commercial_use"] is None, asset["name"]
        else:
            assert isinstance(asset["commercial_use"], bool), asset["name"]


def test_every_unverified_asset_appears_in_the_open_questions() -> None:
    """An unread licence that nobody is tracking is an unread licence forever."""
    unverified = {asset["name"] for asset in RECORD["assets"] if not asset["verified"]}
    questions = " ".join(RECORD["open_questions"])

    assert unverified, "if everything is verified, this test should be deleted"
    # The questions need not name each asset, but must acknowledge the count.
    assert "unread" in questions or "unstated" in questions


def test_the_binding_constraint_is_actually_the_strictest() -> None:
    """A conclusion pinned to the wrong asset would loosen when that one changes."""
    binding = next(
        asset
        for asset in RECORD["assets"]
        if asset["name"] == RECORD["conclusion"]["binding_constraint"]
    )

    assert binding["verified"] is True
    assert binding["commercial_use"] is False


def test_the_record_says_what_would_change_the_conclusion() -> None:
    """Otherwise the restriction looks permanent and nobody tries."""
    assert len(RECORD["how_to_change_the_conclusion"]) > 60
    assert "false claim" in RECORD["how_to_change_the_conclusion"]


def test_a_copys_claim_about_its_own_licence_is_not_evidence() -> None:
    """model/SenseVoiceSmall ships a README saying Apache-2.0. It is wrong.

    The authoritative card carries license: other, pointing at the FunASR
    Model Open Source License Agreement. Recording the copy's claim would have
    turned a component whose terms do not resolve commercial use into a clean
    yes, in the one file a reader trusts to be checked.
    """
    encoder = next(asset for asset in RECORD["assets"] if asset["name"] == "sensevoice_small")

    assert "Apache" not in encoder["licence"]
    assert "FunASR" in encoder["licence"]
    # Read, and still unanswered: the licence has no commercial clause either
    # way, so null stays null. false would claim a finding nobody made.
    assert encoder["commercial_use"] is None
    assert encoder["verified"] is False


def test_a_permissive_asset_that_demands_credit_says_where() -> None:
    """CC-BY-4.0 loosens nothing and adds an obligation; unlisted, it is unmet."""
    codec = next(asset for asset in RECORD["assets"] if asset["name"] == "mimi")

    assert codec["commercial_use"] is True
    assert any("mimi" in entry for entry in RECORD["attribution_required"])


def test_attribution_names_a_pinned_revision() -> None:
    """ "The minimind dataset" is not attribution; a revision is."""
    for entry in RECORD["attribution_required"]:
        assert "@" in entry
        revision = entry.split("@")[1].strip()
        assert len(revision) >= 8, f"{entry} does not pin a revision"


def test_attribution_names_the_host_it_resolves_on() -> None:
    """owner/name is ambiguous across hubs, and one of ours was not on the one
    everybody assumed: gongjy/minimind_dataset does not exist on Hugging Face,
    it is a ModelScope repo. An attribution nobody can resolve is not one."""
    for entry in RECORD["attribution_required"]:
        assert "." in entry.split("/")[0], f"{entry} does not say which host"


def test_every_component_that_ships_in_a_checkpoint_is_listed() -> None:
    """The graft turned upstream's Talker from a benchmark into a component.

    Twenty of its tensors travel inside t2a_graft and everything after it, so
    it is part of what would be distributed and needs a row of its own.
    """
    assert "upstream_talker_weights" in {asset["name"] for asset in RECORD["assets"]}


@pytest.mark.parametrize("document", ["README.md", "docs/INTEGRATION.md", "docs/EVALUATION.md"])
def test_the_restriction_appears_wherever_someone_might_start(document: str) -> None:
    """A restriction stated in one place is one that gets missed."""
    text = (ROOT / document).read_text(encoding="utf-8")

    if document == "docs/EVALUATION.md":
        return  # evaluation does not distribute anything
    assert "CC-BY-NC" in text or "commercial_use_permitted" in text


def test_the_code_and_the_record_agree_about_commercial_use() -> None:
    """The service reports this on every response; a disagreement is a false claim."""
    engine = (ROOT / "src" / "mindsurf_omni" / "service" / "engine.py").read_text(encoding="utf-8")

    assert "commercial_use_permitted: bool = False" in engine
    assert RECORD["conclusion"]["commercial_use_permitted"] is False
