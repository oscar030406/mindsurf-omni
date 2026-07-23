"""Documentation that names things must name things that exist.

A runbook citing a script that was renamed, or an error message that was
reworded, is worse than no runbook: it is consulted under pressure and sends
the reader somewhere wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_every_script_a_document_cites_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")

    missing = [
        script
        for script in re.findall(r"(scripts/[a-z_]+\.py)", text)
        if not (ROOT / script).is_file()
    ]

    assert not missing, f"{doc.name} cites scripts that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_every_repository_path_a_document_links_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    base = doc.parent

    missing = []
    for target in re.findall(r"\]\((\.\./[^)#]+|[a-z][a-z_/]*\.(?:md|py|json|toml|yml))\)", text):
        if not (base / target).resolve().exists():
            missing.append(target)

    assert not missing, f"{doc.name} links to missing files: {missing}"


def test_the_runbook_quotes_error_messages_the_code_actually_emits() -> None:
    """A reworded message makes the runbook unsearchable at the moment it matters."""
    runbook = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    app = (ROOT / "src" / "mindsurf_omni" / "service" / "app.py").read_text(encoding="utf-8")
    config = (ROOT / "src" / "mindsurf_omni" / "service" / "config.py").read_text(encoding="utf-8")

    assert "no speech engine is configured" in runbook
    assert "no speech engine is configured" in app

    assert "mount the weights directory" in runbook
    assert "mount the weights directory" in config


def test_the_runbook_covers_the_failures_that_have_actually_happened() -> None:
    """Each of these cost real time; a runbook that omits them is decorative."""
    runbook = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")

    for symptom in [
        "OOM",  # the full dataset does not fit in memory
        "113.13M",  # silently skipped weights left at random init
        "缓冲",  # the log lags because stdout is block-buffered
        "采样率",  # a wrong rate is a changed pitch, not silence
        "循环论证",  # scoring a model with its own component
    ]:
        assert symptom in runbook, f"the runbook does not cover {symptom!r}"


def test_the_evaluation_guide_states_the_three_rules_it_enforces() -> None:
    """Each is enforced in code; the guide must not omit one and imply choice."""
    guide = (ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")

    assert "循环论证" in guide  # the judge must be independent
    assert "仅报告" in guide  # an instrument that cannot resolve may not judge
    assert "无法区分" in guide  # a difference inside the noise has no direction


def test_the_evaluation_guide_chains_scripts_that_exist_in_that_order() -> None:
    """A guide whose steps do not chain sends the reader in a circle."""
    guide = (ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")

    for step in (
        "generate_speech_samples.py",
        "transcribe_samples.py",
        "evaluate_speech.py",
    ):
        assert step in guide
        assert (ROOT / "scripts" / step).is_file()

    # The order matters: transcription consumes what generation writes.
    assert guide.index("generate_speech_samples.py") < guide.index("transcribe_samples.py")
    assert guide.index("transcribe_samples.py") < guide.index("evaluate_speech.py")


def test_every_decision_says_what_would_overturn_it() -> None:
    """A decision without a reversal condition is a decision nobody can revisit.

    It reads as permanent, so the next person either follows it blindly or
    ignores it entirely -- and neither is what the record is for.
    """
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")

    headings = [line for line in text.splitlines() if line.startswith("## ")]
    reversals = text.count("什么情况下推翻")

    assert len(headings) >= 5
    assert reversals == len(headings), (
        f"{len(headings)} decisions but {reversals} reversal conditions"
    )


def test_the_decisions_that_contradict_the_brief_give_their_evidence() -> None:
    """Departing from the brief is fine; departing without a reason is not."""
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")

    # The parameter-count decision rests on arithmetic that must be shown.
    assert "20 tokens/参数" in text or "tokens/参数" in text
    assert "23.0" in text  # our actual ratio
    # The architecture decision rests on a measurement.
    assert "logit 差 0.0" in text


def test_the_handover_states_that_no_quality_number_exists_yet() -> None:
    """The most likely misreading of this repository is that it has results.

    It has a pipeline that would produce them. Someone skimming the test count
    and the CI badge could easily conclude otherwise.
    """
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")

    assert "没有任何质量数字" in text
    assert "CER" in text


def test_the_handover_names_the_trap_that_silently_ruins_a_run() -> None:
    """Calling the upstream trainer directly loses the base without saying so."""
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")

    assert "train_omni.py" in text
    assert "113.13M" in text  # the wrong parameter count that signals it
    assert "152.06M" in text  # the right one


def test_the_handover_tells_the_reader_how_to_audit_the_work() -> None:
    """A handover that only says "trust this" gives the reader nothing to check."""
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")

    assert "仅报告" in text  # instruments without gating eligibility
    assert "无法区分" in text  # differences inside the noise
    assert "永远不会失败" in text  # checks that cannot fail
