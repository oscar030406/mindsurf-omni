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
