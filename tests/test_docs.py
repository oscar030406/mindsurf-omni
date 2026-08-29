"""Everything the shipped documents name must exist, and match the code.

A README citing a script that was renamed, or quoting a default the contract no
longer carries, is worse than no README: it is read under pressure and sends the
reader somewhere wrong. The public tree carries three documents, so these check
those three and the example client they point at.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# What a stranger reads: the front page and the two cards that travel with the
# weights and the listening packs.
DOCS = [
    ROOT / "README.md",
    ROOT / "configs" / "release" / "MODEL_CARD.md",
    ROOT / "configs" / "release" / "LISTENING_DATASET_CARD.md",
]
# Plus the one the backend and the frontend read to wire this up. It quotes
# figures inside tables where the qualifier sits in a neighbouring column, so it
# gets the link and script checks but not the bare-number one. It was three
# files until 2026-08-06 -- capabilities and the runbook were folded in, because
# one audience reading three documents kept them out of sync with each other.
SHIPPED = DOCS + [
    ROOT / "docs" / "INTEGRATION.md",
    ROOT / "docs" / "TRAINING.md",
]


@pytest.mark.parametrize("doc", SHIPPED, ids=lambda path: path.name)
def test_every_script_a_document_cites_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")

    missing = [
        script
        for script in re.findall(r"(scripts/[a-z_]+\.py)", text)
        if not (ROOT / script).is_file()
    ]

    assert not missing, f"{doc.name} cites scripts that do not exist: {missing}"


@pytest.mark.parametrize("doc", SHIPPED, ids=lambda path: path.name)
def test_every_repository_path_a_document_links_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    base = doc.parent

    missing = []
    for target in re.findall(
        r"\]\((\.\./[A-Za-z][\w./-]*|[a-z][a-z_/]*(?:\.(?:md|py|json|toml|yml))?)\)", text
    ):
        if not (base / target).resolve().exists():
            missing.append(target)

    assert not missing, f"{doc.name} links to missing files: {missing}"


def test_no_document_quotes_a_number_without_saying_what_it_is_worth() -> None:
    """A bare CER reads as a verdict; ours is reported-only and must say so."""
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for line in text.splitlines():
            if re.search(r"CER\s*[:：为=]?\s*\d+(\.\d+)?\s*%?", line):
                assert any(
                    marker in line for marker in ("±", "极限", "阈值", "分辨", "假设", "例", "到")
                ), f"{doc.name} quotes a CER without qualifying it: {line.strip()}"


def test_the_example_client_only_uses_endpoints_that_exist() -> None:
    """A copyable example that calls a missing route is worse than no example."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")
    app = (ROOT / "src" / "mindsurf_omni" / "service" / "app.py").read_text(encoding="utf-8")

    called = set(re.findall(r"/v1/[a-z/-]+", example))
    implemented = set(re.findall(r'@app\.(?:get|post|websocket)\("(/v1/[^"]+)"', app))

    assert called <= implemented, f"the example calls: {sorted(called - implemented)}"


def test_the_example_does_not_hardcode_the_sample_rate() -> None:
    """Assuming it is how a client ends up playing 24 kHz audio at 16 kHz."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")

    assert "x-sample-rate" in example
    assert "input_sample_rate" in example  # read from session.created too


def test_the_example_checks_the_licence_before_using_the_output() -> None:
    """It is the first thing a backend author should see."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")

    assert "commercial_use_permitted" in example
