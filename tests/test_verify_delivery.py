"""The delivery check must fail on an incomplete delivery.

A gate that always passes reports readiness it never verified, so each check
is driven against a repository that is wrong in that specific way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_delivery import (
    Findings,
    check_documents_match_code,
    check_licence_is_consistent,
)


def test_findings_exit_nonzero_when_something_is_missing() -> None:
    findings = Findings()
    findings.check("present", True)
    findings.check("absent", False, "not on disk")

    assert findings.report() == 1


def test_findings_exit_zero_when_everything_is_there() -> None:
    findings = Findings()
    findings.check("present", True)

    assert findings.report() == 0


def test_the_real_repository_passes_every_check() -> None:
    """If this fails, the handover is not ready -- fix the repository."""
    findings = Findings()
    check_documents_match_code(findings)
    check_licence_is_consistent(findings)

    assert findings.gaps == []


def test_a_documented_endpoint_that_does_not_exist_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is the failure mode the check exists for: a guide sending a 404."""
    import scripts.verify_delivery as verify

    (tmp_path / "src" / "mindsurf_omni" / "service").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "mindsurf_omni" / "service" / "app.py").write_text(
        '@app.get("/v1/models")\n', encoding="utf-8"
    )
    (tmp_path / "docs" / "INTEGRATION.md").write_text(
        "| `GET /v1/models` | x |\n| `GET /v1/invented` | y |\n", encoding="utf-8"
    )
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    findings = Findings()
    verify.check_documents_match_code(findings)

    assert findings.gaps
    assert "/v1/invented" in findings.gaps[0][1]


def test_a_licence_record_contradicting_the_code_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conclusion is stated in several places; a disagreement is a false claim."""
    import scripts.verify_delivery as verify

    (tmp_path / "configs" / "release").mkdir(parents=True)
    (tmp_path / "src" / "mindsurf_omni" / "service").mkdir(parents=True)
    (tmp_path / "configs" / "release" / "licence.json").write_text(
        json.dumps(
            {
                "conclusion": {"commercial_use_permitted": True},
                "assets": [{"name": "x", "verified": True, "commercial_use": True}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "mindsurf_omni" / "service" / "engine.py").write_text(
        "commercial_use_permitted: bool = False\n", encoding="utf-8"
    )
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    findings = Findings()
    verify.check_licence_is_consistent(findings)

    assert findings.gaps


def test_a_missing_deliverable_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.verify_delivery as verify

    monkeypatch.setattr(verify, "ROOT", tmp_path)  # empty repository
    findings = Findings()
    verify.check_deliverables(findings)

    assert len(findings.gaps) >= 10
    assert findings.ok == []
