"""The status report exists to resist rounding in a favourable direction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.report_status import licence, training


def test_with_no_report_it_says_there_are_no_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silence here would be read as "results exist and are fine"."""
    import scripts.report_status as report

    monkeypatch.setattr(report, "ROOT", tmp_path)

    lines = report.measured_results()

    assert any("无" in line for line in lines)
    assert any("没有依据" in line for line in lines)


def test_an_ineligible_measurement_is_marked_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A summary that drops the qualifier turns a caveat into a claim."""
    import scripts.report_status as report

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "report.json").write_text(
        json.dumps(
            {
                "candidate": {
                    "measurements": {
                        "mcq": {
                            "value": 0.42,
                            "noise_floor": 0.07,
                            "sample_size": 192,
                            "gating_eligible": False,
                        }
                    }
                },
                "text_regression": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "ROOT", tmp_path)

    lines = report.measured_results()

    assert any("仅报告" in line for line in lines)
    assert any("文本能力未测" in line for line in lines)


def test_the_licence_summary_counts_unread_terms_rather_than_saying_mostly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Mostly verified" is exactly the rounding this report prevents."""
    lines = licence()

    assert any("未读" in line for line in lines)
    assert any("/6" in line for line in lines)


def test_training_summary_reports_more_than_one_window(tmp_path: Path) -> None:
    """A single window has already given the wrong answer about convergence here."""
    log = tmp_path / "train.log"
    lines = []
    for epoch in (1, 2):
        for window_start in (5000, 15000):
            for index in range(61):
                step = window_start + index * 50
                audio = 5.0 - epoch * 0.3 + (index % 5) * 0.01
                lines.append(
                    f"Epoch:[{epoch}/6]({step}/31224), loss: {audio + 1.6:.4f}, "
                    f"text: 1.6000, audio: {audio:.4f}, lr: 0.0004, epoch_time: 1.0min"
                )
    log.write_text("\n".join(lines), encoding="utf-8")

    summary = training(log)

    assert any("5000-8000" in line for line in summary)
    assert any("15000-18000" in line for line in summary)


def test_a_missing_log_is_said_rather_than_skipped(tmp_path: Path) -> None:
    assert any("未提供" in line for line in training(None))
    assert any("未提供" in line or "没有" in line for line in training(tmp_path / "absent.log"))
