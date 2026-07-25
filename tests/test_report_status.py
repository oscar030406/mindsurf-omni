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


def test_a_run_the_model_sat_out_is_marked_as_the_instrument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0.0325 read as a model score is the best-looking wrong number here."""
    import scripts.report_status as report

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "floor-report.json").write_text(
        json.dumps(
            {
                "instrument_only": True,
                "candidate": {
                    "measurements": {
                        "cer": {
                            "value": 0.0325,
                            "noise_floor": 0.0068,
                            "sample_size": 160,
                            "gating_eligible": True,
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

    assert any("模型未参与" in line for line in lines)


def test_an_older_probe_run_is_marked_from_its_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifacts predate the flag; text_source says the same thing."""
    import scripts.report_status as report

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "floor-report.json").write_text(
        json.dumps(
            {
                "generated_by": {"text_source": "probe"},
                "candidate": {
                    "measurements": {
                        "cer": {
                            "value": 0.0414,
                            "noise_floor": 0.0127,
                            "sample_size": 160,
                            "gating_eligible": True,
                        }
                    }
                },
                "text_regression": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "ROOT", tmp_path)

    assert any("模型未参与" in line for line in report.measured_results())


def test_the_licence_summary_counts_unread_terms_rather_than_saying_mostly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Mostly verified" is exactly the rounding this report prevents."""
    lines = licence()

    assert any("未读" in line for line in lines)
    assert any("/6" in line for line in lines)


def _run_log(path: Path, total: int, epochs: int) -> Path:
    lines = []
    for epoch in range(1, epochs + 1):
        for step in range(50, total + 1, 50):
            audio = 5.0 - epoch * 0.3 + (step % 250) * 0.002
            lines.append(
                f"Epoch:[{epoch}/{epochs}]({step}/{total}), loss: {audio + 1.6:.4f}, "
                f"text: 1.6000, audio: {audio:.4f}, lr: 0.0004, epoch_time: 1.0min"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _windows(summary: list[str]) -> list[tuple[int, int]]:
    found = []
    for line in summary:
        if line.strip().startswith("audio ") and ":" in line:
            span = line.strip().split("audio ", 1)[1].split(":", 1)[0]
            low, high = span.split("-")
            found.append((int(low), int(high)))
    return found


def test_training_summary_reports_more_than_one_window(tmp_path: Path) -> None:
    """A single window has already given the wrong answer about convergence here."""
    summary = training(_run_log(tmp_path / "t2a.log", total=31224, epochs=6))

    assert len(_windows(summary)) == 3, summary


def test_the_windows_follow_the_length_of_the_run(tmp_path: Path) -> None:
    """Fixed steps were chosen for T2A's 31224 and hung off the end of A2A's 25877.

    A window reaching past the last step is never full, so it is always
    dropped, and the report quietly loses a third of the evidence it exists to
    show.
    """
    total = 25877
    summary = training(_run_log(tmp_path / "a2a.log", total=total, epochs=3))

    windows = _windows(summary)
    assert len(windows) == 3, summary
    assert all(high <= total for _, high in windows)


def test_a_missing_log_is_said_rather_than_skipped(tmp_path: Path) -> None:
    assert any("未提供" in line for line in training(None))
    assert any("未提供" in line or "没有" in line for line in training(tmp_path / "absent.log"))


def test_a_missing_licence_record_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status command that dies on a missing file surfaces nothing.

    Found by running it from a directory where the record was absent: it
    raised FileNotFoundError instead of saying the file was not there.
    """
    import scripts.report_status as report

    monkeypatch.setattr(report, "ROOT", tmp_path)

    lines = report.licence()

    assert any("不在" in line for line in lines)


def test_a_corrupt_licence_record_is_treated_as_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-written record must not be read as permission."""
    import scripts.report_status as report

    (tmp_path / "configs" / "release").mkdir(parents=True)
    (tmp_path / "configs" / "release" / "licence.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(report, "ROOT", tmp_path)

    lines = report.licence()

    assert any("无法解析" in line for line in lines)
