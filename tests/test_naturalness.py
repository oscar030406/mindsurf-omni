"""The naturalness metric, and the check that earns it the right to judge."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.measure_naturalness import EFFECT_OF_INTEREST, HUB_MODEL, HUB_REPO

ROOT = Path(__file__).resolve().parent.parent


def test_a_sample_without_audio_gets_no_score_rather_than_a_zero(
    monkeypatch: object, tmp_path: Path
) -> None:
    """A zero would read as "unusable speech" for a clip that was never produced.

    Silence and failure are already reported by silent_rate and the failed list;
    folding them into the naturalness mean would blame the voice for a fault in
    the pipeline.
    """
    import scripts.measure_naturalness as module

    monkeypatch.setattr(module, "score_clip", lambda *args, **kwargs: 3.5)  # type: ignore[attr-defined]
    present = tmp_path / "zh000.wav"
    present.write_bytes(b"RIFF")

    rows = [
        {"id": "zh000", "audio_path": str(present)},
        {"id": "zh001", "audio_path": None},
        {"id": "zh002", "audio_path": str(tmp_path / "gone.wav")},
    ]

    annotated = module.annotate(rows, object(), "cpu")

    assert annotated[0]["utmos"] == 3.5
    assert "utmos" not in annotated[1]
    assert "utmos" not in annotated[2]


def test_the_predictor_is_pinned_to_a_tag() -> None:
    """An unpinned hub load changes the instrument between two runs being compared."""
    assert ":" in HUB_REPO, "the hub repo must carry a tag, not a moving branch"
    assert HUB_REPO.endswith(("v1.2.0",))
    assert HUB_MODEL == "utmos22_strong"


def test_the_effect_of_interest_is_smaller_than_the_gap_it_must_judge() -> None:
    """0.26 separates our Talker from upstream's; an instrument that cannot
    resolve that cannot judge the retrain it exists to judge."""
    assert EFFECT_OF_INTEREST <= 0.26


def test_the_recorded_validity_check_separates_every_adjacent_pair() -> None:
    """Its right to judge rests on this, so the evidence ships with the repo.

    Naturalness is the one axis where a plausible-looking number is hardest to
    check by eye -- nobody can read a spectrogram and say "2.05". The guard is
    that the metric reproduced a quality order established by a different
    instrument entirely.
    """
    report = json.loads(
        (ROOT / "artifacts" / "naturalness-validity-2026-07-25.json").read_text(encoding="utf-8")
    )

    assert report["predictor"].startswith(HUB_MODEL)
    comparisons = report["adjacent_comparisons"]
    assert comparisons, "the validity check recorded no comparisons"
    for comparison in comparisons:
        assert comparison["verdict"] == "可区分", comparison
        assert abs(comparison["difference"]) > comparison["threshold"]
    # And the order is the one CER independently produced.
    means = [entry["utmos"] for entry in report["sets"].values()]
    assert means == sorted(means, reverse=True), "sets are not in descending quality order"


def test_the_report_states_that_absolute_values_are_uncalibrated() -> None:
    """Trained on English MOS: 2.05 is not "a Chinese panel scored it 2.05"."""
    report = json.loads(
        (ROOT / "artifacts" / "naturalness-validity-2026-07-25.json").read_text(encoding="utf-8")
    )

    assert "English" in report["caveat"]
    assert "ranking" in report["caveat"]
