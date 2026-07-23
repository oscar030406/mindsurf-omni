"""Latency accounting, including the ways a report can mislead."""

from __future__ import annotations

import pytest

from mindsurf_omni.evaluation.latency import (
    STAGES,
    LatencyReport,
    TurnTimings,
    stage,
)


def _turn(**milliseconds: float) -> TurnTimings:
    return TurnTimings(stages=dict(milliseconds))


def test_total_is_the_sum_of_stages() -> None:
    turn = _turn(vad_endpoint=100, encode=150, first_text_token=400)

    assert turn.time_to_first_audio_ms == 650


def test_a_missing_stage_is_reported_not_counted_as_zero() -> None:
    """An omitted stage reads as a fast stage, which is the wrong conclusion."""
    turn = _turn(vad_endpoint=100, encode=150)

    missing = turn.missing_stages()

    assert "first_text_token" in missing
    assert "synthesis" in missing
    assert len(missing) == len(STAGES) - 2


def test_stage_timer_records_real_elapsed_time() -> None:
    timings = TurnTimings()

    with stage(timings, "encode"):
        sum(range(200_000))

    assert timings.stages["encode"] > 0


def test_an_unknown_stage_is_refused_rather_than_silently_added() -> None:
    """A typo would create a stage the report never aggregates."""
    timings = TurnTimings()

    with pytest.raises(ValueError, match="unknown stage"), stage(timings, "encdoe"):
        pass


def test_the_timer_records_even_when_the_stage_raises() -> None:
    """A stage that failed still consumed time, and hiding it distorts the total."""
    timings = TurnTimings()

    with pytest.raises(RuntimeError), stage(timings, "synthesis"):
        raise RuntimeError("synthesiser unavailable")

    assert "synthesis" in timings.stages


def test_percentile_reports_a_latency_some_turn_actually_had() -> None:
    """Interpolation would invent a number no user experienced."""
    report = LatencyReport()
    for value in [100, 200, 300, 400, 5000]:
        report.add(_turn(encode=value))

    assert report.percentile(0.95) in {100.0, 200.0, 300.0, 400.0, 5000.0}
    assert report.percentile(0.95) == 5000.0
    assert report.percentile(0.5) == 300.0


def test_percentile_of_nothing_is_not_zero() -> None:
    assert LatencyReport().percentile(0.95) != LatencyReport().percentile(0.95)  # NaN


def test_the_dominant_stage_is_named_so_effort_goes_to_the_right_place() -> None:
    """Usually not the model: waiting for a whole reply before speaking is."""
    report = LatencyReport()
    for _ in range(5):
        report.add(
            _turn(
                vad_endpoint=80,
                encode=120,
                first_text_token=300,
                first_clause=1400,
                synthesis=250,
                transport=50,
            )
        )

    name, value = report.dominant_stage()  # type: ignore[misc]

    assert name == "first_clause"
    assert value == 1400


def test_budget_verdict_states_the_margin_in_both_directions() -> None:
    report = LatencyReport()
    for _ in range(20):
        report.add(_turn(encode=1000, synthesis=500))

    assert "within budget" in report.budget_verdict(3000)
    assert "over budget" in report.budget_verdict(1000)


def test_latency_must_earn_gating_eligibility_like_any_other_metric() -> None:
    """Three timed turns cannot resolve a 200 ms difference."""
    report = LatencyReport()
    for value in [1000, 1100, 1050]:
        report.add(_turn(encode=value))

    assert not report.measurement().gating_eligible


def test_enough_consistent_turns_may_judge() -> None:
    import random

    rng = random.Random(0)
    report = LatencyReport()
    for _ in range(200):
        report.add(_turn(encode=rng.gauss(1000, 40)))

    assert report.measurement(effect_of_interest_ms=200).gating_eligible
