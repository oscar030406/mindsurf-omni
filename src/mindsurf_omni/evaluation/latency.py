"""Where the time goes between the user stopping and the reply starting.

Time-to-first-audio is a system property, not a model property. Shaving the
model alone rarely moves it, because the largest term is usually *when*
synthesis is allowed to begin: waiting for a whole reply spends the entire
generation before the listener hears anything.

So this measures per stage rather than end to end. A single total tells you
that you missed the budget; a breakdown tells you which stage to go fix.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from mindsurf_omni.evaluation.metrics import Measurement, assess

# The stages, in the order a turn passes through them. Named here so a report
# cannot silently omit one -- a missing stage reads as a fast stage.
STAGES = (
    "vad_endpoint",  # deciding the user has stopped
    "encode",  # audio to tokens or features
    "first_text_token",  # the model's first output
    "first_clause",  # enough text to be worth speaking
    "synthesis",  # that clause becoming sound
    "transport",  # reaching the client
)


@dataclass(slots=True)
class TurnTimings:
    """One turn's stage timings, in milliseconds."""

    stages: dict[str, float] = field(default_factory=dict)

    @property
    def time_to_first_audio_ms(self) -> float:
        return sum(self.stages.values())

    def missing_stages(self) -> list[str]:
        return [stage for stage in STAGES if stage not in self.stages]


@contextmanager
def stage(timings: TurnTimings, name: str) -> Iterator[None]:
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}; add it to STAGES rather than passing it through")
    started = time.perf_counter()
    try:
        yield
    finally:
        timings.stages[name] = (time.perf_counter() - started) * 1000


@dataclass(slots=True)
class LatencyReport:
    turns: list[TurnTimings] = field(default_factory=list)

    def add(self, timings: TurnTimings) -> None:
        self.turns.append(timings)

    def percentile(self, fraction: float) -> float:
        """Nearest-rank percentile of time-to-first-audio.

        Nearest-rank rather than interpolated: an interpolated P95 reports a
        latency no turn actually experienced, which is the wrong thing to hold
        a budget against.
        """
        if not self.turns:
            return float("nan")
        values = sorted(turn.time_to_first_audio_ms for turn in self.turns)
        index = min(int(fraction * len(values)), len(values) - 1)
        return values[index]

    def stage_medians(self) -> dict[str, float]:
        """Median per stage, so the largest term is visible at a glance."""
        medians = {}
        for name in STAGES:
            values = sorted(turn.stages[name] for turn in self.turns if name in turn.stages)
            if values:
                medians[name] = values[len(values) // 2]
        return medians

    def budget_verdict(self, budget_ms: float = 3000.0) -> str:
        """Whether these turns met the budget, and whether that generalises.

        Two different claims, and the second is the one a release gate needs.
        "Over budget" describes the turns that were measured, which is always
        safe to say; that the next forty would also miss is a judgement, and
        this project does not let an instrument make one before its noise floor
        has been shown smaller than the effect it is judging. Printing the
        verdict alone puts a pass or fail at the top of the report with the
        qualification somewhere below it, which is how the qualification stops
        being read.
        """
        p95 = self.percentile(0.95)
        if p95 != p95:  # NaN
            return "no turns recorded"
        margin = budget_ms - p95
        verdict = "within budget" if margin >= 0 else "over budget"
        line = f"P95 {p95:.0f} ms against {budget_ms:.0f} ms: {verdict} by {abs(margin):.0f} ms"
        if not self.measurement().gating_eligible:
            line += "（仅描述这批轮次，不能据此判定通过或失败）"
        return line

    def measurement(self, effect_of_interest_ms: float = 200.0) -> Measurement:
        """Latency as a metric that must earn the right to judge, like any other."""
        return assess(
            "time_to_first_audio_ms",
            [turn.time_to_first_audio_ms for turn in self.turns],
            effect_of_interest=effect_of_interest_ms,
            minimum_samples=30,
        )

    def dominant_stage(self) -> tuple[str, float] | None:
        """The stage worth optimising, which is rarely the one people assume."""
        medians = self.stage_medians()
        if not medians:
            return None
        name = max(medians, key=lambda key: medians[key])
        return name, medians[name]
