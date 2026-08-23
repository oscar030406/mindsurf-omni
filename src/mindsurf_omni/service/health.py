"""What the service can say about itself when something is wrong.

A health check that answers "ok" whenever the process is alive tells an
operator nothing they could not get from `ps`. The useful question is narrower:
can this instance serve a request right now, and if not, which part is missing?

So readiness reports per-component state. A container that is up but cannot
reach its weights should fail readiness while still answering the endpoint that
explains why -- crash-looping takes that explanation away.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["ready", "degraded", "unavailable"]


# What has happened since this process started, and nothing else. Not a metrics
# system: no registry, no exposition format, no scraper. The question it exists
# to answer is the one an operator asks first -- "is it refusing everything, or
# was it just me?" -- and for that, numbers that reset on restart are enough.
# Reach for Prometheus when the answer needs to survive a restart or span
# replicas.
#
# Incremented from the event loop thread only, so a plain Counter is safe;
# nothing here runs in the recogniser's thread.
COUNTS: Counter[str] = Counter()
_STARTED = time.monotonic()


def count(name: str, amount: float = 1) -> None:
    COUNTS[name] += amount


def counters() -> dict[str, object]:
    """The counters and how long they have been counting.

    The uptime rides along because a count without a window is unreadable: 40
    refusals in an hour is a bad client, 40 in four seconds is an outage.
    """
    return {
        "uptime_seconds": round(time.monotonic() - _STARTED, 1),
        "counts": {name: round(value, 3) for name, value in sorted(COUNTS.items())},
    }


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    name: str
    ready: bool
    detail: str = ""


@dataclass
class Health:
    components: list[ComponentHealth] = field(default_factory=list)

    def add(self, name: str, ready: bool, detail: str = "") -> None:
        self.components.append(ComponentHealth(name=name, ready=ready, detail=detail))

    @property
    def status(self) -> Status:
        if not self.components:
            return "unavailable"
        if all(component.ready for component in self.components):
            return "ready"
        # Degraded rather than unavailable when the fallback path survives: a
        # service that can still answer over the cascade is not down, and
        # reporting it as down would take it out of rotation needlessly.
        if any(component.ready for component in self.components):
            return "degraded"
        return "unavailable"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "components": [
                {"name": c.name, "ready": c.ready, "detail": c.detail} for c in self.components
            ],
            # Named explicitly so a reader is not left inferring it from the
            # component list.
            "not_ready": [c.name for c in self.components if not c.ready],
        }


def _weights_are_loaded(engine: Any) -> bool:
    """Whether the recogniser is holding weights right now."""
    recogniser = getattr(engine, "recogniser", None)
    return recogniser is not None and getattr(recogniser, "_model", None) is not None


def assess(
    engine: object | None,
    configuration_error: str | None = None,
    warm_up_error: str | None = None,
) -> Health:
    """Build a health report from what the process actually holds."""
    health = Health()

    if configuration_error:
        health.add("configuration", False, configuration_error)
        return health

    if engine is None:
        health.add(
            "engine",
            False,
            "no engine configured; set MINDSURF_ENGINE to 'native' or 'cascade'",
        )
        return health

    description = engine.describe()  # type: ignore[attr-defined]
    health.add("engine", True, f"path={description.path}")

    # The one thing here that was actually exercised. `describe()` reports what
    # was assembled, which is not the same as what loaded: the components below
    # are listed because configuration named them, and a checkpoint truncated
    # on disk changes none of that. Startup is where the weights are really
    # pulled in, so its exception is the only evidence this function has that
    # the parts work, and it was being thrown away.
    # Only while the weights are still not loaded. The warm-up error is a fact
    # about one moment at start-up, and health has to answer about this one:
    # ``load()`` does not cache a failure, so a card that was busy for the eight
    # seconds of warm-up is retried on the first request and succeeds. Reported
    # from the stored error alone, that machine says it is broken for ever and
    # never comes back -- and an operator watching ``not_ready`` takes a healthy
    # box out of rotation with nothing in the log to explain it. Worse than the
    # bug it replaced: "ready but 500" at least leaves a traceback.
    if warm_up_error and not _weights_are_loaded(engine):
        health.add("warm-up", False, f"loading the weights failed: {warm_up_error}")

    for component in description.components:
        # A frozen component that failed to load is as fatal as a trainable
        # one; frozen describes what training did, not whether it is needed.
        health.add(component.name, True, "frozen" if component.frozen else "trainable")

    # A stage that raises when called is not ready, whatever its components
    # report. Reporting the whole engine ready because the parts are present
    # is the failure this endpoint exists to prevent: the parts being present
    # is what `ps` already tells you.
    for stage in getattr(engine, "unwired", ()):
        health.add(stage, False, "not wired; this stage refuses with the reason")
    return health
