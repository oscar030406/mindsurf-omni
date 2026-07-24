"""What the service can say about itself when something is wrong.

A health check that answers "ok" whenever the process is alive tells an
operator nothing they could not get from `ps`. The useful question is narrower:
can this instance serve a request right now, and if not, which part is missing?

So readiness reports per-component state. A container that is up but cannot
reach its weights should fail readiness while still answering the endpoint that
explains why -- crash-looping takes that explanation away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["ready", "degraded", "unavailable"]


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


def assess(engine: object | None, configuration_error: str | None = None) -> Health:
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
