"""Health reporting, which exists to be more useful than "the process is alive"."""

from __future__ import annotations

from mindsurf_omni.contract import ComponentInfo
from mindsurf_omni.service.engine import EngineDescription
from mindsurf_omni.service.health import Health, assess


class _Engine:
    def __init__(self, path: str = "cascade") -> None:
        self.path = path

    def describe(self) -> EngineDescription:
        return EngineDescription(
            path=self.path,  # type: ignore[arg-type]
            components=[
                ComponentInfo(name="thinker", frozen=False),
                ComponentInfo(name="sensevoice-small", frozen=True),
            ],
        )


def test_an_unconfigured_service_is_unavailable_with_a_reason() -> None:
    """ "Unavailable" alone sends the operator to the source."""
    report = assess(engine=None)

    assert report.status == "unavailable"
    assert "MINDSURF_ENGINE" in report.components[0].detail


def test_a_configuration_error_is_reported_instead_of_the_generic_message() -> None:
    """The specific missing file is what an operator needs at three in the morning."""
    report = assess(engine=None, configuration_error="tokenizer=/app/weights/tokenizer")

    assert report.status == "unavailable"
    assert "tokenizer=" in report.components[0].detail


def test_a_loaded_engine_reports_ready_and_names_its_path() -> None:
    report = assess(_Engine("native"))

    assert report.status == "ready"
    assert any("path=native" in component.detail for component in report.components)


def test_every_component_appears_including_the_frozen_ones() -> None:
    """A frozen component that failed to load is as fatal as a trainable one."""
    report = assess(_Engine())

    names = {component.name for component in report.components}
    assert {"thinker", "sensevoice-small"} <= names


def test_a_partial_failure_is_degraded_not_down() -> None:
    """A service still answering over the fallback should stay in rotation."""
    report = Health()
    report.add("engine", True, "path=cascade")
    report.add("talker", False, "checkpoint missing")

    assert report.status == "degraded"


def test_everything_failing_is_unavailable() -> None:
    report = Health()
    report.add("engine", False)
    report.add("codec", False)

    assert report.status == "unavailable"


def test_the_report_names_what_is_not_ready_rather_than_making_it_be_inferred() -> None:
    report = Health()
    report.add("engine", True)
    report.add("codec", False, "not on disk")

    assert report.to_dict()["not_ready"] == ["codec"]


def test_an_empty_report_is_unavailable_not_ready() -> None:
    """ "Nothing checked" must never read as "everything fine"."""
    assert Health().status == "unavailable"


def test_the_endpoint_answers_200_while_degraded(monkeypatch: object) -> None:
    """A service answering over the fallback must not be taken out of rotation."""
    from fastapi.testclient import TestClient

    from mindsurf_omni.service.app import create_app

    app = create_app(_Engine())
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_the_endpoint_answers_503_when_nothing_can_serve() -> None:
    from fastapi.testclient import TestClient

    from mindsurf_omni.service.app import create_app

    response = TestClient(create_app(None)).get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["not_ready"] == ["engine"]
