"""The smoke test must fail on a broken service, not just pass on a good one.

A check that cannot fail is worse than no check: it reports health it never
verified. So these drive the smoke checks against services that are wrong in
specific ways and confirm each one is caught.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripts.smoke_service import Results, check_models, check_speech


def _client(app: FastAPI) -> Any:
    return TestClient(app)


def test_results_reports_failures_with_a_nonzero_exit() -> None:
    results = Results()
    results.check("fine", True)
    results.check("broken", False, "because")

    assert results.report() == 1


def test_results_exits_zero_when_everything_passes() -> None:
    results = Results()
    results.check("fine", True)

    assert results.report() == 0


def test_a_model_response_missing_the_licence_is_caught() -> None:
    """A response that omits it lets a caller use the model without meeting it."""
    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"data": [{"id": "x", "path": "cascade"}]}  # no licence fields

    results = Results()
    check_models(_client(app), results)

    assert any("许可" in name for name, _ in results.failed)


def test_a_model_response_with_an_unknown_path_is_caught() -> None:
    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "data": [
                {
                    "id": "x",
                    "path": "whatever",
                    "licence": "CC-BY-NC-4.0",
                    "commercial_use_permitted": False,
                }
            ]
        }

    results = Results()
    check_models(_client(app), results)

    assert any("路径" in name for name, _ in results.failed)


def test_speech_without_rate_headers_is_caught() -> None:
    """Without them a client guesses, and a wrong guess changes the pitch."""
    from fastapi.responses import Response

    app = FastAPI()

    @app.post("/v1/audio/speech")
    async def speech() -> Response:
        return Response(content=b"\x00\x01", media_type="audio/wav")

    results = Results()
    check_speech(_client(app), results)

    assert any("采样率" in name for name, _ in results.failed)


def test_a_healthy_service_passes_every_check() -> None:
    from mindsurf_omni.service.app import create_app
    from tests.test_app import FakeEngine

    results = Results()
    client = _client(create_app(FakeEngine()))
    check_models(client, results)
    check_speech(client, results)

    assert results.failed == []
    assert len(results.passed) >= 3


@pytest.mark.parametrize("status", [500, 404])
def test_a_service_that_errors_on_models_is_caught(status: int) -> None:
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse({}, status_code=status)

    results = Results()
    check_models(_client(app), results)

    assert results.failed


def test_an_unavailable_websocket_client_is_a_failure_not_a_skip() -> None:
    """ "Could not check" and "checked, it works" must not exit the same way.

    Printing a note and returning zero turns the first into the second: a green
    smoke run that never touched the realtime path. A sister project lost a
    fleet of devices to this shape -- a catch branch that treated blocked
    playback as playback finished, so everything went silent and nothing
    reported a fault.
    """
    import builtins

    from scripts.smoke_service import Results, check_realtime

    real_import = builtins.__import__

    def without_websockets(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("websockets"):
            raise ImportError("no websockets")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    results = Results()
    builtins.__import__ = without_websockets  # type: ignore[assignment]
    try:
        check_realtime("http://test", results)
    finally:
        builtins.__import__ = real_import  # type: ignore[assignment]

    assert not results.passed
    assert results.failed and "未检查" in results.failed[0][1]
    assert results.report() == 1  # non-zero exit
