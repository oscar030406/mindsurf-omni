"""The guard that keeps a timing run off a shared card."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest
from scripts.measure_voxcpm_cost import refuse_if_the_card_is_busy


def test_a_service_answering_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sharing the card has already produced a reading in the wrong direction on
    this project -- the arm with an extra model in it came back faster. A timing
    run that starts anyway produces a number nobody can tell is wrong."""

    def answering(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("should not reach the body")

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())

    with pytest.raises(SystemExit) as stopped:
        refuse_if_the_card_is_busy("http://127.0.0.1:8099")

    assert "health" in str(stopped.value)


def test_nothing_answering_lets_the_run_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refused)

    refuse_if_the_card_is_busy("http://127.0.0.1:8099")
