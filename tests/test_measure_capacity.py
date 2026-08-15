"""The capacity curve the backend's queue limit is supposed to be read off."""

from __future__ import annotations

import asyncio

from scripts.measure_capacity import nearest_rank, sweep_level


def test_percentiles_are_values_someone_actually_waited() -> None:
    """An interpolated P95 is a latency no request experienced."""
    values = [10.0, 20.0, 30.0, 40.0]

    assert nearest_rank(values, 0.50) == 20.0
    assert nearest_rank(values, 0.95) == 40.0
    assert nearest_rank([], 0.5) != nearest_rank([], 0.5)  # nan


def test_a_level_counts_every_request_and_reports_the_failures(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A refused turn is data: dropping it would report the survivors' latency."""
    import scripts.measure_capacity as capacity

    seen: list[int] = []

    async def fake(client: object, stage: str, payload: object) -> tuple[float, str, int]:
        seen.append(1)
        # Every third one fails, warm-up included.
        return (5.0, "HTTPStatusError: 503" if len(seen) % 3 == 0 else "", 42)

    class _Client:
        def __init__(self, **_: object) -> None: ...
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_: object) -> None: ...

    monkeypatch.setattr(capacity, "one_request", fake)
    monkeypatch.setattr(capacity.httpx, "AsyncClient", _Client)

    point = asyncio.run(sweep_level("http://x", "asr", b"", level=2, requests=6, timeout=1.0))

    assert point["requests"] == 6  # the warm-up is not counted
    assert point["failed"] == 2
    assert point["throughput_per_second"] > 0
    assert point["p95_ms"] == 5.0
    assert point["median_characters"] == 42
