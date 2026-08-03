"""The paired chunk sweep, checked on the part that is easy to get wrong.

The measurement itself needs a GPU and a checkpoint. The bookkeeping around it
does not, and the bookkeeping is where a sweep quietly stops being paired: an
arm order that never rotates, a round whose arms replied differently counted
anyway, a baseline that does not subtract to zero.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from scripts.measure_chunk_latency import run


@dataclass
class _Config:
    chunk_frames: int = 4


class _Chunk:
    def __init__(self, pcm: bytes = b"", text: str = "") -> None:
        self.pcm = pcm
        self.text = text


_now = 0.0


def _advance(seconds: float) -> None:
    global _now
    _now += seconds


class _FakeTime:
    """perf_counter driven by the engine's declared waits, nothing else.

    The engine used to really sleep 1 ms per chunk unit and let the script
    time it with the real perf_counter. Windows resolves sleeps at ~15.6 ms,
    so 1 ms and 8 ms land on the same tick under load, and the direction
    assertion (8 slower than 1) flaked whenever the suite was busy -- which is
    how this was found. The waits are bookkeeping now, and the timing the
    script reads is exactly the timing the engine declared.
    """

    @staticmethod
    def perf_counter() -> float:
        return _now


class _Engine:
    """Answers in a fixed time per chunk size, and records the order asked."""

    def __init__(self, *, drift_after: int | None = None) -> None:
        self._config = _Config()
        self.order: list[int] = []
        self._calls = 0
        self._drift_after = drift_after

    async def respond(self, pcm: bytes, rate: int, settings: Any) -> Any:
        size = self._config.chunk_frames
        self.order.append(size)
        self._calls += 1
        # Bigger chunks wait for more frames, so first audio arrives later.
        _advance(0.001 * size)
        await asyncio.sleep(0)
        drifted = self._drift_after is not None and self._calls > self._drift_after
        yield _Chunk(text="乙" if drifted else "甲")
        yield _Chunk(pcm=b"\x00\x00")

    def close(self) -> None:  # pragma: no cover - the script never calls it
        pass


def _args(**overrides: Any) -> argparse.Namespace:
    base = {
        "stimulus": None,
        "rounds": 4,
        "levels": [1, 2, 4, 8],
        "seed": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "effect": 200.0,
        "output": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _run(engine: _Engine, args: argparse.Namespace, monkeypatch: Any) -> dict[str, Any]:
    from mindsurf_omni.service import factory

    global _now
    _now = 0.0
    monkeypatch.setattr("scripts.measure_chunk_latency.time", _FakeTime())
    monkeypatch.setattr(factory, "build", lambda settings: engine)
    monkeypatch.setattr(
        "scripts.measure_chunk_latency.load_stimulus", lambda path: (b"\x00\x00", 16_000)
    )
    # Stands in for torch, which the dev environment does not carry. The real
    # one must stay unguarded -- see seed_rng.
    monkeypatch.setattr("scripts.measure_chunk_latency.seed_rng", lambda seed: None)
    return asyncio.run(run(args))


def test_every_arm_leads_at_least_once(monkeypatch: Any) -> None:
    """Rotation is the whole defence against a warm-up trend sitting on one arm."""
    engine = _Engine()
    _run(engine, _args(rounds=4), monkeypatch)

    leaders = {engine.order[index * 4] for index in range(4)}
    assert leaders == {1, 2, 4, 8}


def test_the_baseline_subtracts_to_zero(monkeypatch: Any) -> None:
    engine = _Engine()
    report = _run(engine, _args(rounds=4), monkeypatch)

    verdicts = " ".join(report["paired_verdicts"])
    assert "chunk_frames 4 vs 4" not in verdicts
    assert report["dropped_unpaired"] == 0
    # Larger chunks wait for more frames; the direction must survive pairing.
    assert report["levels"]["8"]["mean_ms"] > report["levels"]["1"]["mean_ms"]


def test_a_round_whose_arms_replied_differently_is_dropped(monkeypatch: Any) -> None:
    """Without this the sweep silently subtracts two unrelated replies."""
    engine = _Engine(drift_after=6)
    report = _run(engine, _args(rounds=4), monkeypatch)

    assert report["dropped_unpaired"] >= 1
    assert report["dropped_unpaired"] < report["rounds"]
