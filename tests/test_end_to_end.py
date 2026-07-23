"""The whole pipeline, wired together against stubs.

Every stage has its own tests; this is the one that would catch a mismatch
between them -- a manifest field one script writes and the next does not read,
a transcript column named differently at each end. Those pass every unit test
and fail the first time the pipeline is run for real, which is the moment the
GPU time has already been spent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from scripts.evaluate_speech import load, score
from scripts.generate_speech_samples import generate, load_probes
from scripts.transcribe_samples import transcribe_all


class _Service:
    """A service that answers well enough to exercise the chain."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "mindsurf-omni",
                            "path": "cascade",
                            "licence": "CC-BY-NC-4.0",
                            "components": [{"name": "thinker"}, {"name": "sensevoice-small"}],
                        }
                    ]
                },
            )
        if path == "/v1/chat/completions":
            prompt = json.loads(request.content)["messages"][0]["content"]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": f"回答{prompt}"}}]}
            )
        if path == "/v1/audio/speech":
            return httpx.Response(200, content=b"RIFF" + b"\x00" * 40)
        return httpx.Response(404)


def _generate(tmp_path: Path, probe_count: int) -> dict:
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        "\n".join(
            json.dumps({"id": f"zh{i:03d}", "prompt": f"问题{i}"}, ensure_ascii=False)
            for i in range(probe_count)
        ),
        encoding="utf-8",
    )
    transport = httpx.MockTransport(_Service().handler)
    original = httpx.AsyncClient

    class Patched(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: object) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)  # type: ignore[arg-type]

    httpx.AsyncClient = Patched  # type: ignore[misc]
    try:
        return asyncio.run(generate("http://test", load_probes(probes), tmp_path / "audio", 5.0))
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]


def test_the_three_stages_hand_off_without_losing_a_field(tmp_path: Path) -> None:
    """Generation writes, transcription reads, scoring reads what both wrote."""
    manifest = _generate(tmp_path, probe_count=120)

    # Stage two: an independent judge that hears exactly what was meant.
    rows = transcribe_all(manifest, lambda path: "回答问题0", lineage="whisper")

    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )

    # Stage three reads it back off disk, as it would in a real run.
    report = score("candidate", load(scored), {"cer": 0.05})

    assert report.measurements["cer"].sample_size == 120
    assert "silent_rate" in report.measurements


def test_a_perfect_synthesiser_scores_zero_and_an_indifferent_one_does_not(
    tmp_path: Path,
) -> None:
    """The chain must be able to tell those apart, or it measures nothing."""
    manifest = _generate(tmp_path, probe_count=120)

    faithful = transcribe_all(manifest, lambda p: "", lineage="whisper")
    for row in faithful:
        row["transcript"] = row["reference_text"]  # heard exactly what was said

    wrong = transcribe_all(manifest, lambda p: "完全不同的内容", lineage="whisper")

    good = score("good", [_sample(row) for row in faithful], {"cer": 0.05})
    bad = score("bad", [_sample(row) for row in wrong], {"cer": 0.05})

    assert good.measurements["cer"].value == 0.0
    assert bad.measurements["cer"].value > 0.5


def _sample(row: dict) -> object:
    from scripts.evaluate_speech import Sample

    return Sample(
        prompt=row["prompt"],
        reference_text=row["reference_text"],
        transcript=row.get("transcript"),
    )


def test_silence_travels_the_whole_chain_as_a_failure(tmp_path: Path) -> None:
    """A model that says nothing must not arrive at scoring looking perfect."""
    manifest = _generate(tmp_path, probe_count=120)
    rows = transcribe_all(manifest, lambda p: "", lineage="whisper")

    report = score("silent", [_sample(row) for row in rows], {"cer": 0.05})

    assert report.measurements["silent_rate"].value == 1.0
    assert "120 of 120 produced no speech" in report.measurements["silent_rate"].note


def test_the_path_that_produced_the_audio_survives_to_the_report(tmp_path: Path) -> None:
    """A report that cannot say which path answered has measured nothing."""
    manifest = _generate(tmp_path, probe_count=10)

    assert manifest["generated_by"]["path"] == "cascade"
    assert manifest["generated_by"]["licence"] == "CC-BY-NC-4.0"
