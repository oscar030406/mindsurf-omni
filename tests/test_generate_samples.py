"""Sample generation, especially what it records about failures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from scripts.generate_speech_samples import generate, load_probes


def _probes(tmp_path: Path, count: int = 3) -> Path:
    path = tmp_path / "probes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"zh{i:03d}", "prompt": f"问题{i}"}, ensure_ascii=False)
            for i in range(count)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_probes_load_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = _probes(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert len(load_probes(path)) == 3


def test_an_empty_probe_file_is_refused(tmp_path: Path) -> None:
    """Generating nothing and reporting success would look like a passing run."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="no probes"):
        load_probes(empty)


class _Service:
    """A stand-in service, with configurable failures."""

    def __init__(self, *, path: str = "cascade", fail_audio_for: set[str] | None = None) -> None:
        self.path = path
        self.fail_audio_for = fail_audio_for or set()
        self.seen: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "mindsurf-omni",
                            "path": self.path,
                            "licence": "CC-BY-NC-4.0",
                            "components": [],
                        }
                    ]
                },
            )
        if request.url.path == "/v1/chat/completions":
            prompt = json.loads(request.content)["messages"][0]["content"]
            self.seen.append(prompt)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": f"回答{prompt}"}}]},
            )
        if request.url.path == "/v1/audio/speech":
            spoken = json.loads(request.content)["input"]
            if any(marker in spoken for marker in self.fail_audio_for):
                return httpx.Response(500)
            return httpx.Response(200, content=b"RIFF" + b"\x00" * 40)
        return httpx.Response(404)


def _run(service: _Service, tmp_path: Path, count: int = 3, text_source: str = "model") -> dict:
    transport = httpx.MockTransport(service.handler)
    original = httpx.AsyncClient

    class Patched(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: object) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)  # type: ignore[arg-type]

    httpx.AsyncClient = Patched  # type: ignore[misc]
    try:
        return asyncio.run(
            generate(
                "http://test",
                load_probes(_probes(tmp_path, count)),
                tmp_path / "out",
                timeout=5.0,
                text_source=text_source,
                sampling={"temperature": 0.7, "top_p": 0.9},
            )
        )
    finally:
        httpx.AsyncClient = original  # type: ignore[misc]


def test_the_manifest_records_which_path_produced_the_audio(tmp_path: Path) -> None:
    """A report that cannot say this has measured nothing."""
    report = _run(_Service(path="native"), tmp_path)

    assert report["generated_by"]["path"] == "native"
    assert report["generated_by"]["licence"] == "CC-BY-NC-4.0"


def test_the_manifest_records_the_sampling_that_produced_the_replies(tmp_path: Path) -> None:
    """Two runs at different temperatures are two measurements, not one repeated.

    Nothing else in the artifacts distinguishes them, so a comparison across
    them would look like a model difference.
    """
    report = _run(_Service(), tmp_path, count=1)

    assert report["generated_by"]["sampling"] == {"temperature": 0.7, "top_p": 0.9}


def test_a_floor_run_records_no_sampling_because_nothing_was_sampled(tmp_path: Path) -> None:
    report = _run(_Service(), tmp_path, count=1, text_source="probe")

    assert report["generated_by"]["sampling"] is None


def test_the_model_reply_is_the_reference_not_the_prompt(tmp_path: Path) -> None:
    """CER measures whether the synthesiser said the reply, not answer quality."""
    report = _run(_Service(), tmp_path, count=1)

    sample = report["samples"][0]
    assert sample["reference_text"] == "回答问题0"
    assert sample["prompt"] == "问题0"


def test_a_floor_run_says_the_model_never_spoke(tmp_path: Path) -> None:
    """The most flattering wrong number this project could print.

    Speaking the probe text measures what the synthesiser and the judge cost
    between them. Read as a model score it looks excellent, so the manifest
    has to carry the distinction rather than leaving it to whoever remembers
    which command was run.
    """
    report = _run(_Service(), tmp_path, count=1, text_source="probe")

    assert report["generated_by"]["text_source"] == "probe"
    sample = report["samples"][0]
    assert sample["reference_text"] == "问题0"  # the probe, not a reply


def test_a_normal_run_is_marked_as_the_model_speaking(tmp_path: Path) -> None:
    report = _run(_Service(), tmp_path, count=1)

    assert report["generated_by"]["text_source"] == "model"


def test_failed_samples_are_named_not_silently_dropped(tmp_path: Path) -> None:
    """A shrunken sample size changes every noise floor computed from it."""
    report = _run(_Service(fail_audio_for={"问题1"}), tmp_path)

    assert report["probe_count"] == 3
    assert report["generated"] == 2
    assert report["failed"] == ["zh001"]


def test_a_service_with_no_model_is_refused_rather_than_producing_silence(
    tmp_path: Path,
) -> None:
    service = _Service()
    service.handler = lambda request: httpx.Response(200, json={"data": []})  # type: ignore[assignment,method-assign]

    with pytest.raises(SystemExit, match="no model"):
        _run(service, tmp_path)


def test_audio_files_land_where_the_manifest_says(tmp_path: Path) -> None:
    report = _run(_Service(), tmp_path, count=2)

    for sample in report["samples"]:
        assert Path(sample["audio_path"]).exists()
