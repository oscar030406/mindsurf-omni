"""The service surface, exercised the way the backend will call it."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from mindsurf_omni.contract import ComponentInfo, TokenSpec
from mindsurf_omni.service.app import create_app
from mindsurf_omni.service.engine import (
    EngineDescription,
    GenerationSettings,
    SpeechChunk,
    SpeechEngine,
)

SPEC = TokenSpec(
    text_vocab_size=6400,
    audio_codebooks=8,
    audio_codebook_size=2048,
    audio_frame_rate_hz=12.5,
    special_tokens={"im_start": 1, "im_end": 2, "audio_start": 14},
    audio_special_tokens={"pad": 2049, "stop": 2050, "spk": 2051},
)


class FakeEngine(SpeechEngine):
    def describe(self) -> EngineDescription:
        return EngineDescription(
            path="cascade",
            components=[ComponentInfo(name="thinker", parameters=89_864_448)],
        )

    def token_spec(self) -> TokenSpec:
        return SPEC

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        return "今天天气怎么样", "zh"

    async def complete(  # type: ignore[override]
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        for piece in ["今天", "天气", "很好。"]:
            yield piece

    async def speak(  # type: ignore[override]
        self, text: str, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(pcm=b"\x00\x01" * 8, text=text, is_final=True)

    async def respond(  # type: ignore[override]
        self, pcm: bytes, sample_rate: int, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(pcm=b"\x00\x01" * 8, text="今天天气很好。", is_final=True)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FakeEngine()))


@pytest.fixture
def bare() -> TestClient:
    return TestClient(create_app(None))


def test_models_reports_the_path_and_the_licence(client: TestClient) -> None:
    """A quality report is meaningless without knowing which path answered."""
    body = client.get("/v1/models").json()["data"][0]

    assert body["path"] == "cascade"
    assert body["fallback_available"] is True
    # The restriction is inherited from the training data and carried by every
    # derivative, so it travels with every response.
    assert body["commercial_use_permitted"] is False
    assert body["licence"] == "CC-BY-NC-4.0"


def test_an_unconfigured_service_says_so_instead_of_faking_success(bare: TestClient) -> None:
    """An integration that passes against stubs fails on the day the model lands."""
    assert bare.get("/v1/models").json()["data"] == []
    for path in ["/v1/token-spec", "/v1/voices"]:
        response = bare.get(path)
        assert response.status_code == 503
        assert "engine" in response.json()["detail"]


def test_chat_completion_matches_the_openai_shape(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "今天天气很好。"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_streaming_chat_emits_sse_deltas_then_done(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "你好"}], "stream": True},
    ) as response:
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    assert '"content": "今天"' in lines[0]


def test_transcription_reports_the_detected_language(client: TestClient) -> None:
    """Chinese audio read as English is a failure the caller should be able to see."""
    body = client.post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000).json()

    assert body["text"] == "今天天气怎么样"
    assert body["language"] == "zh"
    assert body["duration_seconds"] == pytest.approx(1.0)


def test_empty_audio_is_rejected_not_transcribed(client: TestClient) -> None:
    assert client.post("/v1/audio/transcriptions", content=b"").status_code == 400


def test_speech_declares_its_sample_rate_in_headers(client: TestClient) -> None:
    """The client must not have to guess how to play what it received."""
    response = client.post("/v1/audio/speech", json={"input": "你好"})

    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-encoding"] == "pcm_s16le"
    assert response.content == b"\x00\x01" * 8


def test_realtime_turn_produces_text_then_audio(client: TestClient) -> None:
    with client.websocket_connect("/v1/realtime") as socket:
        assert socket.receive_json()["type"] == "session.created"
        socket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"\x00\x01" * 100).decode(),
            }
        )
        socket.send_json({"type": "input_audio_buffer.commit"})

        kinds = [socket.receive_json()["type"] for _ in range(5)]

    assert kinds == [
        "response.created",
        "response.text.delta",
        "response.audio.delta",
        "response.audio.done",
        "response.done",
    ]


def test_committing_an_empty_buffer_errors_rather_than_hanging(client: TestClient) -> None:
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "input_audio_buffer.commit"})

        event = socket.receive_json()

    assert event["type"] == "error"
    assert "no audio" in event["error"]["message"]


def test_an_unknown_event_is_answered_not_ignored(client: TestClient) -> None:
    """Silence would leave the client waiting for a reply that never comes."""
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "response.make_coffee"})

        event = socket.receive_json()

    assert event["type"] == "error"
    assert "unknown event type" in event["error"]["message"]


def test_cancel_drops_the_buffer_so_it_cannot_leak_into_the_next_turn(
    client: TestClient,
) -> None:
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "response.cancel"})
        assert socket.receive_json()["cancelled"] is True

        socket.send_json({"type": "input_audio_buffer.commit"})
        event = socket.receive_json()

    assert event["type"] == "error"


def test_token_spec_carries_what_a_client_needs_to_build_prompts(client: TestClient) -> None:
    body = client.get("/v1/token-spec").json()

    assert body["text_vocab_size"] == 6400
    assert body["audio_codebooks"] == 8
    assert body["input_sample_rate"] == 16_000
    assert body["output_sample_rate"] == 24_000
    assert body["special_tokens"]["audio_start"] == 14


def test_the_factory_entrypoint_works_with_no_arguments() -> None:
    """The container runs `uvicorn --factory create_app`, which passes nothing.

    A signature change that broke this would only show up at deploy time.
    """
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert {"/v1/models", "/v1/realtime", "/v1/chat/completions"} <= paths
