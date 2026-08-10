"""The service surface, exercised the way the backend will call it."""

from __future__ import annotations

import asyncio
import base64
import struct
from collections.abc import AsyncIterator
from pathlib import Path

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
        self,
        pcm: bytes,
        sample_rate: int,
        settings: GenerationSettings,
        history: list[dict[str, str]] | None = None,
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


def test_a_stage_that_is_not_wired_answers_503_rather_than_a_traceback() -> None:
    """A 500 sends the caller to a log the caller cannot read.

    The service already answers 503-with-a-reason when no engine exists; a
    stage that is present but unwired must fail the same way, so an integration
    handles one failure shape instead of two.
    """
    from mindsurf_omni.service.config import ConfigurationError

    class HalfBuilt(FakeEngine):
        async def complete(  # type: ignore[override]
            self, messages: list[dict[str, str]], settings: GenerationSettings
        ) -> AsyncIterator[str]:
            raise ConfigurationError("the widget stage is not wired; set MINDSURF_WIDGET")
            yield ""  # pragma: no cover - makes this an async generator

    response = TestClient(create_app(HalfBuilt())).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "你好"}]}
    )

    assert response.status_code == 503
    assert "MINDSURF_WIDGET" in response.json()["detail"]


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
    response = client.post("/v1/audio/speech", json={"input": "你好", "response_format": "pcm"})

    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-encoding"] == "pcm_s16le"
    assert response.content == b"\x00\x01" * 8


def test_speech_asked_for_wav_returns_a_container_a_decoder_will_open(
    client: TestClient,
) -> None:
    """A body labelled audio/wav that carries bare PCM is refused by every decoder.

    The evaluation harness writes this response to a .wav and hands the path to
    a recogniser, so the failure surfaces as every sample transcribing to the
    empty string -- a whole eval run that reports nothing rather than an error.
    """
    response = client.post("/v1/audio/speech", json={"input": "你好"})

    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    assert struct.unpack("<I", response.content[24:28])[0] == 24_000
    assert response.content[44:] == b"\x00\x01" * 8


def test_the_wav_length_is_real_and_not_a_placeholder(client: TestClient) -> None:
    """A placeholder size read as signed is negative, and a client crashed on it.

    A sister project on this team had exactly this: an upstream that streamed
    0xFFFFFFFF chunk sizes, and an Android parser that computed a negative
    offset from them and fell over. ffmpeg tolerates the placeholder; a
    hand-written parser does not, and there is one downstream of this service.
    """
    body = client.post("/v1/audio/speech", json={"input": "你好"}).content

    audio = len(body) - 44
    assert struct.unpack("<I", body[4:8])[0] == 36 + audio
    assert struct.unpack("<I", body[40:44])[0] == audio
    # Both sizes survive a signed read, which is the failure being prevented.
    assert struct.unpack("<i", body[4:8])[0] > 0
    assert struct.unpack("<i", body[40:44])[0] > 0


def test_speech_asked_for_pcm_carries_no_container(client: TestClient) -> None:
    """Raw PCM is the streaming format; prefixing a header would corrupt it."""
    response = client.post("/v1/audio/speech", json={"input": "你好", "response_format": "pcm"})

    assert response.content[:4] != b"RIFF"
    assert response.headers["content-type"].startswith("application/octet-stream")


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


def test_the_integration_guide_documents_exactly_what_exists() -> None:
    """A guide that names a missing endpoint sends the backend to a 404.

    Checked rather than trusted, because the guide and the routes are edited
    at different times by different concerns.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    guide = (root / "docs" / "INTEGRATION.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"(?:POST|GET|WS) (/v1/[a-z/-]+)", guide))

    implemented = {
        route.path for route in create_app().routes if getattr(route, "path", "").startswith("/v1")
    }

    assert documented == implemented, (
        f"documented but missing: {sorted(documented - implemented)}; "
        f"implemented but undocumented: {sorted(implemented - documented)}"
    )


def test_a_configuration_error_reaches_the_caller_instead_of_crashing_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash-looping container gives an operator no log and no health check.

    The error has to arrive as a 503 body, naming the missing file, so it can
    be read from outside the container.
    """
    monkeypatch.setenv("MINDSURF_ENGINE", "cascade")
    monkeypatch.setenv("MINDSURF_WEIGHTS", str(tmp_path / "absent"))

    app = create_app()  # must not raise
    response = TestClient(app).get("/v1/voices")

    assert response.status_code == 503
    assert "tokenizer=" in response.json()["detail"]


def test_an_engine_passed_in_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests and embedders supply their own; the environment must not override."""
    monkeypatch.setenv("MINDSURF_ENGINE", "cascade")
    monkeypatch.setenv("MINDSURF_WEIGHTS", "/nowhere")

    client = TestClient(create_app(FakeEngine()))

    assert client.get("/v1/models").json()["data"][0]["path"] == "cascade"


def test_the_licence_endpoint_serves_the_chain_behind_the_conclusion(
    client: TestClient,
) -> None:
    """A caller deciding what they may ship needs the unread terms, not just "no"."""
    body = client.get("/v1/licence").json()

    assert body["conclusion"]["commercial_use_permitted"] is False
    unverified = [asset for asset in body["assets"] if not asset["verified"]]
    assert unverified, "if everything is verified, delete this assertion"
    # Null, not false: nobody has read these.
    assert all(asset["commercial_use"] is None for asset in unverified)


def test_the_licence_endpoint_is_available_without_an_engine(bare: TestClient) -> None:
    """It describes what the image carries, which does not depend on a model."""
    assert bare.get("/v1/licence").status_code == 200


def test_a_turn_reports_the_context_state_so_forgetting_is_visible(
    client: TestClient,
) -> None:
    """History that shortens silently reads as the model forgetting for no reason."""
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"\x00" * 32000).decode(),
            }
        )
        socket.send_json({"type": "input_audio_buffer.commit"})

        events = [socket.receive_json() for _ in range(5)]

    done = events[-1]
    assert done["type"] == "response.done"
    assert done["context"]["turns"] == 2  # the user's and the assistant's
    assert "dropped_turns" in done["context"]


def test_a_session_can_be_cleared_between_conversations(client: TestClient) -> None:
    """Carrying one caller's history into the next is worse than losing it."""
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "session.clear"})

        event = socket.receive_json()

    assert event["type"] == "session.created"
    assert event["context"]["turns"] == 0


def test_an_engine_that_cannot_speak_arbitrary_text_answers_503() -> None:
    """The native model says its own words; reading someone else's is not a wire it lacks.

    Refusing inside the StreamingResponse would land after the headers, and the
    caller would see a truncated body rather than the reason.
    """
    from mindsurf_omni.service.config import ConfigurationError

    class OnlyItsOwnWords(FakeEngine):
        def speak(  # type: ignore[override]
            self, text: str, settings: GenerationSettings
        ) -> AsyncIterator[SpeechChunk]:
            raise ConfigurationError("the native path has no text-to-speech step")

    response = TestClient(create_app(OnlyItsOwnWords())).post(
        "/v1/audio/speech", json={"input": "随便一句话"}
    )

    assert response.status_code == 503
    assert "text-to-speech" in response.json()["detail"]


class SlowEngine(FakeEngine):
    """Streams for a long time unless abandoned, and records the abandonment.

    The waits are real but only run to completion when nobody cancels; a
    cancelled turn abandons the generator at the first await, so the test's
    duration does not depend on them.
    """

    def __init__(self) -> None:
        self.closed = False

    async def respond(  # type: ignore[override]
        self,
        pcm: bytes,
        sample_rate: int,
        settings: GenerationSettings,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[SpeechChunk]:
        try:
            for index in range(50):
                yield SpeechChunk(pcm=b"\x00\x01" * 8, text=f"第{index}句。", is_final=False)
                await asyncio.sleep(0.05)
        finally:
            # What the leak fix watches for: the consumer leaving must reach
            # the producer. In the real engine this stops the GPU.
            self.closed = True


def test_cancel_lands_mid_stream_not_after_the_turn() -> None:
    """The old loop drove generation inline, so cancel waited out the whole turn."""
    engine = SlowEngine()
    with TestClient(create_app(engine)).websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        assert socket.receive_json()["type"] == "response.created"
        # One delta proves streaming started; fifty more would prove it ended.
        first = socket.receive_json()
        assert first["type"] in ("response.text.delta", "response.audio.delta")

        socket.send_json({"type": "response.cancel"})
        while True:
            event = socket.receive_json()
            if event["type"] == "response.done":
                break
        assert event["cancelled"] is True
        # 沒有 audio.done：被打断的回合没有说完，装作说完就是谎报。
        assert engine.closed, "取消没有传到引擎——真机上这就是 GPU 泄漏"


def test_barge_in_audio_appended_during_the_reply_survives_the_cancel() -> None:
    """The interjection that caused the cancel must not be eaten by it."""
    engine = SlowEngine()
    with TestClient(create_app(engine)).websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        assert socket.receive_json()["type"] == "response.created"
        socket.receive_json()

        # The user starts talking over the reply, then the client cancels.
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01" * 8).decode()}
        )
        socket.send_json({"type": "response.cancel"})
        while True:
            event = socket.receive_json()
            if event["type"] == "response.done":
                break
        assert event["cancelled"] is True

        # Committing now must answer from the interjection, not error on an
        # empty buffer -- which is what the idle-cancel semantics would give.
        socket.send_json({"type": "input_audio_buffer.commit"})
        assert socket.receive_json()["type"] == "response.created"


def test_the_partial_reply_enters_history_as_what_was_said() -> None:
    engine = SlowEngine()
    with TestClient(create_app(engine)).websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        socket.receive_json()
        socket.receive_json()
        socket.send_json({"type": "response.cancel"})
        while True:
            event = socket.receive_json()
            if event["type"] == "response.done":
                break

    assert event["context"]["turns"] >= 2


class RecordingEngine(FakeEngine):
    """Reports what the session handed it, and what it heard."""

    def __init__(self) -> None:
        self.histories: list[list[dict[str, str]]] = []

    async def respond(  # type: ignore[override]
        self,
        pcm: bytes,
        sample_rate: int,
        settings: GenerationSettings,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[SpeechChunk]:
        self.histories.append(list(history or []))
        yield SpeechChunk(pcm=b"", transcript="那它多少钱")
        yield SpeechChunk(pcm=b"\x00\x01" * 8, text="两百块。", is_final=True)


def test_the_second_turn_carries_the_first_one_into_the_prompt() -> None:
    """The session counted turns it never sent, so every turn was the first one.

    A caller asking "那它多少钱" after "这个杯子好看吗" gets an answer about
    nothing unless the history reaches the prompt. The context figures in
    response.done described a history that was bookkeeping only.
    """
    engine = RecordingEngine()
    client = TestClient(create_app(engine))

    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()  # session.created
        for _ in range(2):
            socket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\x00" * 8).decode(),
                }
            )
            socket.send_json({"type": "input_audio_buffer.commit"})
            while socket.receive_json()["type"] != "response.done":
                pass

    assert engine.histories[0] == []
    assert engine.histories[1] == [
        {"role": "user", "content": "那它多少钱"},
        {"role": "assistant", "content": "两百块。"},
    ]


def test_the_transcript_is_reported_so_a_client_can_show_what_it_heard() -> None:
    """Declared in SERVER_EVENTS from the start and never sent until 2026-08-10."""
    client = TestClient(create_app(RecordingEngine()))

    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        events = []
        while (event := socket.receive_json())["type"] != "response.done":
            events.append(event)

    heard = [
        event for event in events if event["type"].endswith("input_audio_transcription.completed")
    ]
    assert [event["transcript"] for event in heard] == ["那它多少钱"]


def test_clearing_a_session_repeats_the_sample_rates() -> None:
    """A client reads them in one place: the session.created branch.

    Sending a second, thinner session.created leaves that branch reading
    undefined and the playback rate is what it sets from it.
    """
    client = TestClient(create_app(FakeEngine()))

    with client.websocket_connect("/v1/realtime") as socket:
        opening = socket.receive_json()
        socket.send_json({"type": "session.clear"})
        cleared = socket.receive_json()

    assert cleared["type"] == "session.created"
    for field in ("input_sample_rate", "output_sample_rate", "encoding"):
        assert cleared[field] == opening[field]
    assert cleared["context"]["turns"] == 0


def test_clearing_a_session_stops_the_history_reaching_the_next_caller() -> None:
    """The reason session.clear exists, and the one thing the wiring must not lose.

    Before history reached the prompt this line cost nothing to break. Now a
    clear that does not clear puts one caller's words in the next caller's
    prompt.
    """
    engine = RecordingEngine()
    client = TestClient(create_app(engine))

    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        while socket.receive_json()["type"] != "response.done":
            pass
        socket.send_json({"type": "session.clear"})
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        while socket.receive_json()["type"] != "response.done":
            pass

    assert engine.histories[1] == []


def test_a_cancelled_turn_still_remembers_what_the_caller_said() -> None:
    """Barge-in is the main path, so the turn it interrupts has to enter history.

    The transcript is yielded before any audio for exactly this reason: a turn
    cut off halfway still happened, and the next one has to know what it was
    about.
    """

    class SlowAfterTranscript(FakeEngine):
        def __init__(self) -> None:
            self.histories: list[list[dict[str, str]]] = []

        async def respond(  # type: ignore[override]
            self,
            pcm: bytes,
            sample_rate: int,
            settings: GenerationSettings,
            history: list[dict[str, str]] | None = None,
        ) -> AsyncIterator[SpeechChunk]:
            self.histories.append(list(history or []))
            yield SpeechChunk(pcm=b"", transcript="那它多少钱")
            for _ in range(50):
                yield SpeechChunk(pcm=b"\x00\x01" * 8, text="两")
                await asyncio.sleep(0.05)

    engine = SlowAfterTranscript()
    client = TestClient(create_app(engine))

    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        while socket.receive_json()["type"] != "response.text.delta":
            pass
        socket.send_json({"type": "response.cancel"})
        while socket.receive_json()["type"] != "response.done":
            pass
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 8).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        while socket.receive_json()["type"] != "response.done":
            pass

    assert engine.histories[1][0] == {"role": "user", "content": "那它多少钱"}
