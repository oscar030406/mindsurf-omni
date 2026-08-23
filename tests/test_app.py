"""The service surface, exercised the way the backend will call it."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
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
    TooLongForModel,
)
from mindsurf_omni.service.tts import SynthesiserUnavailable

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

    The error has to arrive as a 503 body, naming the variable behind the
    missing file, so it can be read from outside the container. The variable
    and not the path: this body is unauthenticated.
    """
    monkeypatch.setenv("MINDSURF_ENGINE", "cascade")
    monkeypatch.setenv("MINDSURF_WEIGHTS", str(tmp_path / "absent"))

    app = create_app()  # must not raise
    response = TestClient(app).get("/v1/voices")

    assert response.status_code == 503
    assert "MINDSURF_TOKENIZER" in response.json()["detail"]
    assert str(tmp_path) not in response.json()["detail"]


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


def test_transcription_carries_the_polished_text_when_a_polisher_is_wired() -> None:
    """The dictation product reads this field; null means the stage is absent."""

    class _Polishing(FakeEngine):
        async def polish(self, transcript: str) -> str:
            return transcript.replace("那个", "")

    body = (
        TestClient(create_app(_Polishing()))
        .post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000)
        .json()
    )

    assert body["text"] == "今天天气怎么样"
    assert body["polished"] == "今天天气怎么样"


def test_transcription_says_null_rather_than_echoing_when_no_polisher_is_wired(
    client: TestClient,
) -> None:
    """ "Not wired" and "nothing to change" must not look the same to the caller."""
    body = client.post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000).json()

    assert body["polished"] is None


async def test_a_stage_that_refuses_answers_503_even_when_it_streams() -> None:
    """An async generator runs no code until first iterated, so a refusal
    raised inside a StreamingResponse lands after the headers. Driving the real
    service with no synthesiser and no thinker: the non-streaming chat path
    answered 503 with the variables to set, while streaming chat, wav speech
    and pcm speech all returned an empty 200 the caller reads as
    IncompleteRead."""
    from mindsurf_omni.service.app import first_and_rest
    from mindsurf_omni.service.config import ConfigurationError

    async def refuses():
        raise ConfigurationError("no synthesiser is wired")
        yield  # pragma: no cover - unreachable, makes this a generator

    with pytest.raises(ConfigurationError):
        await first_and_rest(refuses())


async def test_an_empty_source_is_not_an_error() -> None:
    """Silence recognises as no audio and no text."""
    from mindsurf_omni.service.app import first_and_rest

    async def nothing():
        return
        yield  # pragma: no cover - unreachable, makes this a generator

    head, rest = await first_and_rest(nothing())

    assert head is None
    assert [item async for item in rest] == []


async def test_the_item_taken_for_the_check_is_put_back() -> None:
    """Taking it to move the refusal in front of the headers must not eat it."""
    from mindsurf_omni.service.app import first_and_rest

    async def three():
        for value in ("a", "b", "c"):
            yield value

    head, rest = await first_and_rest(three())

    assert head == "a"
    assert [item async for item in rest] == ["b", "c"]


# ---------------------------------------------------------------------------
# Found by driving the running service the way a client would, 2026-08-16.
# Each of these was a 500, a dropped connection or a silent wrong answer that
# the four polish criteria could not see.
# ---------------------------------------------------------------------------


def test_no_messages_is_a_field_error_not_a_500(client: TestClient) -> None:
    """It reached the chat template as an empty list and raised IndexError."""
    response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 422


class TooLongEngine(FakeEngine):
    """A model that refuses input longer than it was trained to answer."""

    async def complete(  # type: ignore[override]
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        raise TooLongForModel("one user message is 2005 tokens; this checkpoint answers up to 400")
        yield ""  # pragma: no cover - makes this an async generator


def test_a_message_the_model_cannot_answer_is_413_not_an_empty_200() -> None:
    """Measured: 977 tokens in one message came back as "" on a 200, eight times of eight."""
    client = TestClient(create_app(TooLongEngine()))

    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "长"}]}
    )

    assert response.status_code == 413
    assert "400" in response.json()["detail"]


def test_the_same_refusal_reaches_a_streaming_caller() -> None:
    """Raised inside the generator it would land after the headers, as a truncated 200."""
    streaming = TestClient(create_app(TooLongEngine()))

    response = streaming.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "长"}], "stream": True},
    )

    assert response.status_code == 413


def test_a_speed_that_is_not_applied_here_says_where_it_belongs(client: TestClient) -> None:
    """0.5, 1.0 and 2.0 returned byte-identical audio; the caller could not hear
    the difference. And the refusal names the player, because "not wired" reads
    as something the backend still owes -- the client then waits for it and
    nobody builds it."""
    response = client.post("/v1/audio/speech", json={"input": "你好", "speed": 1.5})

    assert response.status_code == 400
    assert "playbackRate" in response.json()["detail"]

    assert client.post("/v1/audio/speech", json={"input": "你好", "speed": 1.0}).status_code == 200


def test_a_voice_that_is_not_in_v1_voices_is_refused(client: TestClient) -> None:
    response = client.post("/v1/audio/speech", json={"input": "你好", "voice": "nobody"})

    assert response.status_code == 400
    assert "/v1/voices" in response.json()["detail"]


def test_a_frame_that_is_not_json_costs_the_frame_not_the_conversation(
    client: TestClient,
) -> None:
    """It escaped as JSONDecodeError and the socket died at 1006 with no reason."""
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_text("hello")

        event = socket.receive_json()
        assert event["type"] == "error"
        assert "not JSON" in event["error"]["message"]

        # Still usable: this is the whole point of answering rather than dying.
        socket.send_json({"type": "input_audio_buffer.commit"})
        assert socket.receive_json()["type"] == "error"


def test_a_binary_frame_is_answered_rather_than_raising_keyerror(client: TestClient) -> None:
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_bytes(b"\x00\x01\x02\x03")

        event = socket.receive_json()

    assert event["type"] == "error"
    assert "binary" in event["error"]["message"]


def test_audio_that_is_not_base64_is_answered_rather_than_raising(client: TestClient) -> None:
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "input_audio_buffer.append", "audio": "!!!not base64!!!"})

        event = socket.receive_json()
        assert event["type"] == "error"
        assert "base64" in event["error"]["message"]

        # And the buffer did not silently take the garbage.
        socket.send_json({"type": "input_audio_buffer.commit"})
        assert "no audio" in socket.receive_json()["error"]["message"]


def test_an_audio_field_that_is_not_a_string_is_answered(client: TestClient) -> None:
    """Whatever JSON carried reaches b64decode, so a number arrives as an int."""
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "input_audio_buffer.append", "audio": 12345})

        event = socket.receive_json()

    assert event["type"] == "error"
    assert "base64" in event["error"]["message"]


class SilentEngine(FakeEngine):
    """What the cascade produces when the recogniser heard no speech."""

    async def respond(  # type: ignore[override]
        self,
        pcm: bytes,
        sample_rate: int,
        settings: GenerationSettings,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(pcm=b"", transcript="")


def test_committed_audio_with_no_speech_in_it_says_so() -> None:
    """A bare response.done cannot separate "nothing was heard" from "the model said nothing"."""
    client = TestClient(create_app(SilentEngine()))

    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00" * 64).decode()}
        )
        socket.send_json({"type": "input_audio_buffer.commit"})

        kinds = []
        while True:
            event = socket.receive_json()
            kinds.append(event["type"])
            if event["type"] == "response.done":
                break
            if event["type"] == "error":
                message = event["error"]["message"]

    assert "error" in kinds
    assert "no speech" in message


class MuteEngine(FakeEngine):
    """A synthesiser that answers with no audio, which the hosted one does."""

    async def speak(  # type: ignore[override]
        self, text: str, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        raise SynthesiserUnavailable("the synthesiser returned no audio for '你好' twice")
        yield SpeechChunk(pcm=b"")  # pragma: no cover - makes this a generator


def test_a_synthesiser_that_returns_nothing_is_502_not_500() -> None:
    """Measured at 2 turns in 160 on a healthy network, and every one of them
    was an Internal Server Error with an empty body -- indistinguishable from a
    bug in this service."""
    client = TestClient(create_app(MuteEngine()), raise_server_exceptions=False)

    response = client.post("/v1/audio/speech", json={"input": "你好"})

    assert response.status_code == 502
    assert "no audio" in response.json()["detail"]


def test_an_emotion_the_build_cannot_deliver_is_refused_over_the_socket(
    client: TestClient,
) -> None:
    """The speech endpoint refuses it; this branch used to assign whatever
    arrived, so "angry" was accepted in silence and delivered as neutral."""
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "session.update", "emotion": "angry"})

        event = socket.receive_json()

    assert event["type"] == "error"
    assert "angry" in event["error"]["message"]


def test_a_voice_the_build_does_not_have_is_refused_over_the_socket(
    client: TestClient,
) -> None:
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "session.update", "voice": "nobody"})

        event = socket.receive_json()

    assert event["type"] == "error"
    assert "/v1/voices" in event["error"]["message"]


def test_an_accepted_update_is_acknowledged(client: TestClient) -> None:
    """A silent success and a silent failure look identical from the other end."""
    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json({"type": "session.update", "emotion": "care"})

        event = socket.receive_json()

    assert event["type"] == "session.updated"
    assert event["emotion"] == "care"
    assert event["voice"] == "default"


def test_every_event_the_service_sends_is_declared_in_the_contract() -> None:
    """An event a client is not told about is one it will not handle."""
    from mindsurf_omni.contract import SERVER_EVENTS

    assert "session.updated" in SERVER_EVENTS


def test_a_multipart_upload_is_refused_rather_than_read_as_audio(client: TestClient) -> None:
    """The standard OpenAI call for this route answered 200 with the MIME envelope
    decoded as speech: 128210 bytes reached the recogniser where 128000 were sent."""
    response = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.pcm", b"\x00\x01" * 16_000)}
    )

    assert response.status_code == 415
    assert "multipart" in response.json()["detail"]


def test_the_multipart_refusal_does_not_depend_on_the_caller_shouting(
    client: TestClient,
) -> None:
    """Header values are case-insensitive (RFC 9110), so a startswith against the raw
    value lets the same upload through in capitals."""
    response = client.post(
        "/v1/audio/transcriptions",
        content=b"\x00\x01" * 16_000,
        headers={"Content-Type": "MULTIPART/FORM-DATA; boundary=zz"},
    )

    assert response.status_code == 415


@pytest.mark.parametrize("coding", ["gzip", "br", "deflate", "zstd", "GZIP", "gzip, br"])
def test_a_compressed_body_is_refused_rather_than_read_as_audio(
    client: TestClient, coding: str
) -> None:
    """Nothing in this path decompresses, so the compressed bytes recognise as
    plausible speech. Every coding fails that way, not only gzip."""
    response = client.post(
        "/v1/audio/transcriptions",
        content=b"\x00\x01" * 16_000,
        headers={"Content-Encoding": coding},
    )

    assert response.status_code == 415


@pytest.mark.parametrize("coding", ["identity", "IDENTITY"])
def test_an_uncompressed_body_is_not_refused_for_saying_so(
    client: TestClient, coding: str
) -> None:
    response = client.post(
        "/v1/audio/transcriptions",
        content=b"\x00\x01" * 16_000,
        headers={"Content-Encoding": coding},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "content_type",
    ["application/json", "text/plain", "application/octet-stream", "audio/wav", None],
)
def test_the_content_types_clients_already_send_still_work(
    client: TestClient, content_type: str | None
) -> None:
    """An allowlist would be the shorter guard and would refuse every one of these,
    all of which reach the endpoint today."""
    headers = {} if content_type is None else {"Content-Type": content_type}
    response = client.post(
        "/v1/audio/transcriptions", content=b"\x00\x01" * 16_000, headers=headers
    )

    assert response.status_code == 200


def test_a_whole_wav_posted_as_the_body_is_still_accepted(client: TestClient) -> None:
    """Posting the container rather than bare samples is supported -- unwrap_wav reads
    it -- so the envelope guard must not mistake a container for an envelope."""
    header = (
        b"RIFF"
        + (36 + 32_000).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (16_000).to_bytes(4, "little")
        + (32_000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + (32_000).to_bytes(4, "little")
    )
    response = client.post(
        "/v1/audio/transcriptions",
        content=header + b"\x00\x01" * 16_000,
        headers={"Content-Type": "audio/wav"},
    )

    assert response.status_code == 200


def test_an_unhandled_failure_answers_json_like_every_other_error() -> None:
    """A client calling response.json() on the error path met Starlette's plain-text
    'Internal Server Error' and threw a parse error instead of showing the failure."""

    class Exploding(FakeEngine):
        async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
            raise RuntimeError("CUDA out of memory: tried to allocate 44.00 GiB")

    client = TestClient(create_app(Exploding()), raise_server_exceptions=False)
    response = client.post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000)

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "internal error"}
    # The engine's own words name internals the caller has no business reading.
    assert "CUDA" not in response.text


def _wav(rate: int, samples: int, *, channels: int = 1, bits: int = 16) -> bytes:
    """A container the way a recorder writes one."""
    width = channels * bits // 8
    data = bytes(samples * width)
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(data)),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * width, width, bits),
            b"data",
            struct.pack("<I", len(data)),
            data,
        ]
    )


@pytest.mark.parametrize("rate", [16_000, 24_000, 44_100, 48_000])
def test_the_reported_duration_is_the_recording_not_the_byte_count(
    client: TestClient, rate: int
) -> None:
    """The field divided the whole body -- container header included -- by the rate
    the endpoint assumes rather than the one the recording states. Ten seconds of
    24 kHz was reported as 15.00 and 48 kHz as 30.00, while the transcript was
    right, so nothing else in the response disagreed with it."""
    body = client.post("/v1/audio/transcriptions", content=_wav(rate, rate * 10)).json()

    assert body["duration_seconds"] == pytest.approx(10.0, abs=0.01)


def test_a_polisher_that_raises_does_not_take_the_transcript_with_it() -> None:
    """Dictation's second stage is optional and its first is not.

    Evaluated inside the response object, a polisher that raised became a 500 and
    the caller got no words at all -- for a failure in the step that only tidies
    words it already had.
    """

    class _Breaks(FakeEngine):
        async def polish(self, transcript: str) -> str:
            raise RuntimeError("polish checkpoint returned NaN")

    response = TestClient(create_app(_Breaks()), raise_server_exceptions=False).post(
        "/v1/audio/transcriptions", content=b"\x00\x01" * 16_000
    )

    assert response.status_code == 200
    assert response.json()["text"] == "今天天气怎么样"
    assert response.json()["polished"] is None
    # Null now means three things and only this one is a fault; a deployment whose
    # polisher broke on a bad deploy must not look like one that never had a
    # polisher, from every response it sends.
    assert response.json()["polish_failed"] is True


def test_a_working_polisher_does_not_claim_to_have_failed(client: TestClient) -> None:
    body = client.post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000).json()

    assert body["polish_failed"] is False


@pytest.mark.parametrize(("channels", "bits"), [(2, 16), (1, 8)], ids=["stereo", "8-bit"])
def test_a_wav_this_service_cannot_read_is_refused_rather_than_transcribed(
    client: TestClient, channels: int, bits: int
) -> None:
    """Handed on as raw PCM these transcribed at mean CER 0.029 and 0.852 over 12
    clips, both HTTP 200. See test_audio for the readings."""
    response = client.post(
        "/v1/audio/transcriptions",
        content=_wav(16_000, 8_000, channels=channels, bits=bits),
    )

    assert response.status_code == 400
    assert "16-bit mono" in response.json()["detail"]


def test_a_session_survives_audio_it_cannot_read_instead_of_dropping() -> None:
    """The realtime path has no status code to answer with, so an exception here
    closes the connection and the caller sees the model go silent."""
    from mindsurf_omni.service.audio import UnsupportedAudio

    class _Refuses(FakeEngine):
        async def respond(  # type: ignore[override]
            self,
            pcm: bytes,
            sample_rate: int,
            settings: GenerationSettings,
            history: list[dict[str, str]] | None = None,
        ) -> AsyncIterator[SpeechChunk]:
            raise UnsupportedAudio("this wav holds 2-channel 16-bit samples")
            yield  # pragma: no cover - makes this an async generator

    with TestClient(create_app(_Refuses())).websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        socket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"\x00" * 32_000).decode(),
            }
        )
        socket.send_json({"type": "input_audio_buffer.commit"})
        events = [socket.receive_json() for _ in range(3)]
        # Still open: the caller sends a different file rather than reconnecting.
        socket.send_json({"type": "session.clear"})
        events.append(socket.receive_json())

    assert any(event["type"] == "error" and "2-channel" in str(event) for event in events)
    assert events[-1]["type"] == "session.created"


def test_every_response_carries_an_id_that_finds_its_line_in_the_log(
    client: TestClient,
) -> None:
    """A user reporting "it returned nothing at 14:02" needs to hand over something
    narrower than a timestamp."""
    good = client.post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000)
    refused = client.post("/v1/audio/transcriptions", content=b"")

    assert len(good.headers["x-request-id"]) == 16
    assert len(refused.headers["x-request-id"]) == 16
    assert good.headers["x-request-id"] != refused.headers["x-request-id"]


def test_the_failure_path_carries_an_id_too() -> None:
    """The 500 is produced outside the middleware that sets the header, so it is
    the one response that could quietly lose it -- and the one that needs it."""

    class _Explodes(FakeEngine):
        async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
            raise RuntimeError("CUDA out of memory: tried to allocate 44.00 GiB")

    response = TestClient(create_app(_Explodes()), raise_server_exceptions=False).post(
        "/v1/audio/transcriptions", content=b"\x00\x01" * 16_000
    )

    assert response.status_code == 500
    assert len(response.headers["x-request-id"]) == 16


def test_the_access_line_says_what_happened_without_saying_what_was_said(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """A dictation service that logs transcripts keeps a copy of everything its
    users said, on a disk with a different retention policy from the product.
    Lengths answer "did it return nothing" without keeping the words."""
    with caplog.at_level(logging.INFO, logger="mindsurf.access"):
        client.post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000)

    line = json.loads(caplog.messages[-1])
    assert line["path"] == "/v1/audio/transcriptions"
    assert line["status"] == 200
    assert line["chars"] == len("今天天气怎么样")
    assert line["audio_seconds"] == pytest.approx(1.0)
    assert "asr_ms" in line
    assert "polish_ms" in line
    assert "今天天气怎么样" not in caplog.text


def test_the_counters_separate_a_refusal_from_a_success(client: TestClient) -> None:
    """"It is returning errors" and "it is returning errors to one caller" want
    different people woken up, so they are counted apart."""
    before = client.get("/stats").json()["counts"]
    client.post("/v1/audio/transcriptions", content=b"\x00\x01" * 16_000)
    client.post("/v1/audio/transcriptions", content=_wav(16_000, 8_000, channels=2))
    after = client.get("/stats").json()

    moved = {
        key: after["counts"][key] - before.get(key, 0)
        for key in after["counts"]
        if after["counts"][key] != before.get(key, 0)
    }
    assert moved["transcriptions"] == 1
    assert moved["audio refused"] == 1
    assert moved["requests /v1/audio/transcriptions 200"] == 1
    assert moved["requests /v1/audio/transcriptions 400"] == 1
    assert after["uptime_seconds"] >= 0


def test_the_counters_do_not_grow_a_key_per_path_a_scanner_tries(
    client: TestClient,
) -> None:
    """Keyed on the path, anything walking /a, /b, /c grows this without limit."""
    for path in ("/aaa", "/bbb", "/ccc"):
        client.get(path)

    keys = client.get("/stats").json()["counts"]

    assert not [key for key in keys if "/aaa" in key or "/bbb" in key]
    assert keys["requests unmatched 404"] >= 3
# --- resource limits: the request body, the realtime buffer, the spoken text ---


class _CountingStream:
    """A body that reports how much of itself was actually pulled."""

    def __init__(self, chunk: bytes, count: int) -> None:
        self.chunk, self.count, self.pulled = chunk, count, 0

    async def stream(self) -> AsyncIterator[bytes]:
        for _ in range(self.count):
            self.pulled += 1
            yield self.chunk


def test_an_oversized_body_is_refused_before_all_of_it_has_been_read() -> None:
    """`await request.body()` decided the 413 only after the whole thing landed:
    512 MB sent chunked cost 1036 MB of peak RSS and 11.4 s before the refusal."""
    from mindsurf_omni.service.app import LONGEST_BODY_BYTES, audio_body

    body = _CountingStream(b"\x00" * 2**20, LONGEST_BODY_BYTES // 2**20 * 4)

    with pytest.raises(TooLongForModel):
        asyncio.run(audio_body(body))

    assert body.pulled < body.count, "the whole body was read before it was refused"


def test_a_body_inside_the_limit_arrives_whole() -> None:
    from mindsurf_omni.service.app import audio_body

    body = _CountingStream(b"\x01\x40" * 1024, 8)

    assert asyncio.run(audio_body(body)) == b"\x01\x40" * 1024 * 8
    assert body.pulled == 8


def test_an_oversized_recording_is_413_over_http(client: TestClient) -> None:
    from mindsurf_omni.service.app import LONGEST_BODY_BYTES

    response = client.post("/v1/audio/transcriptions", content=b"\x00" * (LONGEST_BODY_BYTES + 2))

    assert response.status_code == 413


def test_a_realtime_turn_stops_growing_at_the_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """256 MiB appended with no commit was taken in full and counted nowhere."""
    from mindsurf_omni.service import app as app_module

    monkeypatch.setattr(app_module, "LONGEST_BUFFER_BYTES", 4096)
    audio = base64.b64encode(b"\x01\x40" * 1024).decode()

    with client.websocket_connect("/v1/realtime") as socket:
        socket.receive_json()
        for _ in range(2):
            socket.send_json({"type": "input_audio_buffer.append", "audio": audio})
        socket.send_json({"type": "input_audio_buffer.append", "audio": audio})
        refusal = socket.receive_json()
        # Refused, not disconnected, and what was already said is still there.
        socket.send_json({"type": "input_audio_buffer.commit"})
        assert socket.receive_json()["type"] == "response.created"

    assert refusal["type"] == "error"
    assert "seconds" in refusal["error"]["message"]


def test_text_longer_than_the_synthesiser_is_given_is_refused(client: TestClient) -> None:
    """Two million characters were accepted and handed straight on, HTTP 200."""
    from mindsurf_omni.contract import LONGEST_SPOKEN_CHARACTERS

    over = client.post("/v1/audio/speech", json={"input": "啊" * (LONGEST_SPOKEN_CHARACTERS + 1)})
    at_the_line = client.post("/v1/audio/speech", json={"input": "啊" * LONGEST_SPOKEN_CHARACTERS})

    assert over.status_code == 413
    assert at_the_line.status_code == 200


def test_the_refusal_is_not_the_size_of_the_thing_it_refused(client: TestClient) -> None:
    """`Field(max_length=...)` reads like a limit and behaves like a mirror.

    pydantic puts the rejected value into the ValidationError and FastAPI
    serialises the error whole, so the endpoint answers in proportion to what
    was thrown at it: two million characters came back as a 6 MB body, and
    512 MB posted chunked came back as 512 MB with peak memory up half a
    gigabyte. The check belongs in the route, where the answer can name two
    numbers and nothing else.
    """
    huge = "啊" * 200_000
    response = client.post("/v1/audio/speech", json={"input": huge})

    assert response.status_code == 413
    assert len(response.content) < 1_000, "被拒的正文本身成了放大器"
    assert huge not in response.text


def test_a_warm_up_that_failed_is_not_reported_ready() -> None:
    """/health answered 200 "ready" with an empty not_ready list at the same
    moment every transcription returned 500."""

    class Broken(FakeEngine):
        def warm(self) -> None:
            raise RuntimeError("checkpoint is truncated")

    with TestClient(create_app(Broken()), raise_server_exceptions=False) as started:
        report = started.get("/health").json()

    assert report["status"] != "ready"
    assert "warm-up" in report["not_ready"]
    assert "truncated" in dict((c["name"], c["detail"]) for c in report["components"])["warm-up"]


def test_a_warm_up_that_failed_once_stops_being_reported_once_it_works() -> None:
    """/health 报的是此刻，不是开机时的一段历史。

    `load()` 加载失败不缓存，所以开机那八秒卡被别的进程占了一下，第一个请求
    会重试并成功——机器其实完全是好的。而 warm_up_error 写一次之后没人清它，
    于是 /health 会永远说它坏了，也不会自己好。运维照 not_ready 摘轮换的话，
    这台好机器被永久摘掉，而且什么日志都不会响——比原来那个「坏了还说 ready」
    更难查，至少那个有 traceback。
    """
    from typing import Any

    class Flaky(FakeEngine):
        """warm 挂过一次，引擎本身好好的。"""

        def __init__(self) -> None:
            super().__init__()
            self.recogniser: Any = type("R", (), {"_model": None})()

        def warm(self) -> None:
            raise RuntimeError("CUDA out of memory: another process held the card")

    engine = Flaky()
    with TestClient(create_app(engine), raise_server_exceptions=False) as started:
        assert "warm-up" in started.get("/health").json()["not_ready"]

        # 第一个请求把权重加载上了（load() 不缓存失败）。
        engine.recogniser._model = object()

        report = started.get("/health").json()
        assert "warm-up" not in report["not_ready"], "好了之后还在说坏"
