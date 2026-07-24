"""The HTTP and WebSocket surface, built against the contract.

Runnable before the model is finished, deliberately: the backend and the client
are built in parallel with training, and they need something to point at now
rather than in a week. With no engine configured every endpoint answers 503
with a reason, which is a better contract than a fake success -- an integration
that appears to work against stubbed data fails on the day the real model
arrives.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse

from mindsurf_omni.contract import (
    AUDIO_ENCODING,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelList,
    SpeechRequest,
    TranscriptionResponse,
    VoiceInfo,
    VoiceList,
)
from mindsurf_omni.service.audio import frames, to_wav
from mindsurf_omni.service.config import ConfigurationError
from mindsurf_omni.service.engine import GenerationSettings, SpeechEngine

UNAVAILABLE = (
    "no speech engine is configured; set MINDSURF_ENGINE to 'native' or 'cascade' "
    "and point it at a checkpoint"
)


def create_app(engine: SpeechEngine | None = None) -> FastAPI:
    """Build the app, taking the engine from the environment when not given.

    The container runs this with no arguments, so the environment is what
    decides. A configuration error here is reported by the endpoints rather
    than raised at startup: a container that crash-loops gives an operator no
    log to read and no health check to consult.
    """
    app = FastAPI(title="MindSurf Omni", version="0.1.0")

    if engine is None:
        from mindsurf_omni.service.config import Settings
        from mindsurf_omni.service.factory import build

        try:
            engine = build(Settings.from_environment())
        except ConfigurationError as error:
            app.state.configuration_error = str(error)

    app.state.engine = engine

    @app.exception_handler(ConfigurationError)
    async def unconfigured(request: Request, error: Exception) -> JSONResponse:
        """A stage that is not wired answers 503 with its reason, not a traceback.

        The same shape the service gives when no engine exists at all, so a
        caller integrating against a half-built service handles one failure
        rather than two -- and the reason names the stage rather than arriving
        as a 500 someone has to read a log to explain.
        """
        return JSONResponse({"detail": str(error)}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    def require_engine(request: Request) -> SpeechEngine:
        configured: SpeechEngine | None = getattr(request.app.state, "engine", None)
        if configured is None:
            # A configuration error is more useful than the generic message:
            # it names the variable or the file that was missing.
            detail = getattr(request.app.state, "configuration_error", None) or UNAVAILABLE
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)
        return configured

    @app.get("/v1/models", response_model=ModelList)
    async def list_models(request: Request) -> ModelList:
        """What is actually loaded, including whether it may be used commercially.

        The licence rides in the response rather than only in the README,
        because a restriction nobody sees is a restriction nobody honours.
        """
        configured: SpeechEngine | None = getattr(request.app.state, "engine", None)
        if configured is None:
            return ModelList(data=[])
        description = configured.describe()
        return ModelList(
            data=[
                ModelInfo(
                    id="mindsurf-omni",
                    created=int(time.time()),
                    path=description.path,
                    fallback_available=True,
                    components=description.components,
                    licence=description.licence,
                    commercial_use_permitted=description.commercial_use_permitted,
                )
            ]
        )

    @app.get("/health")
    async def health(request: Request) -> Any:
        """Whether this instance can serve, and which part cannot.

        Answers 200 while degraded, because a service still able to reply over
        the fallback should stay in rotation; 503 only when nothing can serve.
        """
        from fastapi.responses import JSONResponse

        from mindsurf_omni.service.health import assess as assess_health

        report = assess_health(
            getattr(request.app.state, "engine", None),
            getattr(request.app.state, "configuration_error", None),
        )
        code = status.HTTP_503_SERVICE_UNAVAILABLE if report.status == "unavailable" else 200
        return JSONResponse(report.to_dict(), status_code=code)

    @app.get("/v1/licence")
    async def licence() -> dict[str, Any]:
        """The whole chain, including the parts nobody has read.

        /v1/models states the conclusion; this is what it rests on. A caller
        deciding whether they may ship something needs to see that four of the
        six assets have unread terms, not just that the answer is no.
        """
        from pathlib import Path as _Path

        record = _Path(__file__).resolve().parents[3] / "configs" / "release" / "licence.json"
        if not record.is_file():
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "the licence record is not in this image; it lives at "
                "configs/release/licence.json in the repository",
            )
        return json.loads(record.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    @app.get("/v1/token-spec")
    async def token_spec(request: Request) -> dict[str, Any]:
        """Served from the running model so it cannot drift from the weights."""
        return require_engine(request).token_spec().model_dump()

    @app.get("/v1/voices", response_model=VoiceList)
    async def list_voices(request: Request) -> VoiceList:
        require_engine(request)
        return VoiceList(
            data=[
                VoiceInfo(
                    id="default",
                    description="发布权重的默认音色",
                    speaker_embedding_dim=192,
                )
            ]
        )

    @app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
    async def transcribe(request: Request) -> TranscriptionResponse:
        engine = require_engine(request)
        pcm = await request.body()
        if not pcm:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "request body carried no audio")
        text, language = await engine.transcribe(pcm, INPUT_SAMPLE_RATE)
        return TranscriptionResponse(
            text=text,
            language=language,
            duration_seconds=len(pcm) / 2 / INPUT_SAMPLE_RATE,
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request, body: ChatCompletionRequest) -> Any:
        engine = require_engine(request)
        settings = GenerationSettings(
            temperature=body.temperature, top_p=body.top_p, max_tokens=body.max_tokens
        )
        messages = [message.model_dump() for message in body.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not body.stream:
            text = "".join([chunk async for chunk in engine.complete(messages, settings)])
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=body.model,
                choices=[
                    ChatChoice(
                        message={"role": "assistant", "content": text},  # type: ignore[arg-type]
                        finish_reason="stop",
                    )
                ],
            )

        async def stream() -> Any:
            async for delta in engine.complete(messages, settings):
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {"content": delta}}],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/audio/speech")
    async def speech(request: Request, body: SpeechRequest) -> StreamingResponse:
        engine = require_engine(request)
        settings = GenerationSettings(voice=body.voice, emotion=body.emotion)
        as_wav = body.response_format == "wav"

        async def stream() -> Any:
            # wav is buffered so the header can carry a real length; pcm
            # streams, which is what a caller wanting bytes as they arrive
            # should ask for.
            #
            # A streaming wav would have to write a placeholder size, and a
            # sister project on this team already paid for that: their upstream
            # sent 0xFFFFFFFF chunk sizes, an Android client read them as
            # signed, computed a negative offset and crashed. ffmpeg tolerates
            # the placeholder, a hand-written parser does not, and the clients
            # here include one. Without any header at all the body is bare PCM
            # labelled audio/wav, which every decoder refuses -- and the
            # evaluation harness writes this response straight to a .wav, so
            # the whole chain came back with empty transcripts.
            if as_wav:
                spoken = bytearray()
                async for chunk in engine.speak(body.input, settings):
                    spoken += chunk.pcm
                yield to_wav(bytes(spoken), OUTPUT_SAMPLE_RATE)
                return
            async for chunk in engine.speak(body.input, settings):
                if chunk.pcm:
                    yield chunk.pcm

        return StreamingResponse(
            stream(),
            media_type="audio/wav" if as_wav else "application/octet-stream",
            headers={"X-Sample-Rate": str(OUTPUT_SAMPLE_RATE), "X-Encoding": AUDIO_ENCODING},
        )

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        engine: SpeechEngine | None = getattr(websocket.app.state, "engine", None)
        if engine is None:
            await websocket.send_json({"type": "error", "error": {"message": UNAVAILABLE}})
            await websocket.close()
            return

        await websocket.send_json(
            {
                "type": "session.created",
                "input_sample_rate": INPUT_SAMPLE_RATE,
                "output_sample_rate": OUTPUT_SAMPLE_RATE,
                "encoding": AUDIO_ENCODING,
            }
        )
        buffer = bytearray()
        settings = GenerationSettings()
        # History is trimmed as it grows: audio tokens exhaust a small model's
        # context within a few turns, and the caller is told when turns were
        # dropped rather than left to wonder why the model forgot.
        from mindsurf_omni.service.session import Conversation, Turn

        conversation = Conversation()

        try:
            while True:
                event = await websocket.receive_json()
                kind = event.get("type")

                if kind == "input_audio_buffer.append":
                    buffer.extend(base64.b64decode(event.get("audio", "")))
                elif kind == "input_audio_buffer.clear":
                    buffer.clear()
                elif kind == "session.update":
                    settings.voice = event.get("voice", settings.voice)
                    settings.emotion = event.get("emotion", settings.emotion)
                elif kind == "session.clear":
                    conversation.clear()
                    await websocket.send_json(
                        {"type": "session.created", "context": conversation.summary()}
                    )
                elif kind == "response.cancel":
                    # Barge-in. Generation is driven from this loop, so there is
                    # nothing running to interrupt between events; the buffer is
                    # dropped so the cancelled turn cannot leak into the next.
                    buffer.clear()
                    await websocket.send_json({"type": "response.done", "cancelled": True})
                elif kind == "input_audio_buffer.commit":
                    if not buffer:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": "no audio was buffered"}}
                        )
                        continue
                    await websocket.send_json({"type": "response.created"})
                    spoken_seconds = len(buffer) / 2 / INPUT_SAMPLE_RATE
                    reply = ""
                    async for chunk in engine.respond(bytes(buffer), INPUT_SAMPLE_RATE, settings):
                        if chunk.text:
                            reply += chunk.text
                            await websocket.send_json(
                                {"type": "response.text.delta", "delta": chunk.text}
                            )
                        # Split rather than sent whole: a clause of speech can
                        # exceed the peer's frame limit, and the peer's answer
                        # to that is to close the connection, which the caller
                        # sees as the reply stopping rather than as a fault.
                        for frame in frames(chunk.pcm):
                            await websocket.send_json(
                                {
                                    "type": "response.audio.delta",
                                    "audio": base64.b64encode(frame).decode("ascii"),
                                }
                            )
                    buffer.clear()
                    conversation.append(Turn(role="user", text="", audio_seconds=spoken_seconds))
                    conversation.append(Turn(role="assistant", text=reply))
                    await websocket.send_json({"type": "response.audio.done"})
                    await websocket.send_json(
                        {"type": "response.done", "context": conversation.summary()}
                    )
                else:
                    # Answered rather than ignored: a silent no-op leaves the
                    # client waiting for a reply that will never come.
                    await websocket.send_json(
                        {"type": "error", "error": {"message": f"unknown event type: {kind!r}"}}
                    )
        except WebSocketDisconnect:
            return

    return app
