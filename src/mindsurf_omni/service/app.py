"""The HTTP and WebSocket surface, built against the contract.

Runnable before the model is finished, deliberately: the backend and the client
are built in parallel with training, and they need something to point at now
rather than in a week. With no engine configured every endpoint answers 503
with a reason, which is a better contract than a fake success -- an integration
that appears to work against stubbed data fails on the day the real model
arrives.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, StreamingResponse

from mindsurf_omni.contract import (
    AUDIO_ENCODING,
    EMOTIONS,
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
from mindsurf_omni.service.engine import GenerationSettings, SpeechEngine, TooLongForModel
from mindsurf_omni.service.tts import SynthesiserUnavailable


class Unreadable(Exception):
    """A frame from the client that could not be parsed.

    Its own type because the realtime loop has to answer it and carry on. The
    three ways it happens -- a binary frame, a text frame that is not JSON, and
    audio that is not valid base64 -- used to escape as KeyError,
    JSONDecodeError and binascii.Error, none of which anything caught: the
    connection died at 1006 with no close frame and no reason, and the caller
    lost the whole conversation over one malformed chunk.
    """


async def receive_event(websocket: WebSocket) -> dict[str, Any]:
    """One client event, or Unreadable saying why it is not one.

    WebSocketDisconnect still travels, because a closed connection is not a bad
    frame and the loop above ends on it.
    """
    try:
        event = await websocket.receive_json()
    except KeyError as error:  # a binary frame; starlette reads text
        raise Unreadable("this endpoint takes JSON text frames; that one carried binary") from error
    except json.JSONDecodeError as error:
        raise Unreadable(f"the frame is not JSON: {error}") from error
    if not isinstance(event, dict):
        raise Unreadable(f"an event has to be a JSON object, not {type(event).__name__}")
    return event


def event_audio(event: dict[str, Any]) -> bytes:
    """The audio an append event carries, or Unreadable saying it is not base64.

    TypeError as well as binascii.Error: the field is whatever JSON carried, so
    a client that puts a number there reaches this with an int.
    """
    try:
        return base64.b64decode(event.get("audio", ""), validate=True)
    except (binascii.Error, TypeError) as error:
        raise Unreadable(f"the audio field is not base64: {error}") from error


async def first_and_rest(source: Any) -> tuple[Any, Any]:
    """Pull one item out before the response starts, and hand back the rest.

    An ``async def`` generator runs no code until it is first iterated, so a
    stage that refuses -- "no synthesiser is wired" -- raises inside the
    StreamingResponse, after the headers are gone. The caller then reads zero
    bytes off a 200 and sees ``IncompleteRead``, which names nothing it could
    act on. Measured against a service with no synthesiser and no thinker: the
    non-streaming chat path answered 503 with the environment variables to set,
    while streaming chat, wav speech and pcm speech all returned the empty
    stream.

    Taking the first item here moves the refusal back in front of the headers,
    where the ConfigurationError handler can turn it into that same 503.
    ``None`` for the first item means the source was empty, which is not an
    error -- silence recognises as no audio and no text.
    """
    iterator = source.__aiter__()
    try:
        head = await iterator.__anext__()
    except StopAsyncIteration:
        return None, iterator
    return head, iterator


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

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Pull the recogniser's weights in while nobody is waiting on them.

        The recogniser loads lazily so that a container which cannot reach its
        weights still starts and explains itself, and that is worth keeping.
        But deferring the load is not the same as deferring it *into a request*
        -- measured on the machine that hosts the model, SenseVoice takes
        7272 ms to load, and every one of those milliseconds lands on the first
        user to press record. Warming here keeps the behaviour on a missing
        checkpoint exactly as it was (the failure still surfaces at the first
        request) and takes the wait off the person.

        Failure is swallowed on purpose: this is a warm-up, and a service that
        refuses to start because a warm-up failed is the crash-loop the lazy
        load exists to avoid.
        """
        warm_up = getattr(app.state.engine, "warm", None)
        if warm_up is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(warm_up)
        yield

    app = FastAPI(title="MindSurf Omni", version="0.1.0", lifespan=lifespan)

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

    @app.exception_handler(TooLongForModel)
    async def too_long(request: Request, error: Exception) -> JSONResponse:
        """413, not 503: retrying the same request cannot help.

        A caller that reads 503 backs off and sends it again. This one has to
        change the request, and the reason says how.
        """
        return JSONResponse(
            {"detail": str(error)}, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        )

    @app.exception_handler(SynthesiserUnavailable)
    async def no_audio(request: Request, error: Exception) -> JSONResponse:
        """502: the thing this service depends on did not answer.

        A 500 says the fault is here and points an operator at our logs. The
        hosted synthesiser returns no audio for about 2 turns in 160 on a
        healthy network, and every one of those was an Internal Server Error
        with an empty body -- indistinguishable from a bug in this code.
        """
        return JSONResponse({"detail": str(error)}, status_code=status.HTTP_502_BAD_GATEWAY)

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
        polish = getattr(engine, "polish", None)
        return TranscriptionResponse(
            text=text,
            polished=await polish(text) if polish is not None else None,
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

        # Before the headers, so a stage that refuses still answers 503 rather
        # than an empty 200 the caller reads as IncompleteRead.
        head, rest = await first_and_rest(engine.complete(messages, settings))

        async def stream() -> Any:
            async def deltas() -> Any:
                # The first item was taken before the headers so a refusal
                # could still become a 503; it has to be put back in front.
                if head is not None:
                    yield head
                async for item in rest:
                    yield item

            async for delta in deltas():
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
        # Both of these are in the request shape an OpenAI client already
        # sends, and neither reaches a synthesiser: `speed` stops at this
        # function, and the cascade's synthesisers take their voice from
        # configuration rather than from the request. Measured by asking for
        # the same sentence at 0.5, 1.0 and 2.0 and at three voice ids -- all
        # five responses were byte-identical, while the emotions that are
        # wired differ. Saying so beats returning audio that looks like it
        # honoured the request: a caller cannot hear the difference between
        # "spoken at 1.5x" and "your parameter was dropped".
        if body.speed != 1.0:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"speed={body.speed} is not wired; this build speaks at one rate and would "
                "return the same audio as speed=1.0. Use `emotion`, which does change "
                "delivery, or resample the pcm response",
            )
        if body.voice != "default":
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"voice={body.voice!r} is not a voice this build has; /v1/voices lists the "
                "ones it does, and today that is 'default' alone",
            )
        settings = GenerationSettings(voice=body.voice, emotion=body.emotion)
        as_wav = body.response_format == "wav"
        # Started here rather than inside the generator: an engine that cannot
        # speak arbitrary text -- the native path, whose model only says its own
        # words -- refuses on the call, and refusing inside a StreamingResponse
        # happens after the headers are out. The caller then sees a truncated
        # body instead of the 503 that names the reason.
        speaking = engine.speak(body.input, settings)
        # Before the headers, so a stage that refuses still answers 503.
        head, rest = await first_and_rest(speaking)

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
                spoken = bytearray(head.pcm if head is not None else b"")
                async for chunk in rest:
                    spoken += chunk.pcm
                yield to_wav(bytes(spoken), OUTPUT_SAMPLE_RATE)
                return
            if head is not None and head.pcm:
                yield head.pcm
            async for chunk in rest:
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

        # The cascade sends transcripts, so its history costs text; the native
        # path sends the audio itself and pays Mimi's 12.5 tokens a second.
        conversation = Conversation(counts_audio=engine.describe().path == "native")

        # An event heard while a response was streaming but not consumed there.
        # Cancelling the listener races its completion, and a message that
        # arrived in that gap must be handled, not dropped.
        pending: dict[str, Any] | None = None

        try:
            while True:
                if pending is not None:
                    event, pending = pending, None
                else:
                    try:
                        event = await receive_event(websocket)
                    except Unreadable as unreadable:
                        # Answered and survived, like an unknown event type. A
                        # client whose encoder produced one bad frame should
                        # lose that frame, not the conversation.
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(unreadable)}}
                        )
                        continue
                kind = event.get("type")

                if kind == "input_audio_buffer.append":
                    try:
                        buffer.extend(event_audio(event))
                    except Unreadable as unreadable:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(unreadable)}}
                        )
                elif kind == "input_audio_buffer.clear":
                    buffer.clear()
                elif kind == "session.update":
                    # Validated, and answered either way. This branch used to
                    # assign whatever arrived: emotion "angry" was accepted in
                    # silence and delivered as neutral, voice "nobody" was
                    # accepted in silence and delivered as the only voice there
                    # is. The speech endpoint refuses both -- this is the same
                    # request through the other door, and a caller that cannot
                    # hear the difference between "applied" and "dropped" has
                    # no way to find out except by asking someone to listen.
                    wanted_emotion = event.get("emotion", settings.emotion)
                    wanted_voice = event.get("voice", settings.voice)
                    if wanted_emotion not in EMOTIONS:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "error": {
                                    "message": f"emotion {wanted_emotion!r} is not one this "
                                    f"build delivers; it has {', '.join(EMOTIONS)}"
                                },
                            }
                        )
                    elif wanted_voice != "default":
                        await websocket.send_json(
                            {
                                "type": "error",
                                "error": {
                                    "message": f"voice {wanted_voice!r} is not a voice this "
                                    "build has; /v1/voices lists the ones it does, and "
                                    "today that is 'default' alone"
                                },
                            }
                        )
                    else:
                        settings.voice, settings.emotion = wanted_voice, wanted_emotion
                        # Acknowledged, because a silent success and a silent
                        # failure look identical from the other end.
                        await websocket.send_json(
                            {
                                "type": "session.updated",
                                "voice": settings.voice,
                                "emotion": settings.emotion,
                            }
                        )
                elif kind == "session.clear":
                    conversation.clear()
                    # The same shape as the connection's own session.created,
                    # sample rates included: a client handles this event in one
                    # place, and a second version missing half the fields is how
                    # the playback rate ends up undefined after a clear.
                    await websocket.send_json(
                        {
                            "type": "session.created",
                            "input_sample_rate": INPUT_SAMPLE_RATE,
                            "output_sample_rate": OUTPUT_SAMPLE_RATE,
                            "encoding": AUDIO_ENCODING,
                            "context": conversation.summary(),
                        }
                    )
                elif kind == "response.cancel":
                    # Cancel with no response streaming: a turn that never
                    # started. The buffer is dropped so it cannot leak into the
                    # next turn. (Cancel DURING streaming is handled inside the
                    # commit branch, and keeps the buffer -- see there for why.)
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
                    turn_pcm = bytes(buffer)
                    # Cleared before streaming, not after: anything appended
                    # while the reply is playing is the user's interjection,
                    # and it belongs to the next turn rather than to this one.
                    buffer.clear()
                    parts: list[str] = []
                    # Passed in rather than closed over, like `texts` below:
                    # this runs in a task, and the loop that owns the name has
                    # to read what the task appended.
                    said: list[str] = []

                    async def stream(pcm: bytes, texts: list[str], heard: list[str]) -> None:
                        async for chunk in engine.respond(
                            pcm, INPUT_SAMPLE_RATE, settings, history=conversation.messages()
                        ):
                            # Truthy, not "is not None": silence recognises as
                            # an empty string, and an empty transcript event is
                            # a blank bubble in the caller's transcript view.
                            if chunk.transcript:
                                heard.append(chunk.transcript)
                                # Spelled out rather than named: the contract
                                # test looks for the literal where it is sent,
                                # and a constant would answer for a branch that
                                # had been deleted. Too long for the line limit
                                # at this depth, and moving it is the bug.
                                await websocket.send_json(
                                    {
                                        "type": "conversation.item.input_audio_transcription.completed",  # noqa: E501
                                        "transcript": chunk.transcript,
                                    }
                                )
                            if chunk.text:
                                texts.append(chunk.text)
                                await websocket.send_json(
                                    {"type": "response.text.delta", "delta": chunk.text}
                                )
                            # Split rather than sent whole: a clause of speech
                            # can exceed the peer's frame limit, and the peer's
                            # answer to that is to close the connection, which
                            # the caller sees as the reply stopping rather than
                            # as a fault.
                            for frame in frames(chunk.pcm):
                                await websocket.send_json(
                                    {
                                        "type": "response.audio.delta",
                                        "audio": base64.b64encode(frame).decode("ascii"),
                                    }
                                )

                    # Generation runs as a task so this loop can keep hearing
                    # the socket. The previous version drove the async-for
                    # right here, which meant a response.cancel sent mid-reply
                    # sat unread until the whole turn had streamed -- barge-in
                    # could only cancel turns that had not started. Cancelling
                    # the task abandons the generator, and the engine's
                    # producer watches for exactly that (the leak fix), so the
                    # GPU stops with it.
                    speaking = asyncio.create_task(stream(turn_pcm, parts, said))
                    cancelled = False
                    while not speaking.done():
                        listener = asyncio.create_task(receive_event(websocket))
                        await asyncio.wait(
                            {speaking, listener}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if not listener.done():
                            listener.cancel()
                            try:
                                pending = await listener
                            except asyncio.CancelledError:
                                pass
                            except Unreadable as unreadable:
                                # A bad frame that landed in the race gap. Same
                                # answer as below; it must not be left in
                                # `pending` for the outer loop to re-raise.
                                await websocket.send_json(
                                    {"type": "error", "error": {"message": str(unreadable)}}
                                )
                            except WebSocketDisconnect:
                                # Completed with a disconnect in the race gap.
                                speaking.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await speaking
                                raise
                            continue
                        try:
                            heard = listener.result()
                        except Unreadable as unreadable:
                            # Handled here rather than by the outer loop: the
                            # reply is still streaming, and leaving this frame
                            # to unwind out of the loop would abandon the task
                            # mid-turn, with no response.done and no history.
                            await websocket.send_json(
                                {"type": "error", "error": {"message": str(unreadable)}}
                            )
                            continue
                        except WebSocketDisconnect:
                            speaking.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await speaking
                            raise
                        told = heard.get("type")
                        if told == "response.cancel":
                            # Barge-in. The buffer is KEPT, unlike the idle
                            # cancel above: audio appended during the reply is
                            # the interjection that caused this cancel, and
                            # dropping it would eat the user's next words.
                            speaking.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await speaking
                            cancelled = True
                        elif told == "input_audio_buffer.append":
                            try:
                                buffer.extend(event_audio(heard))
                            except Unreadable as unreadable:
                                await websocket.send_json(
                                    {"type": "error", "error": {"message": str(unreadable)}}
                                )
                        elif told == "input_audio_buffer.clear":
                            buffer.clear()
                        else:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "error": {
                                        "message": "a response is streaming; only "
                                        "response.cancel and buffer events are "
                                        f"accepted now, not {told!r}"
                                    },
                                }
                            )
                    if not cancelled and (failure := speaking.exception()) is not None:
                        if isinstance(failure, TooLongForModel):
                            # A dictated turn longer than the model answers.
                            # Reported like any other event the session can
                            # survive: the caller says something shorter and
                            # keeps the connection, rather than being
                            # disconnected by an exception it never sees.
                            await websocket.send_json(
                                {"type": "error", "error": {"message": str(failure)}}
                            )
                            await websocket.send_json(
                                {"type": "response.done", "context": conversation.summary()}
                            )
                            continue
                        raise failure

                    if not cancelled and not said and not parts:
                        # The sibling of "no audio was buffered" above: audio
                        # arrived, the recogniser found no speech in it, and a
                        # bare response.done with no text cannot tell a caller
                        # whether nothing was heard or the model said nothing.
                        await websocket.send_json(
                            {
                                "type": "error",
                                "error": {"message": "no speech was heard in the committed audio"},
                            }
                        )

                    reply = "".join(parts)
                    # The transcript when the path produced one, so the next
                    # turn's prompt carries what the user actually said. The
                    # native path leaves it empty: there is no text to keep,
                    # and the turn is recorded for the token budget alone.
                    conversation.append(
                        Turn(
                            role="user",
                            text=said[0] if said else "",
                            audio_seconds=spoken_seconds,
                        )
                    )
                    # The partial reply enters history as what was actually
                    # said: the model spoke those words before it was cut off,
                    # and the next turn's context should not pretend otherwise.
                    conversation.append(Turn(role="assistant", text=reply))
                    if cancelled:
                        await websocket.send_json(
                            {
                                "type": "response.done",
                                "cancelled": True,
                                "context": conversation.summary(),
                            }
                        )
                    else:
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
