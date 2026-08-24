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
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from mindsurf_omni.contract import (
    AUDIO_ENCODING,
    EMOTIONS,
    INPUT_SAMPLE_RATE,
    LONGEST_SPOKEN_CHARACTERS,
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
from mindsurf_omni.service.asr import LONGEST_SECONDS
from mindsurf_omni.service.audio import UnsupportedAudio, frames, to_wav, unwrap_wav, whole_samples
from mindsurf_omni.service.config import ConfigurationError
from mindsurf_omni.service.engine import GenerationSettings, SpeechEngine, TooLongForModel
from mindsurf_omni.service.health import count, counters
from mindsurf_omni.service.tts import SynthesiserUnavailable

# One JSON object per request, on its own line. Not a metrics stack: everything
# an operator needs for the question "which requests went wrong and what did
# they look like" fits on a line, and a line is greppable from inside the
# container without deploying anything to read it.
#
# The transcript never appears here. A dictation service's logs would otherwise
# be a copy of everything its users said, sitting on a disk with a different
# retention policy from the product. Lengths go in instead: they answer "did it
# return nothing" without keeping the words.
_log = logging.getLogger("mindsurf.access")


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


def buffer_audio(buffer: bytearray, event: dict[str, Any]) -> None:
    """Add this event's audio to the turn, or refuse it and say why.

    A turn past LONGEST_SECONDS cannot be committed -- ``transcribe`` raises on
    it -- so storing the bytes only costs memory nobody can spend. Refusing the
    append rather than clearing the buffer is deliberate: what has already been
    said is still committable, and the barge-in path keeps its buffer on
    purpose (see the cancel branch), which a clear-on-overflow would undo.
    """
    chunk = event_audio(event)
    if len(buffer) + len(chunk) > LONGEST_BUFFER_BYTES:
        raise TooLongForModel(
            f"this turn already holds {len(buffer) / 2 / INPUT_SAMPLE_RATE:.0f} seconds of "
            f"audio and the recogniser reads up to {LONGEST_SECONDS:.0f} in one turn; "
            "commit what is buffered or clear it"
        )
    buffer.extend(chunk)


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

# The most audio one realtime turn may buffer before it is committed. Raw
# PCM16 at the rate session.created announces, so no container multiplies it
# the way it can over HTTP: this is exactly LONGEST_SECONDS of audio. Past it
# the commit could only ever raise TooLongForModel, so the bytes are refused
# rather than stored -- an append-only session held 256 MiB with no commit and
# nothing in the service counted it.
LONGEST_BUFFER_BYTES = int(LONGEST_SECONDS) * INPUT_SAMPLE_RATE * 2

# The most audio a single request may carry, counted in bytes as they arrive.
#
# The transport backstop, not the duration rule -- LONGEST_SECONDS still decides
# that, after the container has been unwrapped and resampled. Four times the raw
# hour, because an hour of 16 kHz PCM16 is 115 MB and the same hour sent as a
# 48 kHz wav is 346 MB, and the documented upload format is the first but the
# second reaches the endpoint today.
#
# What four times cost was measured after it was chosen: a 436 MiB body that
# needs resampling peaked at 7554 MB, eighteen times its own size, so this
# number was in effect an 8.3 GB memory limit. That came from ``np.interp``
# and is fixed in ``audio.resample``; the same bodies now peak below their own
# size. Anyone moving this line should measure the peak again rather than the
# bytes.
LONGEST_BODY_BYTES = 4 * int(LONGEST_SECONDS) * INPUT_SAMPLE_RATE * 2


async def audio_body(request: Request) -> bytes:
    """The request body, refused as it arrives rather than after it has landed.

    ``await request.body()`` reads the whole thing into memory and only then
    lets anything look at it, so the 413 for an oversized recording was paid
    for in full first: measured against a real uvicorn, 256 MB cost 523 MB of
    peak RSS before the refusal, and 512 MB sent chunked cost 1036 MB and
    11.4 seconds. Chunked is the case that matters -- there is no
    Content-Length to consult, so nothing but counting can know.
    """
    chunks: list[bytes] = []
    total = 0
    try:
        async for piece in request.stream():
            total += len(piece)
            if total > LONGEST_BODY_BYTES:
                raise TooLongForModel(
                    f"the request body is over {LONGEST_BODY_BYTES} bytes, which is more audio "
                    f"than the {LONGEST_SECONDS:.0f} seconds this endpoint transcribes in one "
                    "request; send it in pieces"
                )
            chunks.append(piece)
    except ClientDisconnect:
        # Somebody hung up mid-upload. Not a server fault, and answered as one
        # it cost 11 kB of traceback per abort -- 33 kB for three, and a process
        # whose stderr goes to a pipe nobody drains stops at 64. There is nobody
        # left to answer, so the shortest legal thing goes out; 499 is what a
        # closed upload is called even though no RFC says so.
        raise HTTPException(499, "the caller closed the connection mid-upload") from None
    return b"".join(chunks)


# Anything that looks like a filesystem path: a Windows drive letter or a POSIX
# absolute path, up to the next whitespace or quote.
_A_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/)[^\s'\"),;]*")


def without_paths(message: str) -> str:
    """A message safe to hand an unauthenticated caller.

    Absolute paths name the account and the directory layout of the machine
    serving the request, and a deployment whose weights sit under a customer's
    name was handing that name to anyone who could reach the port. The variable
    that was wrong is what the caller can act on; the path is what the operator
    needs, and it goes to the log.
    """
    return _A_PATH.sub("<path>", message)


def refuse_envelope(request: Request) -> None:
    """Refuse a body that is wrapped in something this route does not unwrap.

    The route reads the whole body as audio, so anything around the audio is
    read as audio too, and a recogniser handed noise does not fail -- it writes
    something. Both of these were HTTP 200:

    * The standard OpenAI client call for this path is a multipart form.
      Measured, the recogniser received 181 bytes more than were sent, and 181
      is odd, so every 16-bit sample after the boundary was shifted a byte and
      the recording became noise.
    * ``Content-Encoding: gzip`` -- nothing on this path decompresses, so the
      deflate stream went to the recogniser as samples.

    Named rather than allowlisted, on purpose. Callers reach this endpoint
    today with application/json, text/plain, application/octet-stream,
    audio/wav and with no Content-Type at all; an allowlist is the shorter
    guard and would refuse all five.
    """
    # Joined, not ``get``: a header sent on two lines is the same message as
    # one line with a comma (RFC 9110 section 5.2), and Starlette's ``get``
    # returns only the first of them. This gate already refused the comma form,
    # so reading one line let the same request through in two -- measured on a
    # raw socket, ``identity`` then ``gzip`` was a 200. Case-insensitive for the
    # same reason: a startswith against the raw value lets it through in capitals.
    encoding = ",".join(request.headers.getlist("content-encoding")).strip().lower()
    if encoding and encoding != "identity":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"the body arrived under Content-Encoding: {encoding}, and nothing on this "
            "path decompresses it -- the compressed bytes would be read as samples. "
            "Send the audio uncompressed",
        )
    content_type = ",".join(request.headers.getlist("content-type")).strip().lower()
    if "multipart/" in content_type:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "this endpoint takes the audio as the entire request body, not as a "
            "multipart form field -- the MIME boundary would be read as samples. Post "
            "the bytes directly: PCM16 mono little-endian, or a wav container",
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

        Failure is still swallowed: this is a warm-up, and a service that
        refuses to start because a warm-up failed is the crash-loop the lazy
        load exists to avoid. What it no longer does is stay quiet about it.
        Suppressed outright, the only component that actually loads weights
        could fail and /health would answer 200 "ready" with an empty
        not_ready list while every transcription returned 500 -- measured.
        """
        app.state.warm_up_error = None
        warm_up = getattr(app.state.engine, "warm", None)
        if warm_up is not None:
            try:
                await asyncio.to_thread(warm_up)
            except Exception as error:  # noqa: BLE001 -- reported, not raised
                app.state.warm_up_error = f"{type(error).__name__}: {error}"
        yield

    app = FastAPI(title="MindSurf Omni", version="0.1.0", lifespan=lifespan)

    if engine is None:
        from mindsurf_omni.service.config import Settings
        from mindsurf_omni.service.factory import build

        try:
            engine = build(Settings.from_environment())
        except ConfigurationError as error:
            # Scrubbed at the join rather than at each raise: six other places
            # under build() put a Path into a ConfigurationError, and this
            # string reaches /health and /v1/voices, neither of which needs
            # authentication. A deployment whose weights live under a customer's
            # name was handing that name to anyone who could reach the port.
            app.state.configuration_error = without_paths(str(error))
            logging.getLogger("mindsurf").error("configuration: %s", error)

    app.state.engine = engine

    # Nothing prints INFO by default, and uvicorn configures its own loggers
    # rather than the root one, so without this the access line is written and
    # discarded. Attached to "mindsurf" rather than to the root logger on
    # purpose: basicConfig would also turn on INFO for httpx, funasr and torch,
    # and the access line would arrive buried in them. Skipped entirely when
    # anything else is already configured, so a real logging setup wins.
    _service_log = logging.getLogger("mindsurf")
    if not _service_log.handlers and not logging.getLogger().handlers:
        _service_log.setLevel(logging.INFO)
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(message)s"))
        _service_log.addHandler(_handler)

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Give every request an id, count it, and write one line about it.

        The id goes out on the response header so that a user reporting "it
        returned nothing at 14:02" hands over something that finds the line.
        Generated here rather than read from the request: there is no proxy in
        front of this today, and taking an id from the caller means sanitising
        a value that ends up in a log and a header.

        The endpoint hangs its own numbers on ``request.state.observed``;
        Starlette backs that with the ASGI scope, so what the endpoint sets is
        what this reads. For a streaming response the duration is time to the
        headers, not to the last byte -- ``call_next`` returns at the first
        message.
        """
        request_id = uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        code = 500
        try:
            response = await call_next(request)
            code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            # Bounded on purpose: keyed on the matched route rather than the
            # path, so a scanner walking /a, /b, /c cannot grow this without
            # limit. Unmatched paths all land in one bucket.
            route = getattr(request.scope.get("route"), "path", "unmatched")
            count(f"requests {route} {code}")
            _log.info(
                json.dumps(
                    {
                        "event": "request",
                        "id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": code,
                        "ms": round((time.perf_counter() - started) * 1000, 1),
                        **getattr(request.state, "observed", {}),
                    },
                    ensure_ascii=False,
                )
            )

    @app.get("/stats")
    async def stats() -> dict[str, object]:
        """What this process has counted since it started, and nothing older.

        Deliberately not /metrics: a Prometheus scraper pointed at that name
        and handed JSON fails in a way that reads as the service being down.
        """
        return counters()

    @app.exception_handler(UnsupportedAudio)
    async def unreadable_audio(request: Request, error: Exception) -> JSONResponse:
        """400: the file is intact and this service cannot read it.

        Not 415 -- the media type was right, the samples inside it were not --
        and not 500, because nothing here is broken. Sending the same file
        again cannot help, and the message says which conversion does.
        """
        count("audio refused")
        return JSONResponse({"detail": str(error)}, status_code=status.HTTP_400_BAD_REQUEST)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> JSONResponse:
        """Every other failure, in the shape the rest of the contract uses.

        Starlette's default here is 21 bytes of ``text/plain``, so a client that
        calls ``.json()`` on the error path -- which is every client, because
        every other error on this service is JSON -- meets a parse error instead
        of the failure. The id is the one this request has been carrying, so the
        line in the log is findable from what the caller was shown.

        The reason is not in the body. It is the engine's own words -- "CUDA out
        of memory: tried to allocate 44.00 GiB" -- and these endpoints take no
        credential. It is in the log, under the same id.

        Nothing is logged here: Starlette re-raises after this returns, so the
        server logs the traceback itself, and logging it here as well puts two
        copies of it in the file.
        """
        request_id = getattr(request.state, "request_id", None)
        count("unhandled")
        return JSONResponse(
            {"detail": "internal error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"x-request-id": request_id} if request_id else None,
        )

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
            getattr(request.app.state, "warm_up_error", None),
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
        # Headers first, body second: a wrapped body is refused before a byte
        # of it is read, and what is left is refused as it arrives.
        refuse_envelope(request)
        pcm = await audio_body(request)
        if not pcm:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "request body carried no audio")

        # The same call the recogniser makes, for the same reason: a container
        # states its own rate and the endpoint only assumes one. The field used
        # to divide the whole body -- header included -- by the assumed rate,
        # which reported a ten-second 24 kHz recording as 15.00 seconds and a
        # 48 kHz one as 30.00. Doing it here also moves the refusal of a
        # container this service cannot read in front of the model, where it
        # costs nothing.
        samples, rate = unwrap_wav(pcm, INPUT_SAMPLE_RATE)
        seconds = len(whole_samples(samples)) / 2 / rate
        # Dropped here rather than left to fall out of scope: this name is a
        # real copy of the body's data chunk and the scope it is in spans the
        # await below, so without this every in-flight request holds a second
        # whole recording -- measured at 3.0x the body against 2.0x, which for
        # a twenty-minute dictation is 38 MB each.
        del samples

        started = time.perf_counter()
        text, language = await engine.transcribe(pcm, INPUT_SAMPLE_RATE)
        asr_ms = (time.perf_counter() - started) * 1000

        polished: str | None = None
        polish_failed = False
        polish_ms = 0.0
        polish = getattr(engine, "polish", None)
        if polish is not None:
            started = time.perf_counter()
            try:
                polished = await polish(text)
            except Exception as error:  # noqa: BLE001 - transcript is worth more than the reason
                # Dictation's second stage is optional; its first is not. This
                # used to be evaluated inside the response object, so a
                # polisher that raised took the finished transcript with it and
                # the caller got a 500 and not one word. Degrading to the raw
                # text is what a dictation client would do with the failure
                # anyway -- what it could not do was get the text at all.
                polish_failed = True
                # The type and where it was raised, never the message and never
                # the traceback. An exception message is written by whoever
                # raised it, and the ones on this path are raised by a tokeniser
                # and a chat template, which put the input they choked on into
                # the message because that is where the message earns its keep.
                # Measured against a stub that does the same: the transcript
                # appeared in the log ten times. The caller has the request id
                # and the text; what is missing here is only the reason.
                import traceback

                where = [
                    f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}"
                    for frame in traceback.extract_tb(error.__traceback__)[-3:]
                ]
                polish_failed_at = type(error).__name__
                _log.error(
                    json.dumps(
                        {
                            "event": "polish_failed",
                            "id": getattr(request.state, "request_id", None),
                            "chars": len(text),
                            "error": polish_failed_at,
                            "where": where,
                        },
                        ensure_ascii=False,
                    )
                )
            polish_ms = (time.perf_counter() - started) * 1000

        count("transcriptions")
        count("audio seconds", seconds)
        if not text:
            # The entry gate refused it, or there was genuinely no speech. Which
            # rule fired is not visible from here -- it is decided inside the
            # recogniser -- so this counts the outcome the caller sees.
            count("transcriptions empty")
        if polish_failed:
            count("polish failed")
        request.state.observed = {
            "audio_seconds": round(seconds, 3),
            "asr_ms": round(asr_ms, 1),
            "polish_ms": round(polish_ms, 1),
            "chars": len(text),
            "polished_chars": len(polished) if polished is not None else None,
            "polish_failed": polish_failed,
            "language": language,
        }
        return TranscriptionResponse(
            text=text,
            polished=polished,
            polish_failed=polish_failed,
            language=language,
            duration_seconds=seconds,
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
        # Here rather than on the field: see contract.SpeechRequest.input. The
        # message names the two numbers and never the text.
        if len(body.input) > LONGEST_SPOKEN_CHARACTERS:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"this text is {len(body.input)} characters; the synthesiser is given up to "
                f"{LONGEST_SPOKEN_CHARACTERS} in one request",
            )
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
        #
        # `speed` names where it belongs rather than only saying no. Refusing
        # without that reads as a gap the backend still owes, so the client
        # waits for it and nobody builds it -- and playback speed is not a gap:
        # a player does it better than a re-synthesis can. See DECISIONS 30.
        if body.speed != 1.0:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"speed={body.speed} is not applied here, and is not meant to be: set the "
                "rate on the player instead. A web or webview client has "
                "`audio.playbackRate`, which keeps the pitch, works on audio already "
                "fetched, and needs no second request. Ask for this endpoint to carry it "
                "only if the audio is played somewhere without that -- a native player, or "
                "a file handed to someone else",
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
        # A live transcription of this turn. Both recognisers have one: the
        # streaming model writes as it decodes, and the whole-segment one
        # reads the buffer again every second and shows what two readings
        # agree on. Whole-segment used to be treated as having nothing to
        # stream, which is true of the model and false of the product -- the
        # re-read preview is more accurate than the streaming model (0.1094
        # against 0.2796 on the same clips) and cheaper (0.075 of real time
        # against 0.323).
        opener = getattr(getattr(engine, "recogniser", None), "open", None)
        listening = opener(INPUT_SAMPLE_RATE) if callable(opener) else None
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
                    before = len(buffer)
                    try:
                        buffer_audio(buffer, event)
                    except (Unreadable, TooLongForModel) as refused:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(refused)}}
                        )
                    else:
                        # Words while the speaker is still talking, when the
                        # deployment serves a recogniser that can produce them.
                        # Additions only, and true by construction on both
                        # paths: the streaming model commits as it decodes,
                        # and the re-read one emits only the head two
                        # readings agree on. A client appends and never
                        # re-renders what it showed.
                        #
                        # The authoritative transcript is still the one the
                        # commit branch produces: these deltas are what the
                        # user watches, and the text they keep is what the
                        # polish stage returns at the end.
                        if listening is not None:
                            said = await listening.feed(bytes(buffer[before:]))
                            if said:
                                await websocket.send_json(
                                    {
                                        "type": (
                                            "conversation.item.input_audio_transcription.delta"
                                        ),
                                        "delta": said,
                                    }
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
                    if listening is not None:
                        tail = await listening.finish()
                        if tail:
                            await websocket.send_json(
                                {
                                    "type": ("conversation.item.input_audio_transcription.delta"),
                                    "delta": tail,
                                }
                            )
                        listening = opener(INPUT_SAMPLE_RATE)
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
                                buffer_audio(buffer, heard)
                            except (Unreadable, TooLongForModel) as refused:
                                await websocket.send_json(
                                    {"type": "error", "error": {"message": str(refused)}}
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
                        if isinstance(failure, TooLongForModel | UnsupportedAudio):
                            # A dictated turn longer than the model answers, or
                            # a container whose samples cannot be read.
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
