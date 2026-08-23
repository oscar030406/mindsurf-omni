"""The contract between this service and everything downstream of it.

Written before the model exists, and frozen once published: the backend and
the client are built against it in parallel, so a field that moves later costs
two other teams a day each.

Everything here is either an OpenAI request/response shape or a deliberate,
documented extension. Speaking OpenAI's protocol is not deference -- it means
the backend can point its existing client at us, and can point it back at a
hosted provider the moment our model misbehaves. That fallback is worth more
than any protocol we could design ourselves.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Audio formats are fixed, not negotiated. SenseVoice takes 16 kHz, Mimi emits
# 24 kHz, and a client that has to ask which is which will eventually guess.
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
AUDIO_ENCODING = "pcm_s16le"

SCHEMA_VERSION = 1

# The delivery the service can actually produce. Named once because two doors
# lead to it: the speech endpoint validates against this list through pydantic,
# and the realtime session.update event used to validate against nothing at all
# -- "angry" was accepted in silence and delivered as neutral.
Emotion = Literal["neutral", "happy", "care"]
EMOTIONS: tuple[str, ...] = ("neutral", "happy", "care")


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mindsurf-omni"
    # At least one. An empty list reached the tokenizer's chat template and
    # came back as IndexError, which the service served as a 500 -- a caller
    # whose message list was empty for its own reasons could not tell a bug in
    # its request from a fault in ours. Rejected here so it arrives as a field
    # error beside the ones for a bad role or an out-of-range speed.
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    # Zero, so the same question gets the same answer. Upstream ships 0.7,
    # where three asks give three different replies (agreement 0.418 at
    # 0.4, 0.270 at 0.7). Greedy alone loops -- the service pairs it with
    # a repetition penalty, and the pair was judged against 0.4 on 608
    # probes: 0.507 +/- 0.053, indistinguishable. See INTEGRATION 7.4.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage | None = None


# ---------------------------------------------------------------------------
# POST /v1/audio/speech
# ---------------------------------------------------------------------------


# The longest text the speech endpoint will accept in one request. Nothing in
# the service produces more -- a spoken turn is capped at
# ``cascade.ANSWERABLE_CHARACTERS``, the same 1300 -- and the synthesiser has no
# limit of its own and no chunking, so without this the field was unbounded:
# two million characters were accepted and handed straight on, HTTP 200. Stated
# here rather than imported because ``cascade`` imports this module.
LONGEST_SPOKEN_CHARACTERS = 1300


class SpeechRequest(BaseModel):
    model: str = "mindsurf-omni"
    # Length is checked in the route, not with ``max_length``. pydantic puts the
    # rejected value into its ValidationError and FastAPI serialises the error
    # whole, so the field constraint does not keep a large body out -- it sends
    # it back: two million characters came back as a 6 MB 422, and 512 MB posted
    # chunked came back as 512 MB while peak memory rose half a gigabyte. A
    # limit that answers in proportion to the attack is an amplifier.
    input: str
    # An OpenAI client sends a voice name. Ours are reference-audio ids
    # registered through /v1/voices, because voice control is in-context
    # cloning rather than a fixed set of trained voices.
    voice: str = "default"
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    # Extension: emotion rides alongside rather than being smuggled into the
    # text, so the spoken text and the delivery stay separable.
    emotion: Emotion = "neutral"


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    text: str
    # Extension: what the polish stage made of it, for the dictation product.
    # Null means no polisher is wired -- which a caller must be able to tell
    # apart from "this text needed no polishing".
    polished: str | None = None
    # Extension: the polish stage ran and raised. Needed because `polished`
    # being null now means three things -- no polisher is wired, a polisher
    # answered with nothing, and this -- and only this one is a fault. Without
    # it a deployment whose polisher has been broken since a bad deploy looks
    # exactly like one that never had a polisher, from every response.
    polish_failed: bool = False
    # Extension: the language the encoder actually detected, so a caller can
    # notice when Chinese audio was read as English before the answer is wrong.
    language: str | None = None
    # How long the recording is, taken from the container's own header when it
    # has one. Not derived from the body length: a 24 kHz wav is more bytes per
    # second than the endpoint's default rate, and this field used to report
    # that arithmetic rather than the recording -- 15.00 for ten seconds.
    duration_seconds: float | None = None


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


class ComponentInfo(BaseModel):
    name: str
    parameters: int | None = None
    sha256: str | None = None
    frozen: bool = False


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "mindsurf"
    # Which path is serving right now. The backend does not choose it, but it
    # must be able to see it -- a quality report is meaningless without it.
    path: Literal["native", "cascade"]
    fallback_available: bool
    components: list[ComponentInfo]
    licence: str
    commercial_use_permitted: bool


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# WS /v1/realtime
# ---------------------------------------------------------------------------
#
# Event names follow OpenAI's Realtime API so a client written against that
# API needs no new vocabulary. Only the subset the product uses is
# implemented; anything unrecognised is answered with an error frame rather
# than ignored, because silent no-ops are how a client ends up waiting forever.

CLIENT_EVENTS = {
    "input_audio_buffer.append",  # {"audio": base64 PCM16 16 kHz mono}
    "input_audio_buffer.commit",  # user stopped speaking; start responding
    "input_audio_buffer.clear",
    "response.cancel",  # barge-in: stop speaking now
    "session.update",  # {"voice": ..., "emotion": ...}
    "session.clear",  # forget this conversation; the next caller is someone else
}

# Only what the service actually sends. An event declared here and never
# emitted is worse than a missing one: a client written against the list waits
# for it. This set was four entries longer until 2026-08-10, and the four were
# fiction -- speech_started, speech_stopped and response.text.done had no emit
# site anywhere, and the transcription event is emitted now rather than
# declared and skipped.
SERVER_EVENTS = {
    "session.created",
    # The answer to session.update, added 2026-08-16 when that event stopped
    # being silent. A caller could set a voice or an emotion the build does not
    # have and hear nothing back either way, so there was no way to tell an
    # applied setting from a dropped one short of asking somebody to listen.
    "session.updated",
    "conversation.item.input_audio_transcription.completed",  # cascade only
    "response.created",
    "response.text.delta",
    "response.audio.delta",  # {"audio": base64 PCM16 24 kHz mono}
    "response.audio.done",
    "response.done",
    "error",
}


class RealtimeEvent(BaseModel):
    type: str
    event_id: str | None = None
    audio: str | None = None
    delta: str | None = None
    transcript: str | None = None
    voice: str | None = None
    emotion: str | None = None
    error: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Token and voice specification, served as data
# ---------------------------------------------------------------------------
#
# The client needs these to build prompts and control delivery. Serving them
# from the running model rather than a document means they cannot drift out of
# step with the weights.


class TokenSpec(BaseModel):
    schema_version: int = SCHEMA_VERSION
    text_vocab_size: int
    audio_codebooks: int
    audio_codebook_size: int
    audio_frame_rate_hz: float
    special_tokens: dict[str, int]
    audio_special_tokens: dict[str, int]
    input_sample_rate: int = INPUT_SAMPLE_RATE
    output_sample_rate: int = OUTPUT_SAMPLE_RATE
    audio_encoding: str = AUDIO_ENCODING


class VoiceInfo(BaseModel):
    id: str
    description: str
    # In-context cloning: a voice is a reference recording plus its speaker
    # embedding, not a set of trained weights. Adding one costs no training.
    reference_seconds: float | None = None
    speaker_embedding_dim: int | None = None


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]
