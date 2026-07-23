"""The contract is frozen. This is what "frozen" means in practice.

The backend and the client are built against these shapes while the model
trains. A field that moves costs both teams a day, and the cost lands on
people who cannot see the change coming -- so the shapes are pinned here, and
a rename has to be a deliberate act that updates this file too.

Additive changes are fine and deliberately not pinned: the promise is "only
grows", not "never changes".
"""

from __future__ import annotations

from mindsurf_omni.contract import (
    AUDIO_ENCODING,
    CLIENT_EVENTS,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    SCHEMA_VERSION,
    SERVER_EVENTS,
    ChatCompletionRequest,
    ModelInfo,
    SpeechRequest,
    TokenSpec,
    TranscriptionResponse,
)


def test_the_audio_contract_has_not_moved() -> None:
    """Changing a rate silently changes the pitch of everything already shipped."""
    assert INPUT_SAMPLE_RATE == 16_000
    assert OUTPUT_SAMPLE_RATE == 24_000
    assert AUDIO_ENCODING == "pcm_s16le"


def test_the_openai_request_fields_are_the_ones_a_client_already_sends() -> None:
    """The whole reason for this protocol is that clients need no new vocabulary."""
    fields = set(ChatCompletionRequest.model_fields)

    assert {"model", "messages", "stream", "temperature", "top_p", "max_tokens"} <= fields


def test_the_speech_request_keeps_emotion_out_of_the_text() -> None:
    fields = set(SpeechRequest.model_fields)

    assert {"input", "voice", "emotion", "response_format", "speed"} <= fields


def test_the_transcription_response_reports_the_detected_language() -> None:
    """Chinese audio read as English must be visible before the answer is wrong."""
    assert "language" in TranscriptionResponse.model_fields


def test_the_model_description_carries_path_and_licence() -> None:
    """Two things a report is meaningless without."""
    fields = set(ModelInfo.model_fields)

    assert "path" in fields
    assert "commercial_use_permitted" in fields
    assert "components" in fields


def test_the_realtime_events_a_client_may_send_have_not_shrunk() -> None:
    """Removing one breaks a client that already sends it."""
    assert {
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "response.cancel",
    } <= CLIENT_EVENTS


def test_the_realtime_events_a_client_waits_for_have_not_shrunk() -> None:
    """A client blocked on an event that no longer arrives hangs forever."""
    assert {
        "response.text.delta",
        "response.audio.delta",
        "response.audio.done",
        "response.done",
        "error",
    } <= SERVER_EVENTS


def test_the_token_spec_shape_is_stable() -> None:
    """Clients build prompts from this; a rename breaks prompt construction."""
    fields = set(TokenSpec.model_fields)

    assert {
        "text_vocab_size",
        "audio_codebooks",
        "audio_codebook_size",
        "special_tokens",
        "audio_special_tokens",
        "input_sample_rate",
        "output_sample_rate",
    } <= fields


def test_the_schema_version_exists_so_a_breaking_change_can_be_announced() -> None:
    """Without it a client cannot tell a new server from an old one."""
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1
