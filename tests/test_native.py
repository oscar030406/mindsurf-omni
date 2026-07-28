"""The native path's chunking, which is where the latency argument lives."""

from __future__ import annotations

import array
import struct

import pytest

from mindsurf_omni.service.native import (
    DEFAULT_CHUNK_FRAMES,
    MIMI_FRAME_RATE_HZ,
    NativeConfig,
    chunk_codes,
    prepare_output,
)


def frames(count: int, codebooks: int = 8) -> list[list[int]]:
    """Mimi emits one value per codebook per frame."""
    return [[frame * 10 + book for book in range(codebooks)] for frame in range(count)]


def test_chunking_groups_frames_not_codebook_values() -> None:
    """Splitting on the wrong axis interleaves codebooks and decodes to noise."""
    chunks = chunk_codes(frames(8), chunk_frames=4)

    assert len(chunks) == 2
    assert [lead for _, lead in chunks] == [0, 0]  # no overlap asked for
    assert len(chunks[0][0]) == 4  # four frames
    assert len(chunks[0][0][0]) == 8  # eight codebooks in the first frame
    assert chunks[0][0][0] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert chunks[1][0][0] == [40, 41, 42, 43, 44, 45, 46, 47]


def test_the_last_short_chunk_is_kept() -> None:
    """Dropping it would clip the end of every reply."""
    chunks = chunk_codes(frames(10), chunk_frames=4)

    assert [len(chunk) for chunk, _ in chunks] == [4, 4, 2]
    assert sum(len(chunk) for chunk, _ in chunks) == 10


def test_chunking_preserves_every_frame_in_order() -> None:
    original = frames(23)

    flattened = [frame for chunk, _ in chunk_codes(original, 5) for frame in chunk]

    assert flattened == original


def test_each_chunk_carries_the_lead_in_the_decoder_needs() -> None:
    """A chunk decoded from nothing starts with an audible edge.

    The codec has temporal context, so upstream's streaming demo decodes
    chunk-plus-overlap and throws the lead-in away. The count travels with the
    frames because discarding the wrong amount silently shortens every chunk.
    """
    chunks = chunk_codes(frames(12), chunk_frames=4, overlap=2)

    assert [lead for _, lead in chunks] == [0, 2, 2]  # nothing precedes the first
    # The second chunk starts two frames early and says so.
    second, lead = chunks[1]
    assert len(second) == 6
    assert second[lead] == [40, 41, 42, 43, 44, 45, 46, 47]


def test_dropping_the_lead_in_returns_exactly_the_original_frames() -> None:
    """The overlap must not add or lose a frame once it is discarded."""
    original = frames(19)

    kept = [frame for chunk, lead in chunk_codes(original, 4, overlap=2) for frame in chunk[lead:]]

    assert kept == original


def test_a_negative_overlap_is_refused() -> None:
    """It would slice from the end of the list and decode the wrong audio."""
    with pytest.raises(ValueError, match="negative"):
        chunk_codes(frames(4), 2, overlap=-1)


def test_no_frames_yields_no_chunks_rather_than_one_empty_chunk() -> None:
    """An empty chunk would be handed to the decoder and produce a click."""
    assert chunk_codes([], 4) == []


def test_a_chunk_size_below_one_is_refused() -> None:
    """Zero would loop forever; the error should arrive at configuration time."""
    with pytest.raises(ValueError, match="at least 1"):
        chunk_codes(frames(4), 0)
    with pytest.raises(ValueError, match="at least 1"):
        NativeConfig(chunk_frames=0)


def test_the_default_chunk_is_the_latency_claim_it_is_supposed_to_be() -> None:
    """About 320 ms, which is what makes audio start flowing inside a reply."""
    config = NativeConfig()

    assert config.chunk_frames == DEFAULT_CHUNK_FRAMES
    assert config.chunk_ms == pytest.approx(320.0)
    # Two seconds of speech is 25 frames at Mimi's rate; a chunk must be a
    # small fraction of that or streaming buys nothing.
    assert config.chunk_frames < 2 * MIMI_FRAME_RATE_HZ / 4


def pcm(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


def samples_of(data: bytes) -> list[int]:
    out = array.array("h")
    out.frombytes(data)
    return list(out)


def test_output_preparation_resamples_to_the_contract_rate() -> None:
    decoded = pcm([8000] * 2400)  # 100 ms at 24 kHz

    prepared = prepare_output(decoded, 24_000, 16_000, trim=False)

    assert len(samples_of(prepared)) == 1600  # 100 ms at 16 kHz


def test_trimming_happens_before_resampling() -> None:
    """The trim margin is in milliseconds, which only matches its own rate."""
    decoded = pcm([0] * 2400 + [8000] * 2400 + [0] * 2400)

    prepared = prepare_output(decoded, 24_000, 24_000, trim=True)

    # Dead air gone, speech and its margin kept.
    assert 2400 <= len(samples_of(prepared)) < 7200


def test_a_silent_chunk_prepares_to_nothing() -> None:
    """Emitting it would play dead air the model never meant to produce."""
    assert prepare_output(pcm([0] * 2400), 24_000, 24_000, trim=True) == b""


def test_preparation_leaves_a_quiet_chunk_at_its_own_level() -> None:
    """Amplifying to a target would raise the noise floor between chunks."""
    quiet = pcm([200, -180, 150] * 400)

    prepared = prepare_output(quiet, 24_000, 24_000, trim=False)

    assert max(abs(sample) for sample in samples_of(prepared)) <= 200


def test_the_prompt_carries_one_audio_placeholder_per_encoder_frame() -> None:
    """Without them the model never sees the audio, and answers the empty turn.

    The features are written over the embeddings of a run of <|audio_pad|>
    tokens in the user turn. Omitting them is not a partial prompt: the model
    replies to nothing, in fluent generic filler, which reads as a model that
    cannot understand speech rather than as a caller that forgot a token.
    """
    from mindsurf_omni.service.native import AUDIO_PLACEHOLDER, NativeEngine

    class Recorder:
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            return "|".join(f"{turn['role']}:{turn['content']}" for turn in messages)

    engine = NativeEngine.__new__(NativeEngine)
    engine._tokenizer = Recorder()  # type: ignore[attr-defined]

    prompt = engine._prompt([{"role": "user", "content": "今天天气怎么样"}], audio_frames=3)

    assert prompt.count(AUDIO_PLACEHOLDER) == 3
    # The spoken turn replaces the written one rather than joining it: the
    # audio is the question, not a caption for it.
    assert "今天天气怎么样" not in prompt


def test_no_audio_leaves_the_written_turn_alone() -> None:
    """The text path must not grow placeholders for audio that never arrived."""
    from mindsurf_omni.service.native import AUDIO_PLACEHOLDER, NativeEngine

    class Recorder:
        def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            return "|".join(f"{turn['role']}:{turn['content']}" for turn in messages)

    engine = NativeEngine.__new__(NativeEngine)
    engine._tokenizer = Recorder()  # type: ignore[attr-defined]

    prompt = engine._prompt([{"role": "user", "content": "今天天气怎么样"}])

    assert AUDIO_PLACEHOLDER not in prompt
    assert "今天天气怎么样" in prompt


def test_the_serving_loader_reads_the_talker_shape_off_the_checkpoint() -> None:
    """The graft could not be served at all until this existed.

    Our Talker's feed-forward is 3584 wide and upstream's is 2432. The service
    built ours unconditionally, so loading the grafted checkpoint -- the one
    that passed acceptance -- raised twenty size mismatches at startup. A
    shape passed by configuration would be a shape someone can set wrong, and
    the way that goes wrong is a Talker left at random initialisation that
    still emits codes.
    """
    from mindsurf_omni.service.config import ConfigurationError
    from mindsurf_omni.service.native import OMNI_PARAMETERS, detect_shape

    class Tensor:  # only .shape is read, and torch is not installed here
        def __init__(self, *shape: int) -> None:
            self.shape = shape

    ours = {"talker.layers.0.mlp.gate_proj.weight": Tensor(3584, 768)}
    graft = {"talker.layers.0.mlp.gate_proj.weight": Tensor(2432, 768)}
    assert detect_shape(ours) == "mindsurf"
    assert detect_shape(graft) == "graft"
    assert OMNI_PARAMETERS["mindsurf"] != OMNI_PARAMETERS["graft"]

    # A text-only checkpoint is refused here rather than at the first missing
    # key, and an unrecognised width is refused rather than guessed.
    with pytest.raises(ConfigurationError, match="no Talker"):
        detect_shape({"model.layers.0.mlp.gate_proj.weight": Tensor(3584, 768)})
    with pytest.raises(ConfigurationError, match="neither ours"):
        detect_shape({"talker.layers.0.mlp.gate_proj.weight": Tensor(999, 768)})


def test_a_consumer_that_stops_stops_the_generation() -> None:
    """A caller that reads part of a turn must not leave the GPU still decoding.

    This is the bug that made the service get slower the more it was used:
    every realtime turn whose client hung up early -- including every turn of
    the latency measurement, which reads the first audio chunk and closes --
    left a generation running with nowhere to put its output. Nothing raised,
    so it surfaced only as the next turn being slower.
    """
    import asyncio
    import threading
    import time

    from mindsurf_omni.service.engine import GenerationSettings
    from mindsurf_omni.service.native import NativeEngine

    total = 200
    produced: list[int] = []
    finished = threading.Event()

    def stream(messages: object, settings: object, audio: object) -> object:
        try:
            for index in range(total):
                produced.append(index)
                time.sleep(0.002)
                yield f"t{index}", None
        finally:
            finished.set()

    engine = object.__new__(NativeEngine)
    engine._stream = stream  # type: ignore[method-assign]

    async def read_two() -> list[object]:
        turn = engine._turn(None, GenerationSettings(), None)
        seen: list[object] = []
        async for item in turn:
            seen.append(item)
            if len(seen) == 2:
                break
        await turn.aclose()
        return seen

    seen = asyncio.run(read_two())

    assert len(seen) == 2
    assert finished.wait(timeout=5), "the producer never noticed the consumer had left"
    assert len(produced) < total, "generation ran to completion after the consumer stopped"
