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
    assert len(chunks[0]) == 4  # four frames
    assert len(chunks[0][0]) == 8  # eight codebooks in the first frame
    assert chunks[0][0] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert chunks[1][0] == [40, 41, 42, 43, 44, 45, 46, 47]


def test_the_last_short_chunk_is_kept() -> None:
    """Dropping it would clip the end of every reply."""
    chunks = chunk_codes(frames(10), chunk_frames=4)

    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert sum(len(chunk) for chunk in chunks) == 10


def test_chunking_preserves_every_frame_in_order() -> None:
    original = frames(23)

    flattened = [frame for chunk in chunk_codes(original, 5) for frame in chunk]

    assert flattened == original


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
