"""The Talker-isolated protocol: fixed text, forced, paired."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest
from scripts.evaluate_talker import EOS_TOKEN, SHAPES, forced_text_plan, load_texts

ROOT = Path(__file__).resolve().parent.parent


def test_the_plan_is_the_target_then_eos() -> None:
    """The whole point is that the text axis carries no sampling at all."""
    assert forced_text_plan([5, 6, 7]) == [5, 6, 7, EOS_TOKEN]
    assert forced_text_plan([]) == [EOS_TOKEN]


def test_both_shapes_are_stated_not_defaulted() -> None:
    """A shape left to defaults cost one training run and one wrong conclusion.

    "mindsurf" must carry our overrides explicitly; "upstream-default" being
    empty is itself a statement -- the trainer defaults ARE upstream's shape.
    """
    assert SHAPES["mindsurf"] == {"intermediate_size": 3584, "num_key_value_heads": 8}
    assert SHAPES["upstream-default"] == {}


def test_the_frozen_text_set_holds_what_the_protocol_needs() -> None:
    """160 texts long enough that one wrong character is 1%, not 11%.

    The floor-run lesson: CER normalises by reference length, so short texts
    make every error enormous and nothing is comparable across sets. This set
    exists to hold the text axis still; it must not quietly change shape.
    """
    rows = load_texts(ROOT / "configs" / "talker_texts_zh_v1.jsonl")

    assert len(rows) == 160
    assert len({row["id"] for row in rows}) == 160, "duplicate ids would double-count pairs"
    lengths = [len(row["text"]) for row in rows]
    assert statistics.median(lengths) > 60, "short texts inflate per-character error"
    assert all(row["prompt"].strip() for row in rows), "the forced turn needs its context"


def test_an_empty_target_is_refused_up_front(tmp_path: Path) -> None:
    """An empty text would be scored as the Talker producing silence."""
    bad = tmp_path / "texts.jsonl"
    bad.write_text(
        json.dumps({"id": "a", "prompt": "x", "text": " "}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(SystemExit, match="no text"):
        load_texts(bad)


def test_one_temperature_or_eight_but_not_four() -> None:
    """Four values have no meaning for eight codebooks, and broadcasting would invent one."""
    import pytest
    from scripts.evaluate_talker import parse_temperature

    assert parse_temperature("0.2") == 0.2
    assert parse_temperature("0.2,0.2,0.15,0.15,0.1,0.1,0.05,0.05") == [
        0.2,
        0.2,
        0.15,
        0.15,
        0.1,
        0.1,
        0.05,
        0.05,
    ]
    with pytest.raises(SystemExit, match="1 or 8"):
        parse_temperature("0.2,0.3,0.4,0.5")


def test_a_per_codebook_temperature_reaches_the_right_codebook() -> None:
    """The list is indexed by codebook, so an off-by-one would sample the wrong stack."""
    from scripts.evaluate_talker import AudioSampling

    sampling = AudioSampling(temperature=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    assert sampling.for_codebook(0) == 0.1
    assert sampling.for_codebook(7) == 0.8
    assert AudioSampling(temperature=0.2).for_codebook(5) == 0.2


def test_the_reference_strip_is_right_aligned_with_the_speaker_token_ahead_of_it() -> None:
    """Off by one and the model clones a reference that starts mid-syllable.

    stream_generate right-aligns the reference codes to end just before the
    assistant's first position, and puts the speaker token one slot ahead. The
    failure mode is not a crash: it is speech in some other voice, with nothing
    in the artifact saying so. So the arithmetic is pinned here rather than
    trusted to stay copied.
    """
    start_pos, ref_len = 40, 12
    reserve = 1  # a speaker embedding is present
    fill_end = start_pos
    fill_start = max(reserve, start_pos - ref_len)

    assert (fill_start, fill_end) == (28, 40)
    assert fill_end - fill_start == ref_len  # the whole strip fits
    assert fill_start - 1 == 27  # the speaker token sits immediately before it

    # A strip longer than the prompt is truncated from the left, keeping the
    # end that abuts the assistant rather than the beginning.
    long_start, long_ref = 5, 101
    clipped_start = max(reserve, long_start - long_ref)
    assert clipped_start == reserve
    assert long_start - clipped_start == 4  # only the last four frames survive


def test_the_manifest_names_the_voice_even_when_there_is_none() -> None:
    """A conditioned run and an unconditioned one differ in nothing else."""
    source = (ROOT / "scripts" / "evaluate_talker.py").read_text(encoding="utf-8")

    assert '"voice": args.voice,' in source
    # And the vector has to ride on every forward call, not just the first --
    # the cached path forwards one position at a time.
    assert source.count("**speaker,") == 2


def test_guidance_of_one_is_the_conditional_pass_itself() -> None:
    """Off by definition rather than by a branch.

    A guidance knob whose neutral value merely happens to land near the unguided
    result is a knob that silently moves the baseline every arm is compared
    against. 1.0 has to hand back the conditional logits themselves.
    """
    from scripts.evaluate_talker import AudioSampling

    conditional, unconditional = object(), object()

    assert AudioSampling().guided(conditional, unconditional) is conditional


def test_guidance_extrapolates_away_from_the_speakerless_pass() -> None:
    """Above 1.0 it is extrapolation, not interpolation.

    Getting the sign backwards would push generation towards the branch that
    was told nothing about the speaker -- the opposite of the point, and it
    would still produce speech, just not that person's.
    """
    from scripts.evaluate_talker import AudioSampling

    # Plain numbers on purpose: the blend is arithmetic on whatever the model
    # hands back, and pinning it should not need a tensor library the test
    # environment deliberately does not carry.
    assert AudioSampling(guidance=2.0).guided(2.0, 1.0) == 3.0
    assert AudioSampling(guidance=2.0).guided(0.0, 1.0) == -1.0
    # A codebook where the two passes agree is left where it was: the gap the
    # reference opened is the only thing being widened.
    assert AudioSampling(guidance=3.0).guided(0.7, 0.7) == pytest.approx(0.7)


def test_the_unconditional_branch_sees_the_codes_that_were_drawn() -> None:
    """It is a counterfactual about the speaker, not about the words.

    If the speakerless branch is left to drift on padding it is soon scoring a
    different utterance, and the difference the blend extrapolates stops being
    about the reference at all. Nothing about the audio would look wrong.
    """
    source = (ROOT / "scripts" / "evaluate_talker.py").read_text(encoding="utf-8")
    guided = source[source.index("if guiding:\n                # The unconditional") :]

    assert "plain_buffer[0, layer, -1] = audio_codes[layer][-1]" in guided
    # And it starts from padding everywhere, including the speaker slot: a
    # leftover reference code there would quietly weaken the guidance.
    assert "plain_buffer[:, :, :] = model.audio_pad_token" in source
