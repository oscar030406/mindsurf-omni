"""The probe has to carry the corpus's own wording and join, not a lookalike."""

from __future__ import annotations

from scripts.build_emotion_corpus import INSTRUCTIONS, condition
from scripts.build_emotion_probes import probe_rows

ROWS = [{"id": "zh000", "prompt": "今天天气怎么样", "text": "你好，今天天气怎么样？"}]


def test_join_matches_what_the_corpus_builder_did() -> None:
    """Same instruction, same join: the corpus prepends with no separator."""
    built = probe_rows(ROWS, INSTRUCTIONS["happy"])[0]["prompt"]
    conversations = condition(
        '[{"role": "user", "content": "今天天气怎么样"}]', INSTRUCTIONS["happy"]
    )
    assert built in conversations


def test_spoken_text_and_id_are_untouched() -> None:
    """Only the user turn moves, or the arms are not paired on the same sentence."""
    built = probe_rows(ROWS, INSTRUCTIONS["sad"])[0]
    assert built["text"] == ROWS[0]["text"]
    assert built["id"] == ROWS[0]["id"]


def test_every_instruction_produces_a_distinct_prompt() -> None:
    prompts = {probe_rows(ROWS, text)[0]["prompt"] for text in INSTRUCTIONS.values()}
    assert len(prompts) == len(INSTRUCTIONS)
