"""Where the instruction lands decides what the model learns.

A multi-turn row has several user turns and only the last one is answered by
the assistant audio this row was labelled from. Attaching the instruction to
the first turn would pair "be happy" with a question whose answer is a
different clip, and nothing downstream would notice -- the corpus would look
well-formed and the model would learn a mapping built on mismatched pairs.
"""

from __future__ import annotations

import json

from scripts.build_emotion_corpus import FOLD_TO_NEUTRAL, condition, instruction_for


def test_the_instruction_goes_on_the_last_user_turn() -> None:
    conversations = json.dumps(
        [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
        ],
        ensure_ascii=False,
    )

    turns = json.loads(condition(conversations, "请用开心的语气回答。"))

    assert turns[0]["content"] == "第一问"
    assert turns[2]["content"] == "请用开心的语气回答。第二问"


def test_the_rare_tail_folds_to_neutral() -> None:
    """0.9% of the corpus across four labels: too few to learn, too few to certify."""
    for label in FOLD_TO_NEUTRAL:
        assert instruction_for(f"x/{label}") == instruction_for("中立/neutral")
    assert instruction_for("<unk>") == instruction_for("中立/neutral")


def test_each_usable_label_gets_its_own_instruction() -> None:
    texts = {
        instruction_for(f"x/{name}") for name in ("happy", "angry", "surprised", "sad", "neutral")
    }

    assert len(texts) == 5


def test_a_row_with_no_user_turn_is_left_alone() -> None:
    conversations = json.dumps([{"role": "assistant", "content": "只有回答"}], ensure_ascii=False)

    assert condition(conversations, "请用开心的语气回答。") == conversations
