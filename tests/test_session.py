"""Conversation trimming, and which end it drops from."""

from __future__ import annotations

from mindsurf_omni.service.session import AUDIO_TOKENS_PER_SECOND, Conversation, Turn


def test_audio_dominates_the_budget_which_is_why_this_exists() -> None:
    """Ten seconds of speech costs more than a long paragraph of text."""
    spoken = Turn(role="user", text="你好", audio_seconds=10.0)
    written = Turn(role="user", text="你好" * 30)

    assert spoken.token_cost() > written.token_cost()
    assert spoken.token_cost() >= int(10 * AUDIO_TOKENS_PER_SECOND)


def test_history_fits_inside_the_budget_after_appending() -> None:
    conversation = Conversation(max_tokens=200, reserve_for_reply=50)

    for index in range(20):
        conversation.append(Turn(role="user", text=f"第{index}个问题" * 5))

    assert conversation.used_tokens() <= conversation.budget


def test_the_oldest_turn_is_dropped_not_the_newest() -> None:
    """Answering an older question because the latest did not fit is worse."""
    conversation = Conversation(max_tokens=50, reserve_for_reply=10)

    for index in range(10):
        conversation.append(Turn(role="user", text=f"问题{index}" * 4))

    # The budget forces a trim: ten turns at nine tokens each is well over 40.
    assert conversation.dropped > 0, "the budget was too generous to test anything"
    assert "问题9" in conversation.turns[-1].text
    assert "问题0" not in " ".join(turn.text for turn in conversation.turns)


def test_the_newest_turn_survives_even_if_it_alone_exceeds_the_budget() -> None:
    """Trimming to nothing would leave the model with no question at all."""
    conversation = Conversation(max_tokens=50, reserve_for_reply=10)

    conversation.append(Turn(role="user", text="短", audio_seconds=60.0))

    assert len(conversation.turns) == 1
    assert conversation.used_tokens() > conversation.budget  # honest about it


def test_dropped_turns_are_counted_so_the_caller_can_see_it() -> None:
    """Silently shortened history is how a model appears to forget for no reason."""
    conversation = Conversation(max_tokens=100, reserve_for_reply=20)

    for index in range(12):
        conversation.append(Turn(role="user", text=f"第{index}轮" * 4))

    assert conversation.summary()["dropped_turns"] > 0
    assert conversation.summary()["used_tokens"] <= conversation.budget


def test_the_budget_leaves_room_for_the_reply() -> None:
    """A context that exactly fits the prompt has nowhere to put the answer."""
    conversation = Conversation(max_tokens=1000, reserve_for_reply=400)

    assert conversation.budget == 600


def test_clearing_resets_the_dropped_count_too() -> None:
    conversation = Conversation(max_tokens=60, reserve_for_reply=10)
    for index in range(8):
        conversation.append(Turn(role="user", text=f"x{index}" * 5))
    assert conversation.dropped > 0

    conversation.clear()

    assert conversation.turns == []
    assert conversation.summary()["dropped_turns"] == 0


def test_messages_come_out_in_the_shape_the_model_expects() -> None:
    conversation = Conversation()
    conversation.append(Turn(role="user", text="你好"))
    conversation.append(Turn(role="assistant", text="你好呀"))

    assert conversation.messages() == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]
