"""Conversation state for a realtime session.

Speech turns accumulate context, and context is where a small model runs out of
room. Mimi emits 12.5 frames per second, so ten seconds of speech is 125 audio
tokens before any text -- a few turns of that and the sequence is longer than
the model has ever seen.

So history is trimmed, and what gets dropped is chosen rather than left to
whichever end the buffer happens to overflow. Dropping the oldest turn first
keeps the conversation coherent; dropping the newest would make the model
answer a question nobody just asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Mimi's rate. Everything about the audio budget follows from it.
AUDIO_TOKENS_PER_SECOND = 12.5


@dataclass(frozen=True, slots=True)
class Turn:
    role: str  # "user" or "assistant"
    text: str
    audio_seconds: float = 0.0

    def token_cost(self, characters_per_token: float = 1.5) -> int:
        """Roughly what this turn occupies in the sequence.

        Approximate on purpose: an exact count needs the tokenizer, and this
        runs on every append during a live conversation. It errs high, because
        under-counting is what produces the failure it exists to prevent.
        """
        text_tokens = int(len(self.text) / characters_per_token) + 1
        audio_tokens = int(self.audio_seconds * AUDIO_TOKENS_PER_SECOND)
        return text_tokens + audio_tokens


@dataclass
class Conversation:
    """Turns, trimmed to fit, oldest first.

    ``max_tokens`` should leave room for the reply as well as the history --
    a context that exactly fits the prompt has nowhere to put the answer.
    """

    max_tokens: int = 3072
    reserve_for_reply: int = 512
    turns: list[Turn] = field(default_factory=list)
    dropped: int = 0

    @property
    def budget(self) -> int:
        return max(0, self.max_tokens - self.reserve_for_reply)

    def used_tokens(self) -> int:
        return sum(turn.token_cost() for turn in self.turns)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)
        self._trim()

    def _trim(self) -> None:
        # The newest turn is never dropped: answering an older question because
        # the latest one did not fit is worse than losing history.
        while len(self.turns) > 1 and self.used_tokens() > self.budget:
            self.turns.pop(0)
            self.dropped += 1

    def clear(self) -> None:
        self.turns.clear()
        self.dropped = 0

    def messages(self) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.text} for turn in self.turns]

    def summary(self) -> dict[str, int]:
        """What a caller needs to see that history was silently shortened."""
        return {
            "turns": len(self.turns),
            "used_tokens": self.used_tokens(),
            "budget": self.budget,
            "dropped_turns": self.dropped,
        }
