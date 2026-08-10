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

    def token_cost(self, characters_per_token: float = 1.5, count_audio: bool = True) -> int:
        """Roughly what this turn occupies in the sequence.

        Approximate on purpose: an exact count needs the tokenizer, and this
        runs on every append during a live conversation. It errs high, because
        under-counting is what produces the failure it exists to prevent.

        ``count_audio`` is false where the history that reaches the model is
        text: the cascade sends transcripts, so charging ten seconds of speech
        125 tokens there evicts turns while the real prompt is an eighth of the
        budget. The native path does send audio, and pays for it.
        """
        text_tokens = int(len(self.text) / characters_per_token) + 1
        if not count_audio:
            return text_tokens
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
    # Whether the audio itself reaches the model. True on the native path,
    # where a turn is audio tokens; false on the cascade, where what travels
    # is the transcript.
    counts_audio: bool = True

    @property
    def budget(self) -> int:
        return max(0, self.max_tokens - self.reserve_for_reply)

    def used_tokens(self) -> int:
        return sum(turn.token_cost(count_audio=self.counts_audio) for turn in self.turns)

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
        """The history as a prompt, without the turns that have no text.

        A turn can be empty two ways: the native path never transcribes, and
        the cascade produces nothing for silence or for a barge-in that landed
        before recognition finished. Either way an empty turn renders as a bare
        ``<|im_start|>user<|im_end|>`` the model has never seen, and it would
        ride in every prompt for the rest of the session. The turn stays in the
        ledger -- it happened, and it occupied the sequence.
        """
        return [{"role": turn.role, "content": turn.text} for turn in self.turns if turn.text]

    def summary(self) -> dict[str, int]:
        """What a caller needs to see that history was silently shortened."""
        return {
            "turns": len(self.turns),
            "used_tokens": self.used_tokens(),
            "budget": self.budget,
            "dropped_turns": self.dropped,
        }
