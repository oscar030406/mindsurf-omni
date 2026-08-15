"""Tidy a transcript without rewriting it.

The second-phase product is dictation: speech goes to the recogniser, the
recogniser's text goes here, and what comes out lands in the user's text box.
The measured job (see the floor round) is 89% deleting spoken filler, 0.6%
fixing a homophone, and nothing at all for punctuation -- SenseVoice already
writes it, in the same places the original had it.

So this stage is built to delete. The decode is restricted to a subsequence of
the transcript, which makes three things true at once:

* **Invention is impossible.** Every character in the output came from the
  transcript. The unconstrained model rewrote a sentence in roughly one turn in
  ten, and a dictation tool that edits what the user said is worse than one
  that leaves the filler in.
* **Substitution is impossible too.** 它 heard as 他 stays 他. That is the 0.6%
  the floor measured, given up on purpose.
* **A single step cannot jump a clause.** The window is bounded; without that
  bound "delete only" quietly becomes "delete the rest of the sentence", which
  is what the first constrained run did (content retention 0.88).

The instruction is fixed here rather than passed in: the model was trained on
these words, and a caller who changes them is running a different model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSTRUCTION = (
    "把下面这段语音转写整理成通顺的文字。只删掉口语词和重复，不要改写内容，不要添加任何东西。"
)

# How far one step may skip. Six was the best of the widths measured: three
# removed less filler, unbounded deleted clauses.
LOOKAHEAD = 6

# The words this stage exists to remove. Canonical here rather than in the data
# builder: the builder injects them, the decoder is allowed to step over them,
# and a second list would let those two drift apart.
LEADING_FILLERS = ("嗯", "呃", "那个", "这个", "就是", "然后", "反正", "其实", "我觉得")
BRIDGING_FILLERS = ("你知道吧", "怎么说呢", "对吧")

# How the recogniser actually spells the above. The two lists were the same
# thing until the service was driven by hand: 11.3% of the filler that reaches
# a transcript is not spelled the way it was said. SenseVoice writes 呃 as
# 饿 恶 啊 鄂 扼 2 and 嗯 as 恩 摁 温, so the decoder's door -- which matches the
# vocabulary literally -- never opened for one occurrence in nine.
#
# Only the spellings that never occur in the corpus as ordinary words are here.
# Counted over both pools, 4168 sentences: 摁 鄂 唉 哎 呐 appear zero times, so
# stepping over one can only ever be removing filler. 温 (289), 饿 (12) and
# 啊 (11) are left out for exactly the opposite reason -- 水温 and 是不是饿了 are
# what a user said, and a door that opens on them is a door onto content.
RECOGNISED_FILLERS = ("摁", "鄂", "唉", "哎", "呐")


# Where a sentence ends, for splitting a long dictation into pieces the model
# was actually trained on. Commas are not here on purpose: a clause is not a
# sentence, and cutting at every comma would take away the context the model
# needs to tell a filler 就是 from a copula one.
_SENTENCE_END = re.compile(r"[。！？；\n]+")

# How much of a piece the output has to have consumed to be trusted. Not tuned:
# an output that reached less than nine tenths of its input has dropped a
# clause, and dropping a clause is the failure this floor exists to refuse.
FLOOR = 0.90


def split_sentences(text: str) -> list[str]:
    """Sentences with their end marks kept, in order.

    Kept rather than stripped so that joining the pieces back reproduces the
    transcript exactly when nothing is removed.
    """
    pieces, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        pieces.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def consumed(source: str, output: str) -> float:
    """Share of ``source`` the output reached, matching greedily in order.

    Exact rather than approximate: under the copy constraint the output is
    always a subsequence of the input, so walking one against the other says
    how much of it the decode got through before it stopped.
    """
    if not source:
        return 1.0
    pointer = 0
    for char in output:
        while pointer < len(source) and source[pointer] != char:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer / len(source)


def build_prompt(transcript: str) -> str:
    return f"{INSTRUCTION}\n\n{transcript}"


def subsequence_pointer(source: list[int], produced: list[int]) -> int:
    """How far into the transcript the output has consumed, matching greedily."""
    pointer = 0
    for token in produced:
        while pointer < len(source) and source[pointer] != token:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer


def reachable(
    source: list[int],
    pointer: int,
    lookahead: int,
    fillers: tuple[tuple[int, ...], ...] = (),
    protect_head: bool = False,
) -> list[int]:
    """The tokens a single step may land on.

    Three rules, each of which came from a measured failure:

    * **A bounded window.** Unbounded, "delete only" became "delete the rest of
      the sentence" (content retention 0.88).
    * **A door for a known filler.** A window narrow enough to protect a clause
      cannot always step over 你知道吧.
    * **A protected first token.** Measured end to end, 你想看什么类型的电影
      came back as 想看什么类型的电影 -- the opening word deleted. At the very
      start nothing has been said yet, so there is no context to justify a
      deletion that is not a filler.
    """
    if not lookahead:
        return source[pointer:]
    window = list(source[pointer : pointer + lookahead])
    if protect_head and pointer == 0:
        window = source[:1]
    for filler in fillers:
        span = len(filler)
        if tuple(source[pointer : pointer + span]) == filler:
            after = pointer + span
            window += source[after : after + lookahead]
    return window


def project_onto(transcript: str, written: str) -> str:
    """Keep only what the model wrote that the transcript actually said.

    The other way to get deletion-only output. Instead of constraining the
    decode, let the model write freely and then drop whatever it added: the
    result is the model's text intersected with the transcript, in order.

    Measured against constraining the decode, this keeps the free model's
    better filler removal (0.919 against 0.860) while removing its invention,
    because an invented character has nothing to align to.

    Not the same as deleting the difference: a character the model moved counts
    as invented at its new position, which is the conservative reading and the
    one a dictation product wants.
    """
    import difflib

    matcher = difflib.SequenceMatcher(None, transcript, written, autojunk=False)
    return "".join(
        transcript[block.a : block.a + block.size] for block in matcher.get_matching_blocks()
    )


@dataclass(slots=True)
class Polisher:
    """The polish stage, or nothing when no checkpoint was named.

    Holds its own Thinker rather than sharing the cascade's: the polish weights
    are a different fine-tune of the same architecture, and pointing both at one
    checkpoint would silently make the dictation path answer questions.
    """

    checkpoint: Path
    tokenizer_dir: Path
    minimind_root: Path
    device: str = "cpu"
    variant: str = "mindsurf"
    lookahead: int = LOOKAHEAD
    # Off until an arm measures it: the failure it fixes is visible (the first
    # word of a sentence disappearing) but the fix has not been scored yet.
    protect_head: bool = False
    max_new_tokens: int = 256
    _model: Any = None
    _tokeniser: Any = None
    _fillers: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from mindsurf_omni.service.thinker import ThinkerGenerator

        generator = ThinkerGenerator(
            checkpoint=self.checkpoint,
            tokenizer_dir=self.tokenizer_dir,
            minimind_root=self.minimind_root,
            device=self.device,
            variant=self.variant,
        )
        generator.load()
        self._model = generator._model  # noqa: SLF001
        self._tokeniser = generator._tokenizer  # noqa: SLF001
        self._model.eval()

    async def polish(self, transcript: str) -> str:
        """The transcript with its filler removed, or the transcript unchanged."""
        import asyncio

        if not transcript.strip():
            return transcript
        return await asyncio.to_thread(self._polish_whole, transcript)

    def _polish_whole(self, transcript: str) -> str:
        """One sentence at a time, and never less text than came in.

        Both guards come from driving the running service with real dictation
        rather than from any of the four acceptance numbers, which are read on
        single corpus sentences of at most 160 characters.

        **Sentence at a time.** 164 seconds of speech is 718 characters, and
        the model was trained on single sentences. Fed the whole buffer it
        returned 18 characters -- 97% of the dictation gone, HTTP 200, no
        error. Measured end to end through the service: 74 s came back at 0.39
        of the transcript, 164 s at 0.03, while 243 s and 335 s came back
        untouched. Not a length cap, and not monotonic: it is a model a long
        way outside the distribution it was trained on. Splitting on sentence
        marks puts every call back inside it.

        **Never shorter than it should be.** A dictation tool that silently
        drops what the user said is worse than one that leaves the filler in,
        so a piece whose output consumed less than ``FLOOR`` of its input is
        discarded and the input kept. The copy constraint makes that test exact
        -- the output is always a subsequence of the input.
        """
        out = []
        for piece in split_sentences(transcript):
            if not piece.strip():
                out.append(piece)
                continue
            polished = self._polish(piece)
            out.append(polished if consumed(piece, polished) >= FLOOR else piece)
        return "".join(out)

    def _filler_spans(self) -> tuple[tuple[int, ...], ...]:
        """The filler vocabulary as token ids, tokenised once per process."""
        if self._fillers is None:
            self._fillers = tuple(
                tuple(self._tokeniser(word).input_ids)
                for word in (*LEADING_FILLERS, *BRIDGING_FILLERS, *RECOGNISED_FILLERS)
            )
        return self._fillers

    def _polish(self, transcript: str) -> str:
        import torch

        self.load()
        messages = [{"role": "user", "content": build_prompt(transcript)}]
        prompt = str(
            self._tokeniser.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        prompt_ids = torch.tensor([self._tokeniser(prompt).input_ids], device=self.device)
        source = list(self._tokeniser(transcript).input_ids)
        stop = int(self._tokeniser.eos_token_id)

        produced: list[int] = []
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                ids = (
                    prompt_ids
                    if not produced
                    else torch.cat(
                        [prompt_ids, torch.tensor([produced], device=prompt_ids.device)], dim=1
                    )
                )
                logits = self._model(input_ids=ids).logits[0, -1].float()
                pointer = subsequence_pointer(source, produced)
                ahead = set(
                    reachable(
                        source,
                        pointer,
                        self.lookahead,
                        self._filler_spans(),
                        self.protect_head,
                    )
                )
                ahead.add(stop)
                keep = torch.tensor(sorted(ahead), device=logits.device, dtype=torch.long)
                masked = torch.full_like(logits, float("-inf"))
                masked[keep] = logits[keep]
                chosen = int(masked.argmax())
                if chosen == stop:
                    break
                produced.append(chosen)
        # An empty answer means the model chose to delete everything. The
        # transcript is a better answer than nothing, and the caller cannot tell
        # the difference from a silent failure otherwise.
        polished = self._tokeniser.decode(produced, skip_special_tokens=True).strip()
        return polished or transcript
