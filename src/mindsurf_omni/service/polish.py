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
from dataclasses import dataclass, field
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

# The longest text the model was trained on. The pool that built the pairs
# dropped anything past 160 characters, so this is not a guess about capacity --
# it is the edge of what the weights have seen. Below it the transcript goes in
# whole; above it, consecutive sentences are grouped up to this length.
#
# Grouping rather than one sentence per call, because a sentence on its own is
# also out of distribution: measured over 986 held-out transcripts, splitting
# every sentence cost 0.039 of filler clearance (0.8625 to 0.8234) and 0.007 of
# CER for nothing, since none of them was long enough to need splitting.
TRAINED_LENGTH = 160

# Spellings that mean the model might have work to do here. Wider than the
# decoder's door on purpose: the door decides whether a span may be deleted, so
# a wrong entry costs content, while this decides whether the model is called at
# all, so a wrong entry costs one decode. The costs are not symmetric and the
# lists should not be either -- 饿 恶 啊 温 are here and not in the door.
#
# Measured over 986 held-out transcripts: skipping the pieces that match nothing
# here removes 46.7% of the calls and the four numbers do not get worse -- CER
# 0.0566 to 0.0537, retention 0.9683 to 0.9717. The model was editing sentences
# with nothing to remove, and by construction those edits were over-deletion.
_WORTH_A_LOOK = (
    *LEADING_FILLERS,
    *BRIDGING_FILLERS,
    *RECOGNISED_FILLERS,
    "饿",
    "恶",
    "啊",
    "恩",
    "扼",
    "温",
    "反证",
    "其是",
    "奇实",
)


def worth_polishing(piece: str, longest_repeat: int = 5) -> bool:
    """Whether this piece holds anything the stage could remove.

    Filler word or an exact adjacent repetition. Nothing else is in scope --
    substitution was given up on purpose (DECISIONS §16) and punctuation is the
    recogniser's to write.
    """
    if any(word in piece for word in _WORTH_A_LOOK):
        return True
    # From one character, unlike the merge rule's exemption. There the cost of
    # calling 天天 a repetition is keeping a deletion; here it is one decode,
    # while missing 我我想问一下 means never removing it at all.
    return any(
        piece[start : start + size] == piece[start + size : start + 2 * size]
        for size in range(1, longest_repeat + 1)
        for start in range(len(piece) - 2 * size + 1)
    )


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


def group_sentences(text: str, longest: int = TRAINED_LENGTH) -> list[str]:
    """The text in pieces the model was trained on, splitting only when it must.

    A transcript inside ``longest`` comes back as one piece, so the ordinary
    dictation is polished exactly as it was before any of this existed. Past
    that, consecutive sentences are grouped until adding the next would cross
    the line -- and a single sentence longer than the line is left whole rather
    than cut mid-clause, because a piece that starts halfway through a sentence
    is worse input than a long one.
    """
    if len(text) <= longest:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if current and len(current) + len(sentence) > longest:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
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
    # How many pieces one decode may carry. Wide enough that a step is full,
    # narrow enough that one caller's long dictation does not hold everyone
    # else's: a batch runs until its longest member stops, so the cost of a
    # wide batch is paid by its shortest piece. Measured at 64 on real-length
    # pieces the wait became minutes; eight keeps the throughput and not that.
    max_batch: int = 8
    _model: Any = None
    _queue: Any = field(default_factory=list)
    _queue_lock: Any = None
    _ready: Any = None
    _worker: Any = None
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
        """The transcript with its filler removed, or the transcript unchanged.

        Grouped to what the model was trained on, and never returning less text
        than came in. Both from driving the running service: 164 seconds of
        dictation came back as 18 characters, HTTP 200, no error, because the
        whole buffer was ten times longer than anything the weights had seen. A
        piece whose output stopped early is thrown away and its input kept -- a
        dictation tool that silently drops what the user said is worse than one
        that leaves the filler in. The copy constraint makes that test exact:
        the output is always a subsequence of the input.

        A piece with no filler word and no repetition never reaches the model.
        Measured over 986 held-out transcripts that removes 46.7% of the calls
        and the four numbers do not get worse -- the model was editing
        sentences with nothing to remove, and by construction those edits were
        over-deletion.
        """
        if not transcript.strip():
            return transcript
        pieces = group_sentences(transcript)
        wanted = [piece for piece in pieces if piece.strip() and worth_polishing(piece)]
        if not wanted:
            return transcript
        answers = dict(zip(wanted, await self._decode(wanted), strict=True))

        out = []
        for piece in pieces:
            if piece not in answers:
                out.append(piece)
                continue
            answer = answers[piece]
            out.append(answer if consumed(piece, answer) >= FLOOR else piece)
        return "".join(out)

    async def _decode(self, pieces: list[str]) -> list[str]:
        """Queue these pieces and wait for them, sharing a batch with whoever else is waiting.

        One card, so requests queue whatever happens. Measured on the running
        service, eight concurrent dictations each waited 7182 ms for work that
        takes 1380 ms alone -- the eighth caller paying for the other seven
        while the GPU ran a batch of one. Sharing the batch took that to
        3098 ms.

        Batching does not change what comes out. Checked twice, because the
        first check said otherwise and the first check was wrong: 260 pieces
        decoded alone, in batches of 8 and in batches of 32 came back
        character-identical, 0 of 260 different. The one difference seen
        earlier was the recogniser's dither, not this -- it was measured before
        that was fixed, and re-running the same concurrent test afterwards
        gives 0 of 8 different in the transcript and 0 of 8 in the polish.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        waiting = [loop.create_future() for _ in pieces]
        async with self._lock():
            self._queue.extend(zip(pieces, waiting, strict=True))
            self._wake().set()
        return list(await asyncio.gather(*waiting))

    def _lock(self) -> Any:
        import asyncio

        if self._queue_lock is None:
            self._queue_lock = asyncio.Lock()
        return self._queue_lock

    def _wake(self) -> Any:
        """The event the worker sleeps on, and the worker, made on demand.

        Built here rather than at assembly because both belong to a running
        loop and assembly happens before there is one.
        """
        import asyncio

        if self._ready is None:
            self._ready = asyncio.Event()
            self._worker = asyncio.get_running_loop().create_task(self._drain())
        return self._ready

    async def _drain(self) -> None:
        """Take whatever is waiting, decode it together, hand each answer back.

        Whatever is waiting rather than a fixed batch: the point is to fill a
        step that was going to run anyway, so a lone request must not wait for
        company.
        """
        import asyncio

        while True:
            await self._ready.wait()
            async with self._lock():
                batch = self._queue[: self.max_batch]
                self._queue = self._queue[self.max_batch :]
                if not self._queue:
                    self._ready.clear()
            if not batch:
                continue
            try:
                answers = await asyncio.to_thread(self._polish_batch, [piece for piece, _ in batch])
            except Exception as error:  # noqa: BLE001 - every waiter needs the reason
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)
                continue
            for (_, future), answer in zip(batch, answers, strict=True):
                if not future.done():
                    future.set_result(answer)

    def _polish_batch(self, pieces: list[str]) -> list[str]:
        """Every piece decoded in one pass, each under its own copy constraint.

        The pieces of a dictation do not depend on each other, and the model is
        90M parameters against sequences of fifty tokens -- so a step costs the
        same whether it carries one sentence or fifty. Measured on this card:
        12.62 ms a step at batch 1 and 9.30 ms at batch 57, which is 315 ms a
        sentence against 4.1 ms. The serial loop was paying that 77 times over,
        and the polish stage is 76-90% of what a dictation costs.

        Left-padded, because the constraint is per row and the rows have
        different prompt lengths. RoPE is relative, so shifting a row's real
        tokens along the position axis leaves the attention between them
        unchanged, and the pad positions are masked out of every score.

        The mask is rebuilt per row per step rather than shared: what each row
        may emit next depends on how far its own output has consumed its own
        transcript, and that is the whole point of the constraint.
        """
        import torch

        if not pieces:
            return []
        self.load()
        stop = int(self._tokeniser.eos_token_id)
        prompts, sources = [], []
        for piece in pieces:
            messages = [{"role": "user", "content": build_prompt(piece)}]
            text = str(
                self._tokeniser.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            prompts.append(list(self._tokeniser(text).input_ids))
            sources.append(list(self._tokeniser(piece).input_ids))

        width = max(len(ids) for ids in prompts)
        padded = torch.tensor(
            [[stop] * (width - len(ids)) + ids for ids in prompts], device=self.device
        )
        mask = torch.tensor(
            [[0.0] * (width - len(ids)) + [1.0] * len(ids) for ids in prompts],
            device=self.device,
        )

        produced: list[list[int]] = [[] for _ in pieces]
        done = [False] * len(pieces)
        cache, step_in = None, padded
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = self._model(
                    input_ids=step_in, attention_mask=mask, past_key_values=cache, use_cache=True
                )
                cache = out.past_key_values
                logits = out.logits[:, -1].float()
                nxt = []
                for row, piece_source in enumerate(sources):
                    if done[row]:
                        nxt.append(stop)
                        continue
                    ahead = set(
                        reachable(
                            piece_source,
                            subsequence_pointer(piece_source, produced[row]),
                            self.lookahead,
                            self._filler_spans(),
                            self.protect_head,
                        )
                    )
                    ahead.add(stop)
                    keep = torch.tensor(sorted(ahead), device=logits.device, dtype=torch.long)
                    masked = torch.full_like(logits[row], float("-inf"))
                    masked[keep] = logits[row][keep]
                    chosen = int(masked.argmax())
                    if chosen == stop:
                        done[row] = True
                    else:
                        produced[row].append(chosen)
                    nxt.append(chosen)
                if all(done):
                    break
                step_in = torch.tensor([[token] for token in nxt], device=self.device)
                mask = torch.cat([mask, torch.ones((len(pieces), 1), device=self.device)], dim=1)

        answers = []
        for piece, tokens in zip(pieces, produced, strict=True):
            text = self._tokeniser.decode(tokens, skip_special_tokens=True).strip()
            answers.append(text or piece)
        return answers

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
        cache = None
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                # Only what the model has not seen yet. Without the cache this
                # loop re-ran the whole prefix every step: a 25-token answer
                # behind a 40-token prompt cost 25 forwards averaging 52
                # positions instead of 65 in total, twenty times the work, and
                # the polish stage was 76-90% of the time a dictation took.
                # Mathematically the same decode -- same logits, same argmax.
                ids = (
                    prompt_ids
                    if cache is None
                    else torch.tensor([produced[-1:]], device=prompt_ids.device)
                )
                out = self._model(input_ids=ids, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0, -1].float()
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
