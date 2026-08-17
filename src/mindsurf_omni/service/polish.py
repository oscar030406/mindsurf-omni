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

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindsurf_omni.service.config import ConfigurationError

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

PUNCTUATION = set("，。！？；：、")

# Particles that never open a Chinese sentence. 啊 and 呀 are deliberately not
# here -- "啊，太好了" is ordinary, and deleting that 啊 would be deleting content.
STRANDED_PARTICLES = set("吧呢嘛")

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


def drop_bridging_with_mark(text: str) -> str:
    """A bridging filler that survived, together with the mark it brought.

    The mirror image of the first shape ``tidy`` handles. That one is a filler
    that *was* deleted and left its punctuation behind; this one is a filler the
    arms did not delete at all, sitting mid-sentence with its own question mark:
    "记了下来。对吧？训练好之后", "热水擦，怎么说呢？重油污就上小苏打".

    It is the only leftover filler that gets its own rule, and the reason is
    that it is the only one a listener meets as a *wrong sentence boundary*
    rather than as an extra word. Read aloud, that mark is a 0.66 s pause with
    the pitch held flat where it should fall -- measured, and independently
    reported by a person who only heard the audio and said the phrasing did not
    match the grammar. The four criteria price it at zero: ``normalise_for_cer``
    strips punctuation before comparing.

    Only mid-sentence. A mark with nothing after it ends the sentence it is
    supposed to end, so there is no wrong boundary to remove -- and taking the
    filler there would strand the mark, which is the defect this file already
    has a function for.

    Only the bridging three. Measured on 986 held-out transcripts against the
    wider option of deleting every leftover vocabulary filler: this shape moves
    CER 0.0315 to 0.0302, filler clearance 0.9296 to 0.9364, invention 0.0248 to
    0.0236, retention unchanged at 0.9806, and takes mid-sentence marks from 41
    to 29. Deleting every leftover filler instead reads a better clearance
    (0.9489) and a worse retention (0.9796), and leaves the mid-sentence marks
    at 40 -- it is a lexicon overruling the model, and it does not fix the thing
    this is for. Twelve of the 41 marks are this shape; the other 29 are the
    recogniser putting a question mark inside a sentence it heard wrong, which
    nothing here can reach.

    Deleting only, so the copy constraint holds.
    """
    words = sorted(BRIDGING_FILLERS, key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for index, char in enumerate(text):
            if char not in "？！" or not text[index + 1 :].strip("".join(PUNCTUATION) + " \n"):
                continue
            word = next((w for w in words if text[:index].endswith(w)), None)
            if word is None:
                continue
            text = text[: index - len(word)] + text[index + 1 :]
            changed = True
            break
    return text


def tidy(text: str) -> str:
    """Drop what a deletion left with nothing in front of it.

    Three shapes, all read off the output rather than tracked back to which
    filler they belonged to, because all three are visible in the result.
    The third is ``drop_bridging_with_mark`` above, run first because it can
    leave doubled punctuation for the loop below to clear.

    Punctuation: deleting 对吧 out of "彩塑，对吧？特别震撼" leaves "彩塑，？特别
    震撼", a stranded question mark that reads worse than the filler did.

    Particles: deleting 我觉得 out of "我觉得吧这个真的挺好的" leaves "吧这个真的
    挺好的", which is not a sentence. Measured on the running service, both of
    these ship today.

    Neither costs anything on the four acceptance numbers -- ``normalise_for_cer``
    strips punctuation from both sides, and the stranded particle is a character
    the source contained, so content_kept counts it as kept. That is how the
    first one survived a whole round of tuning that reported the arm as better
    on three criteria at once, and the second one survived the round that fixed
    the first.

    Deleting only, so the copy constraint holds: the result is still a
    subsequence of the transcript.
    """
    text = drop_bridging_with_mark(text)
    out: list[str] = []
    for char in text:
        if char in PUNCTUATION and (not out or out[-1] in PUNCTUATION):
            continue
        if char in STRANDED_PARTICLES and (not out or out[-1] in PUNCTUATION):
            continue
        out.append(char)
    return "".join(out)


# ---------------------------------------------------------------------------
# Merging two arms.
#
# These moved out of scripts/merge_polish_arms.py on 2026-08-16, for the reason
# `tidy` moved before them: the product runs the merge now, and a function the
# product depends on cannot live in a scoring script. The script imports them
# from here, so there is one copy and the offline arms and the served text
# cannot drift the way tidy did for a whole round.
# ---------------------------------------------------------------------------

# Longest first, so 你知道吧 is matched before 你知道 would be.
VOCABULARY = tuple(sorted((*LEADING_FILLERS, *BRIDGING_FILLERS), key=len, reverse=True))


def dropped(source: str, output: str) -> set[int]:
    """Indices of ``source`` the arm did not carry into ``output``.

    A replaced span counts as dropped: under the copy constraint an arm cannot
    substitute, so anything the alignment calls a replacement is a deletion the
    matcher paired with unrelated context.
    """
    kept = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
        None, source, output, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            kept.update(range(i1, i2))
    return set(range(len(source))) - kept


def vocabulary_spans(source: str, drop: set[int]) -> set[int]:
    """The part of ``drop`` that spells a whole filler word.

    Whole rather than partial: half a filler is the defect this is meant to
    avoid, not one to import. 你知道吧 deleted down to 你知道 reads worse than
    leaving it alone.
    """
    kept: set[int] = set()
    for word in VOCABULARY:
        start = source.find(word)
        while start != -1:
            span = range(start, start + len(word))
            if all(index in drop for index in span):
                kept.update(span)
            start = source.find(word, start + 1)
    return kept


def repetition_spans(source: str, drop: set[int], shortest: int = 2, longest: int = 5) -> set[int]:
    """Deletions that remove one copy of an exact adjacent repetition.

    Exempt from the veto because a per-token head cannot represent the
    judgement at all -- it reads one position at a time and 时间时间 looks like
    two ordinary words. Measured: the tagger clears 0.437 of the injected
    repetition against the generator's 0.603, and letting it veto that work
    costs the whole difference.

    Two characters and up, because one is not a repetition in Chinese: 今天天气
    holds 天天 and 看看 is an ordinary word. Measured both ways over 986 sentences
    -- a floor of one reads 0.9083 filler clearance against two's 0.9055, same
    retention to four places -- so the shorter floor buys 0.003 that is inside
    the noise and pays for it with a class of false positive that is not.

    Either copy counts. The first version asked only whether the *first* copy
    had been dropped, so an arm that removed the second one imported nothing at
    all and the veto blocked the whole judgement -- which reads as the tagger
    not seeing the repetition rather than as this function not recognising it.
    Ten of the 24 repetitions the tagger wanted gone and the product kept were
    that shape: 发挥发挥, 故事故事, 鼻涕鼻涕, 表面表面. Fixing it moves filler
    clearance 0.9364 to 0.9443 and CER 0.0302 to 0.0296 with retention and the
    word-cut rate unchanged.
    """
    found: set[int] = set()
    for size in range(shortest, longest + 1):
        for start in range(len(source) - 2 * size + 1):
            if source[start : start + size] != source[start + size : start + 2 * size]:
                continue
            first = range(start, start + size)
            second = range(start + size, start + 2 * size)
            if all(index in drop for index in first):
                found.update(first)
            elif all(index in drop for index in second):
                found.update(second)
    return found


def reached(source: str, output: str) -> int:
    """How far into ``source`` the output got, matching greedily in order.

    Under the copy constraint the output is a subsequence of the source, so
    this is exact. Used to tell a deletion from an absence: a generative arm
    that emits its stop token early leaves the whole tail looking deleted, and
    on a 184-character dictation that was 100 characters of "opinion" it never
    formed. Measured by hand, the generator returned 82 of 184 characters and
    stopped mid-phrase.
    """
    pointer = 0
    for char in output:
        while pointer < len(source) and source[pointer] != char:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer


def keep_one_copy(source: str, drop: set[int], shortest: int = 2, longest: int = 5) -> set[int]:
    """A repetition loses one copy, never both.

    A repetition is one copy too many, so removing it should leave one behind.
    Taking both is content loss wearing a deletion's clothes, and the criteria
    charge it exactly what they charge a correct removal -- measured on a
    dictated note, 因为，因为上午张老师有课 came back with no 因为 at all and
    the causal link with it.

    Measured over 986 held-out transcripts: the generator alone never did this
    (0 sentences), the merged arm did it in 11 (1.12%) -- 确定 确定, 换个 换个,
    背着 背着. So it is the merge that introduced it, and the merge is where it
    is repaired.

    A repeated filler is left alone: both copies of 就是就是 are filler, and
    taking both is the vocabulary doing its job. Without that exemption this
    reads 1.52% and the examples are 其实 / 这个 / 就是 straight down the list.

    The second copy is the one restored, so what survives sits against the text
    that follows it.
    """
    kept = set(drop)
    for size in range(shortest, longest + 1):
        start = 0
        while start + 2 * size <= len(source):
            unit = source[start : start + size]
            twice = unit == source[start + size : start + 2 * size]
            if twice and all(index in kept for index in range(start, start + 2 * size)):
                if not any(word in unit for word in VOCABULARY):
                    kept -= set(range(start + size, start + 2 * size))
                start += 2 * size
            else:
                start += 1
    return kept


def whole_words(source: str, drop: set[int]) -> set[int]:
    """A word is dropped whole or not at all.

    Deletion here is per character and nothing held it to a word boundary, so
    it ate 那 out of 那边 and left 客户边, 恶 out of 恶魔 and left 魔, 生酱 out
    of 花生酱 and left 花. The criteria charge one character for each; a reader
    sees a word that does not exist. Measured on 986 held-out transcripts, 63
    of them (6.39%) carried at least one, and the arm this replaced carried the
    same 63 -- it is inherited, not introduced.

    The literature's answer is to work in words: disfluency is annotated on
    words and subword tokens inherit their parent word's label (Kumar et al.,
    ACL 2026). Re-annotating is a training round; this is the same idea as a
    constraint, and it repairs all 63.

    A cut that *contains* a filler is exempt: jieba glues 就是 to its neighbour
    in 就是说, and undoing that deletion is undoing the vocabulary's own work.

    Containment one way only. The first version also exempted a cut that was
    *inside* a filler, and that let 那 through because 那 sits inside 那个 --
    so the constraint skipped 客户那边, the single case it was written for, and
    the instrument that counts these skipped it too. Both read a number that
    excluded the defect they were about.

    **What it costs, stated plainly.** Filler clearance reads 0.9415 against
    0.9307 with this on, and CER 0.0366 against 0.0376. But no sentence gains a
    filler in the raw output -- the metric normalises punctuation away before
    counting, so a restored character can re-form a token there that a reader
    never sees. Weighed against 63 repaired words that a reader does see, and
    against this project's standing rule that damaged content beats leftover
    filler, the constraint is on. Read them yourself: 别一挪 -> 别一件件挪,
    血糖上升平 -> 血糖上升平缓, 常穿的当不穿 -> 常穿的当即不穿.
    """
    import jieba

    kept = set(drop)
    cursor = 0
    for word in jieba.cut(source):
        start, end = cursor, cursor + len(word)
        cursor = end
        inside = [index for index in range(start, end) if index in kept]
        if not inside or len(inside) == end - start:
            continue
        cut = "".join(source[index] for index in inside)
        if any(filler in cut for filler in VOCABULARY):
            continue
        kept -= set(inside)
    return kept


def merge(rows: list[dict[str, Any]], mode: str) -> str:
    source = rows[0]["source"]
    drops = [dropped(source, row["polished"]) for row in rows]
    # Past where an arm stopped, it has no opinion. Without this the veto reads
    # a truncated arm as agreeing to delete the entire tail.
    for index, row in enumerate(rows):
        stopped = reached(source, row["polished"])
        if stopped < len(source):
            drops[index] = {position for position in drops[index] if position < stopped}
    if mode == "union":
        combined = set.union(*drops)
    elif mode == "intersection":
        combined = set.intersection(*drops)
    elif mode == "veto":
        # Both kinds are taken from every arm, not just the first.
        #
        # Repetition used to be imported from arm 0 alone, as an exemption --
        # the head could not see repetition at all (0.437 against the
        # generator's 0.603), so letting it veto that work cost the whole
        # difference, and there was nothing of its own worth taking. That
        # premise died on 2026-08-16: with the repetition columns wired and
        # CS2W's human annotation in the training set, the tagger clears
        # 0.5044 of real repetition against the generator's 0.3049. The
        # asymmetry the vocabulary/repetition split was built on has inverted
        # for one of its two halves, so the import is symmetric now.
        exempt: set[int] = set()
        for drop in drops:
            exempt |= vocabulary_spans(source, drop)
            exempt |= repetition_spans(source, drop)
        combined = set.intersection(*drops) | exempt
    else:
        # The first arm whole, the rest only where they spell a filler.
        combined = set(drops[0])
        for drop in drops[1:]:
            combined |= vocabulary_spans(source, drop)
    combined = whole_words(source, keep_one_copy(source, combined))
    return tidy("".join(char for index, char in enumerate(source) if index not in combined))


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
    # The second arm, and the threshold it answers at. Unset is the shape this
    # stage had before 2026-08-16 and still the fallback: generate, tidy, done.
    #
    # Set, the two arms are merged by veto -- the head's confidence protects the
    # generator's content, and neither may veto a vocabulary filler or an exact
    # repetition. Measured on 986 held-out transcripts against the generator
    # alone: CER 0.0463 to 0.0370, filler clearance 0.8592 to 0.9443, retention
    # 0.9760 to 0.9791, invention 0.0292 to 0.0231, and words cut through 6.39%
    # of sentences to 6.59%. Better on four readings, level on the fifth.
    tagger: Path | None = None
    tagger_backbone: Path | None = None
    tagger_threshold: float = 0.5
    _model: Any = None
    _queue: Any = field(default_factory=list)
    _queue_lock: Any = None
    _ready: Any = None
    _worker: Any = None
    _tokeniser: Any = None
    _fillers: Any = None
    _tagger_head: Any = None
    _tagger_model: Any = None
    _tagger_spec: Any = None

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
        self._load_tagger()

    def _load_tagger(self) -> None:
        """The second arm, when one was configured.

        Its own backbone, not the polisher's: the head was trained against a
        backbone with three blocks tuned along with it, and reading it off the
        polish weights would be reading a probe of a model it never saw.
        """
        if self.tagger is None:
            return
        import torch

        from mindsurf_omni.service.thinker import ThinkerGenerator

        if self.tagger_backbone is None:
            raise ConfigurationError(
                "a polish tagger was named without its backbone; the head is a probe of "
                "the blocks that were tuned with it, and reading it off other weights "
                "measures nothing"
            )
        saved = torch.load(str(self.tagger), map_location=self.device, weights_only=False)
        # Same refusal the scoring scripts make. Heads written before
        # 2026-08-16 read their repetition columns off token ids; these read
        # them off characters. Same width, different meaning, no exception.
        if saved.get("repetition", 0) and saved.get("repetition_unit") != "character":
            raise ConfigurationError(
                f"the polish tagger at {self.tagger} reads its repetition columns off "
                "token ids, which this build no longer computes -- the columns would "
                "line up by width and mean something else. Retrain it"
            )
        backbone = ThinkerGenerator(
            checkpoint=self.tagger_backbone,
            tokenizer_dir=self.tokenizer_dir,
            minimind_root=self.minimind_root,
            device=self.device,
            variant=self.variant,
        )
        backbone.load()
        self._tagger_model = backbone._model.eval()  # noqa: SLF001
        head = torch.nn.Linear(saved["hidden"], 2).to(self.device)
        head.load_state_dict(saved["state_dict"])
        self._tagger_head = head.eval()
        self._tagger_spec = (saved["lookahead"], saved.get("repetition", 0))

    def _tag_all(self, pieces: list[str]) -> dict[str, str] | None:
        """Every piece through the head, or None when no head is configured.

        None rather than an empty dict, so the caller cannot mistake "no second
        arm" for "the second arm deleted nothing".
        """
        if self._tagger_head is None:
            return None
        return {piece: self._tagged(piece) for piece in pieces}

    def _tagged(self, piece: str) -> str:
        """What the head alone would keep of this piece."""
        import torch

        from mindsurf_omni.service.tagger import features as tag_features
        from mindsurf_omni.service.tagger import token_spans

        lookahead, repetition = self._tagger_spec
        ids, spans = token_spans(self._tokeniser, piece)
        if not ids:
            return piece
        with torch.no_grad():
            matrix = tag_features(
                self._tagger_model, ids, torch, self.device, lookahead, repetition, piece, spans
            )
            keep = torch.softmax(self._tagger_head(matrix), dim=-1)[:, 1]
        drop = {
            position
            for (start, end), probability in zip(spans, keep.tolist(), strict=True)
            if probability >= self.tagger_threshold
            for position in range(start, min(end, len(piece)))
        }
        return "".join(char for index, char in enumerate(piece) if index not in drop)

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
        # The head runs on a thread, for the reason `_transcribe` does: it is a
        # blocking GPU call and this is the service's only event loop, so inline
        # it stalls every other connection for the length of the forward pass.
        # Measured with eight concurrent dictations, /health answered in 90 ms
        # at worst inline against 45 ms on a thread. Neither failed -- the
        # effect is small at this size and grows with the piece.
        #
        # Gathered rather than awaited in turn because the arms do not depend on
        # each other; that buys 12 ms of the median and no more. The tagger's
        # 109 ms is compute, not scheduling: one card, both arms on it, so
        # "concurrent" is a queue. Measuring said so, the guess did not.
        import asyncio

        decoded, tagged = await asyncio.gather(
            self._decode(wanted),
            asyncio.to_thread(self._tag_all, wanted),
        )
        answers = dict(zip(wanted, decoded, strict=True))

        out = []
        for piece in pieces:
            if piece not in answers:
                out.append(piece)
                continue
            answer = answers[piece]
            written = answer if consumed(piece, answer) >= FLOOR else piece
            if tagged is not None:
                # Veto, not union: the head's confidence protects the
                # generator's content rather than removing more of it, and
                # neither arm may veto a vocabulary filler or an exact
                # repetition. Merged per piece because the pieces are what each
                # arm was given, and a merge across a boundary would align two
                # different sentences.
                written = merge(
                    [
                        {"source": piece, "polished": written},
                        {"source": piece, "polished": tagged[piece]},
                    ],
                    "veto",
                )
            out.append(written)
        # Tidied over the joined text, not per piece: a piece boundary is a
        # sentence boundary, so a particle stranded at the start of one is only
        # visible once its neighbour is in front of it.
        return tidy("".join(out))

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
