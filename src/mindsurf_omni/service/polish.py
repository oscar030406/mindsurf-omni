"""Delete spoken filler from a transcript without rewriting it.

The decode is restricted to a subsequence of the transcript, so the output can
only be the input with characters removed. That buys three properties and costs
one: nothing can be invented, nothing can be substituted (它 heard as 他 stays
他), and a single step cannot jump a clause. Homophone repair is out of reach,
which the floor round measured at 0.6% of the edits and gave up on purpose.

The instruction is fixed here rather than passed in: the model was trained on
these words.
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

# How the recogniser actually spells the above: SenseVoice writes 呃 as
# 饿 恶 啊 鄂 扼 and 嗯 as 恩 摁 温, so a door matching the vocabulary literally
# misses roughly one occurrence in nine.
#
# Only spellings that never occur in the corpus as ordinary words are here.
# 温, 饿 and 啊 do occur -- 水温, 是不是饿了 -- and a door onto those is a door
# onto content.
RECOGNISED_FILLERS = ("摁", "鄂", "唉", "哎", "呐")

# The English half. Only the filled pauses -- the sounds nobody means, which
# every disfluency scheme marks the same way. The discourse markers (like, well,
# actually, right, so, I mean, you know) are the same problem as the Chinese
# double-duty words and are deliberately not here: 我 like 这个 layout and
# it's, like, twice as fast are the same four letters, and the corpus that would
# settle where the line falls is one we do not have yet.
#
# Cased pairs rather than a case fold, because the match is a substring test
# against the transcript: "Um" starts a sentence and "um" sits inside one, while
# a fold would also match the "um" inside "number".
# Interjections that are filler wherever they appear. Measured on CS2W's human
# annotation: 嗯 is deleted 1.000 of the time and 呃 0.956. The double-duty
# words live in DOUBLE_DUTY and are not these -- 那个 is 0.250.
ALWAYS_FILLER = frozenset("嗯呃唔哦喔诶欸呐额啊")

ENGLISH_FILLERS = ("um", "Um", "uh", "Uh", "er", "Er", "erm", "Erm", "uhh", "Uhh")

# Where an English filler needs a boundary to be one. Substring matching finds
# "um" inside "number" and "uh" inside "though"; these are the characters that
# may sit beside a real one.
_LATIN_EDGE = frozenset(" ,.!?;:\n\t\"()[]'") | {""}


# Where a sentence ends, for splitting a long dictation into pieces the model
# was actually trained on. Commas are not here on purpose: a clause is not a
# sentence, and cutting at every comma would take away the context the model
# needs to tell a filler 就是 from a copula one.
#
# The Latin marks need whitespace after them and the CJK ones must not: a full
# stop with no space is a decimal point or an abbreviation, and splitting 3.5 or
# Mr. Chen hands the model half a token. Without the Latin half at all, a
# 388-character English dictation came back as one piece against a TRAINED_LENGTH
# of 160 -- two and a half times the longest input the model has ever seen.
_SENTENCE_END = re.compile(r"[。！？；\n]+|[.!?]+(?=[\s\u3000]|$)")

# How much of a piece the output has to have consumed to be trusted. Not tuned:
# an output that reached less than nine tenths of its input has dropped a
# clause, and dropping a clause is the failure this floor exists to refuse.
FLOOR = 0.90

PUNCTUATION = set("，。！？；：、")

# The same job for Latin text. Kept apart from PUNCTUATION because the rules
# above it are about Chinese shapes; this one only has to catch the mark a
# deleted English filler leaves standing.
_ANY_MARK = PUNCTUATION | set(",;:.!?")

# Particles that never open a Chinese sentence. 啊 and 呀 are deliberately not
# here -- "啊，太好了" is ordinary, and deleting that 啊 would be deleting content.
STRANDED_PARTICLES = set("吧呢嘛")

# The longest text the model was trained on: the pool that built the pairs
# dropped anything past 160 characters. Above it, consecutive sentences are
# grouped up to this length rather than sent one per call, because a lone
# sentence is also out of distribution.
TRAINED_LENGTH = 160

# Spellings that mean the model might have work to do here. Wider than the
# decoder's door on purpose: the door decides whether a span may be deleted, so
# a wrong entry costs content, while this only decides whether the model is
# called, so a wrong entry costs one decode.
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


# Steps a decode may take beyond the length of its input. Marks it inserts have
# no source character to consume, and the stop token costs one more.
INSERT_MARGIN = 32


def decode_budget(longest_source: int, ceiling: int) -> int:
    """How many steps a copy-constrained decode may take.

    The output is a subsequence of the input, so the input's own length is the
    real bound and a fixed ceiling is a length limit on the whole stage wearing
    a safety cap's clothes. Kept as a floor rather than replaced, so a short
    piece still gets the room it always had.
    """
    return max(ceiling, longest_source + INSERT_MARGIN)


def worth_polishing(piece: str, longest_repeat: int = 5) -> bool:
    """Whether this piece holds anything the stage could remove.

    Filler word or an exact adjacent repetition. Nothing else is in scope --
    substitution was given up on purpose (DECISIONS §16) and punctuation is the
    recogniser's to write.
    """
    if any(word in piece for word in _WORTH_A_LOOK):
        return True
    if english_filler_at(piece) or repeated_word(piece):
        return True
    # From one character, unlike the merge rule's exemption. There the cost of
    # calling 天天 a repetition is keeping a deletion; here it is one decode,
    # while missing 我我想问一下 means never removing it at all.
    #
    # Only where both halves hold a CJK character. A doubled Latin letter is
    # spelling, not a stutter, and every English sentence has one: basically,
    # still, all, been. Before this test the trigger fired on ll and missed
    # So um I think, so which English sentences reached the model was decided
    # by their spelling.
    return any(
        piece[start : start + size] == piece[start + size : start + 2 * size]
        and any(_is_cjk(char) for char in piece[start : start + size])
        for size in range(1, longest_repeat + 1)
        for start in range(len(piece) - 2 * size + 1)
    )


def _is_cjk(char: str) -> bool:
    return "㐀" <= char <= "鿿" or "豈" <= char <= "﫿"


def english_filler_at(piece: str) -> bool:
    """Whether a filled pause stands here as a word rather than inside one.

    Substring alone finds the um in number and the uh in though, and every one
    of those would be a decode the stage does not need.
    """
    for word in ENGLISH_FILLERS:
        start = piece.find(word)
        while start != -1:
            before = piece[start - 1] if start else ""
            after = piece[start + len(word) : start + len(word) + 1]
            if before in _LATIN_EDGE and after in _LATIN_EDGE:
                return True
            start = piece.find(word, start + 1)
    return False


def repeated_word(piece: str, longest: int = 3) -> bool:
    """Whether a run of Latin words is said twice over, however it is capitalised.

    The character-level test above cannot see these: The the differs in case,
    Can you can you is two words rather than one, and both are ordinary spoken
    stumbles that reached the model only when some other trigger fired.
    """
    words = [word.lower() for word in re.split(r"[^A-Za-z']+", piece) if word]
    return any(
        words[start : start + size] == words[start + size : start + 2 * size]
        for size in range(1, longest + 1)
        for start in range(len(words) - 2 * size + 1)
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
    """Delete a surviving bridging filler together with its mid-sentence mark.

    The mirror of ``tidy``'s first shape. There a deleted filler stranded its
    punctuation; here an undeleted one keeps a question mark mid-sentence
    ("记了下来。对吧？训练好之后"), which a listener meets as a wrong sentence
    boundary rather than an extra word.

    Mid-sentence only: a mark at the end is ending the sentence it should end.
    Bridging fillers only, because widening it to every leftover filler is a
    lexicon overruling the model. Deleting only, so the copy constraint holds.
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

    Three shapes, read off the output rather than traced back to the filler they
    came from: a mid-sentence mark still attached to its filler
    (``drop_bridging_with_mark``, first because it can double up punctuation),
    punctuation stranded by a deleted filler ("彩塑，？特别震撼"), and a particle
    stranded at the start ("吧这个真的挺好的").

    None of them costs anything on the acceptance numbers, which normalise
    punctuation away. They are visible to a reader and not to the metric.
    """
    text = drop_bridging_with_mark(text)
    out: list[str] = []
    for char in text:
        # ASCII marks count too, and only since English joined: Um, then I …
        # left its comma behind as . , then I …. Whitespace does not separate
        # them -- the comma a deleted filler strands has a space in front of it.
        behind = "".join(out).rstrip(" ")
        if char in _ANY_MARK and (not behind or behind[-1] in _ANY_MARK):
            continue
        if char in PUNCTUATION and (not out or out[-1] in PUNCTUATION):
            continue
        if char in STRANDED_PARTICLES and (not out or out[-1] in PUNCTUATION):
            continue
        # A fourth shape, and the only one Chinese never showed: removing a word
        # from between two spaces leaves both of them. 这个 deadline 我们 um …
        # came back as 这个 deadline  um …, a gap a reader sees immediately.
        if char == " " and out and out[-1] == " ":
            continue
        out.append(char)
    return "".join(out)


# --- Merging two arms. The product runs the merge, so these live here and the
# scoring script imports them: one copy, and no drift between the offline arms
# and the served text. ------------------------------------------------------

# Longest first, so 你知道吧 is matched before 你知道 would be.
VOCABULARY = tuple(sorted((*LEADING_FILLERS, *BRIDGING_FILLERS), key=len, reverse=True))

# The words above that a human-marked corpus says are usually content, not
# filler. The pairs builder injects all nine of LEADING_FILLERS at one
# probability, so in the training data "delete this" and "this spelling" are the
# same label and both arms learned to delete the spelling. Membership here is
# decided by the corpus, not by us: run ``scripts/measure_content_words.py
# --distribution`` to recount it against CS2W. The two words that stay out of
# this list are the ones that corpus marks as filler nearly every time (嗯 78/78,
# 呃 43/45).
#
# The single-character 那 and 就 were added after the first look at real
# spontaneous speech. On the injected hold-out they never appear -- the builder
# plants 那个 and 就是 -- so nothing in the release numbers could see them; on
# 120 real dictations they were the two largest classes of content this stage
# deleted (那 22 times, 就 15). CS2W marks them filler 0.275 and 0.203 of the
# time, against 0.250 for 那个 and 0.190 for 就是 already on the list: by the
# criterion that chose the original seven they always belonged here, and the
# original seven were really just the nine words the injector knew.
# Longest first: 那个 has to be considered before 那, or a wholly deleted 那个
# gets its 那 handed back and leaves a bare 个 behind.
DOUBLE_DUTY = (
    "我觉得",
    "那个",
    "这个",
    "就是",
    "然后",
    "反正",
    "其实",
    "那种",
    "那些",
    "那",
    "就",
    "对",
)


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
    """Indices of deletions that remove one copy of an adjacent repetition.

    Exempt from the veto, because a per-token head cannot represent the judgement
    at all: it reads one position at a time and 时间时间 looks like two ordinary
    words.

    Two characters and up. One is not a repetition in Chinese: 今天天气 holds 天天
    and 看看 is a word.
    """
    found: set[int] = set()
    for copy in repetition_copies(source, drop, shortest, longest):
        found |= copy
    return found


def repetition_copies(
    source: str, drop: set[int], shortest: int = 2, longest: int = 5
) -> list[set[int]]:
    """The same judgement, one copy per entry rather than flattened.

    ``repetition_spans`` answers which characters may not be vetoed; this answers
    which characters go together, which is what a stage that may take some of them
    and leave the rest needs.
    """
    copies: list[set[int]] = []
    for size in range(shortest, longest + 1):
        for start in range(len(source) - 2 * size + 1):
            if source[start : start + size] != source[start + size : start + 2 * size]:
                continue
            first = set(range(start, start + size))
            second = set(range(start + size, start + 2 * size))
            if first <= drop:
                copies.append(first)
            elif second <= drop:
                copies.append(second)
    return copies


def stutter(unit: str) -> bool:
    """Whether a repeated unit is a stumble rather than something said twice.

    Three shapes are not stumbles, and dropping a copy of them changes the
    sentence: A-not-A questions (是不是不舒服 would become 是不舒服), the written
    是否, and digits or letters (2020法则 would become 20法则).
    """
    if any(char in unit for char in "不否"):
        return False
    return not all(char.isascii() and char.isalnum() for char in unit)


def reached(source: str, output: str) -> int:
    """How far into ``source`` the output got, matching greedily in order.

    Exact under the copy constraint. Separates a deletion from an absence: an arm
    that emits its stop token early leaves the whole untouched tail looking
    deleted.
    """
    pointer = 0
    for char in output:
        while pointer < len(source) and source[pointer] != char:
            pointer += 1
        pointer = min(pointer + 1, len(source))
    return pointer


def keep_one_copy(source: str, drop: set[int], shortest: int = 2, longest: int = 5) -> set[int]:
    """A repetition loses one copy, never both.

    Taking both is content loss wearing a deletion's clothes, and the acceptance
    numbers charge it what they charge a correct removal. Neither arm does this on
    its own; the merge introduces it, so the merge repairs it.

    A repeated filler is left alone, both copies of 就是就是 being filler. The
    second copy is the one restored, so what survives sits against the text that
    follows it.
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

    Per-character deletion ate 那 out of 那边 and left 客户边. jieba's segmentation
    is the boundary.

    A cut that *contains* a filler is exempt, since jieba glues 就是 to its
    neighbour in 就是说. Containment one way only: also exempting a cut *inside* a
    filler lets 那 through, because 那 sits inside 那个.

    It costs filler clearance and buys back words a reader can see, which is this
    project's standing trade -- damaged content beats leftover filler.
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


_Word = tuple[int, int, str]


def _gap_is_only_space(source: str, left: _Word, right: _Word) -> bool:
    return source[left[1] : right[0]].strip(" ") == ""


# English words that stand next to themselves on purpose. had had is the past
# perfect, that that opens a clause, very very and so so are emphasis. Short,
# and short on purpose: every entry here is a stutter this stage will now miss,
# and the list only holds the ones where the doubling is grammar rather than a
# stumble.
DOUBLES_ON_PURPOSE = frozenset({"had", "that", "very", "so", "no"})


def _says_it_twice_on_purpose(raw: list[str], words: list[_Word], index: int, size: int) -> bool:
    """Whether this repetition is English rather than a stumble.

    Two shapes. A capitalised word away from the start of a sentence is a name,
    and Pago Pago, Baden Baden and New York, New York are places somebody said
    once -- except I, which is capitalised in every position and so carries no
    information here, while I I is the commonest English stutter there is. The
    rest is grammar: had had, that that, very very.

    Same trade as the Chinese double-duty words, and settled the same way --
    content is the expensive side, so a doubling that might be meant is left
    alone. What it costs is the stutters that happen to use these words.
    """
    if size == 1 and words[index][2] in DOUBLES_ON_PURPOSE:
        return True
    return any(
        raw[place][:1].isupper() and place > 0 and raw[place] != "I"
        for place in range(index, index + 2 * size)
    )


def english_disfluency(source: str, already: set[int] | None = None) -> set[int]:
    """Indices of the Latin disfluency a rule can be sure about.

    A rule and not the model, because the model cannot do this one. It was
    trained on Chinese, and on the 25-sentence English probe it left every
    filled pause in place while deleting the content word So -- the one early
    success (like, you know) does not survive a second sample.

    Two shapes only, and both are the shapes no annotation scheme disagrees
    about: a filled pause standing as its own word, and a run of words said
    twice over. What is deliberately absent is the discourse markers -- like,
    well, actually, right, so, I mean, you know. They are the English half of
    the double-duty problem (I like the layout against it's, like, twice as
    fast) and the corpus that would settle where the line falls is one we do
    not have. Content is the expensive side there too.

    Deletion only, so the copy constraint the stage rests on still holds: no
    character comes out that did not go in.
    """
    drop: set[int] = set()
    for word in ENGLISH_FILLERS:
        start = source.find(word)
        while start != -1:
            stop = start + len(word)
            before = source[start - 1] if start else ""
            after = source[stop : stop + 1]
            if before in _LATIN_EDGE and after in _LATIN_EDGE:
                drop |= set(range(start, stop))
            start = source.find(word, start + 1)

    words = [
        (match.start(), match.end(), match.group().lower())
        for match in re.finditer(r"[A-Za-z']+", source)
    ]
    raw = [source[start:stop] for start, stop, _ in words]
    taken: set[int] = set()
    for size in range(3, 0, -1):
        for index in range(len(words) - 2 * size + 1):
            first, second = words[index : index + size], words[index + size : index + 2 * size]
            if [w for _, _, w in first] != [w for _, _, w in second]:
                continue
            # Adjacent in the source, not merely adjacent in the list of Latin
            # words. Skipping over the Chinese between them made 16G…500G read
            # as a repeated G and 4G 的几倍 lose its G; the dialogue tags in
            # A 哦…B 当然 went the same way. Only spaces may sit in a stumble.
            if not all(
                _gap_is_only_space(source, run[index_], run[index_ + 1])
                for run in (first + second,)
                for index_ in range(len(run) - 1)
            ) or not _gap_is_only_space(source, first[-1], second[0]):
                continue
            # The second copy, not the first: The the migration keeps its
            # capital that way, and recasing is not available to a stage whose
            # output has to be a subsequence of its input.
            if _says_it_twice_on_purpose(raw, words, index, size):
                continue
            span = set(range(second[0][0], second[-1][1]))
            whole = set(range(first[0][0], second[-1][1]))
            # One copy, and only if nobody has taken one already. The arms had
            # deleted the first to of We need to to fix; adding the second on
            # top of that left We need fix.
            if span & taken or (already and already & whole):
                continue
            taken |= span
            drop |= span
    return drop


def spaces_follow_their_words(source: str, drop: set[int]) -> set[int]:
    """A space may only go if the word beside it went.

    In Chinese there is nothing between words, so this never came up. In Latin
    script the space *is* the boundary: deleting one fuses two words into a
    thing nobody wrote. Measured, ``We need to um double check`` came back as
    ``We need toumdouble check`` -- the two spaces around a filler were dropped
    while the filler itself stayed.

    The rule is not "never delete a space": removing a word has to remove one of
    its spaces too, or every deletion leaves a double gap behind. So a space
    goes exactly when a token touching it went whole.
    """
    import jieba

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for word in jieba.cut(source):
        spans.append((cursor, cursor + len(word), word))
        cursor += len(word)

    gone = [
        index
        for index, (start, end, word) in enumerate(spans)
        if word.strip() and all(place in drop for place in range(start, end))
    ]
    kept = set(drop)
    for index, (start, end, word) in enumerate(spans):
        if word.strip() or not (set(range(start, end)) & drop):
            continue
        if not {index - 1, index + 1} & set(gone):
            kept -= set(range(start, end))

    # And the reverse for the seam a deleted Latin word leaves between two
    # Chinese ones: 我们 um 可能 has a space on each side of um because um is
    # Latin, and taking um out leaves 我们 可能, a gap Chinese never has. Both
    # spaces go, not one.
    for index in gone:
        left, right = (
            spans[index - 1] if index else None,
            (spans[index + 1] if index + 1 < len(spans) else None),
        )
        if not (left and right) or left[2] != " " or right[2] != " ":
            continue
        outer_left = spans[index - 2][2] if index >= 2 else ""
        outer_right = spans[index + 2][2] if index + 2 < len(spans) else ""
        if outer_left and outer_right and _is_cjk(outer_left[-1]) and _is_cjk(outer_right[0]):
            kept |= set(range(left[0], left[1])) | set(range(right[0], right[1]))
    return kept


def periodic_runs(source: str, longest: int = 5) -> list[tuple[int, int, int]]:
    """Maximal runs of one repeated unit, as ``(start, end, period)``.

    Shortest period wins, and the leftover tail is included: a run whose length is
    not a multiple of its period is where a deletion can land off-phase.
    """
    runs: list[tuple[int, int, int]] = []
    for period in range(1, longest + 1):
        start = 0
        while start + 2 * period <= len(source):
            if source[start : start + period] != source[start + period : start + 2 * period]:
                start += 1
                continue
            end = start + 2 * period
            while end < len(source) and source[end] == source[end - period]:
                end += 1
            runs.append((start, end, period))
            start = end - period + 1
    kept: list[tuple[int, int, int]] = []
    for start, end, period in sorted(runs, key=lambda run: (run[0], run[2])):
        if any(
            outer_start <= start and end <= outer_end and outer_period <= period
            for outer_start, outer_end, outer_period in kept
        ):
            continue
        kept.append((start, end, period))
    return kept


def snap_periods(source: str, drop: set[int]) -> set[int]:
    """Round a deletion inside a repeated run to a whole number of copies.

    Runs on the settled deletion rather than on either arm, because every stage
    above can land the cut off-phase and the damage is the same whichever did.
    Inside a run of period k the copies are interchangeable, so difflib is free to
    record the deletion at any of them while jieba only knows the leftmost. Half a
    copy is not a smaller edit than a whole one, it is a different and wrong one.

    Two runs are left alone: period one is ordinary reduplication (慢慢看, 试试),
    and a unit spelling a filler belongs to ``keep_one_copy``, which takes both
    copies of 就是就是 on purpose.
    """
    settled = set(drop)
    for start, end, period in periodic_runs(source):
        if period < 2:
            continue
        unit = source[start : start + period]
        if any(word in unit for word in VOCABULARY):
            continue
        inside = [index for index in range(start, end) if index in settled]
        if not inside or len(inside) == end - start:
            continue
        copies = (end - start) // period
        leftover = source[start + period * copies : end]
        survives = "".join(source[i] for i in range(start, end) if i not in settled)
        body = survives[: len(survives) - len(leftover)] if leftover else survives
        already = (
            survives.endswith(leftover)
            and body
            and len(body) % period == 0
            and body == unit * (len(body) // period)
        )
        if already:
            continue
        # Never take the run down to nothing: deleting every copy is a judgement
        # the stages above make on purpose, and this one only decides where an
        # existing cut sits.
        taken = min(max(1, round(len(inside) / period)), copies - 1)
        settled -= set(range(start, end))
        settled |= set(range(start, start + taken * period))
    return settled


# jieba's tags for a plain noun. A repeated noun is a stumble; a repeated verb,
# measure or adverb is Chinese grammar -- see ``duplicate_words``.
NOUN_TAGS = frozenset({"n", "ns", "nt", "nz", "an", "ng"})


def duplicate_words(source: str, drop: set[int]) -> set[int]:
    """Take one copy of a repeated whole word, whether or not an arm asked.

    The only rule here that proposes rather than narrows, because the arms do not
    see most of these: 花椒花椒, 粽子粽子, 利率利率.

    Nouns only, and that is the whole safety argument. Repeating a noun is a
    stumble; repeating anything else is usually grammar. Chinese reduplicates
    verbs to soften them (研究研究, 商量商量), measures to mean one-by-one
    (一个一个, 一步一步, 一年一年), and adverbs to intensify (特别特别, 真的真的) --
    all of them adjacent, all of them identical, none of them a stumble. Deleting
    a copy there does not tidy the sentence, it changes what it says. jieba's
    part-of-speech tagger separates the two well enough: the stumbles this was
    written for tag as plain nouns and every reduplication above tags as
    something else.

    The unit also has to be one whole token. That is what puts 是不是/不/舒服 and
    2020/法则 out of reach.

    A unit spelling a filler is left to ``keep_one_copy``.
    """
    import logging

    import jieba
    import jieba.posseg

    jieba.setLogLevel(logging.ERROR)
    nominal: dict[tuple[int, int], bool] = {}
    cursor = 0
    for token in jieba.posseg.cut(source):
        width = len(token.word)
        nominal[(cursor, cursor + width)] = token.flag in NOUN_TAGS
        cursor += width

    taken = set(drop)
    for size in range(2, 6):
        start = 0
        while start + 2 * size <= len(source):
            unit = source[start : start + size]
            if (
                nominal.get((start, start + size))
                and nominal.get((start + size, start + 2 * size))
                and unit == source[start + size : start + 2 * size]
                and not any(word in unit for word in VOCABULARY)
                and not set(range(start, start + 2 * size)) & taken
            ):
                taken |= set(range(start, start + size))
                start += 2 * size
            else:
                start += 1
    return taken


# The characters that flip a clause on their own. Kept apart from every other
# protection here because what a deleted one costs is not a share of the
# sentence -- it is the opposite of the sentence.
NEGATIONS = frozenset("不没别无非莫勿甭")

# Where these characters are part of a word instead of a negation. 特别 分别
# 区别 are not 别; 非常 非得 are not 非; 沉没 淹没 are not 没. Written as the
# character in front, which is what decides it.
NOT_A_NEGATION_AFTER = {
    "没": frozenset("沉淹埋出湮"),
    "别": frozenset("特分差个区级性告送派类识"),
    "非": frozenset("除莫并"),
}
# And where the character after settles it: 非常 非得 非要 are adverbs.
NOT_A_NEGATION_BEFORE = {"非": frozenset("常得要凡")}


def negation_positions(source: str) -> set[int]:
    """Indices of ``source`` where a character is genuinely negating."""
    out = set()
    for index, char in enumerate(source):
        if char not in NEGATIONS:
            continue
        before = source[index - 1] if index else ""
        after = source[index + 1] if index + 1 < len(source) else ""
        if before and before in NOT_A_NEGATION_AFTER.get(char, ()):
            continue
        if after and after in NOT_A_NEGATION_BEFORE.get(char, ()):
            continue
        out.add(index)
    return out


def negations_kept(source: str, drop: set[int]) -> set[int]:
    """The part of ``drop`` that would take a clause's last negation with it.

    Driving the service on real conversation returned 对不对？as 对对 and
    人也没了 as 人也了. Both are one character out of a dozen, which every
    reading we take scores as a rounding error and a reader scores as the
    sentence saying the opposite thing.

    A repeated negation is still fair game -- 不能不能 has a copy to spare, and
    the stage that exists to remove stutters must keep working. So the rule is
    not "never delete a negation", it is "leave the clause one".
    """
    if not drop:
        return set()
    negations = negation_positions(source)
    if not negations & drop:
        return set()
    give_back: set[int] = set()
    start = 0
    for stop in [i for i, c in enumerate(source) if c in PUNCTUATION] + [len(source)]:
        here = {i for i in negations if start <= i < stop}
        start = stop + 1
        going = here & drop
        # Something negating still survives here, so the deletion is a stutter's
        # spare copy and not the clause's meaning.
        if not going or len(going) < len(here):
            continue
        give_back.add(min(going))
    return give_back


def _word_edges(source: str) -> set[int]:
    """Offsets where the segmenter starts a word.

    ``whole_words`` already treats jieba's segmentation as the boundary for
    deletions; this is the same boundary on the other side of the ledger. Without
    it a single-character entry reaches inside a longer word: 血糖上升，反正相对
    平缓 came back as 血糖上升对平缓, because 对 sitting inside 相对 was handed
    back on its own. 那 in 刹那 and 就 in 成就 are the same shape and had simply
    never been hit.

    Edges rather than whole tokens, because a list entry is not always one of
    them: jieba cuts 我觉得 into 我 and 觉得, and requiring a single token would
    refuse the longest entry on the list. What is being ruled out is a span the
    segmenter cuts *across* -- the same test the hotword door uses.
    """
    import jieba

    edges = {0}
    cursor = 0
    for word in jieba.cut(source):
        cursor += len(word)
        edges.add(cursor)
    return edges


def content_words(source: str, drop: set[int]) -> set[int]:
    """The part of ``drop`` that spells a double-duty word standing in a content slot.

    This is the only stage that overrules every arm at once, so each of its
    four conditions earns its place; dropping any one of them was measured to
    make the output worse, and ``tests/test_merge_polish_arms.py`` holds the
    shape that goes wrong.

    Dropped whole. Half a word is a different defect and ``whole_words`` owns it.

    Something follows it, and that something survived. An arm that ate the word
    together with the content behind it has already lost the content; handing
    the word back then wedges filler into the hole and reads worse than the
    plain over-deletion -- 淘米反正轻轻搓两遍 -> 淘米反正搓两遍, with 轻轻
    still gone.

    Neither neighbour is another copy of it, whole or half. A stutter is one
    word said twice and one copy is meant to go; which copy the alignment
    recorded depends only on whether there was text in front of it, so both
    sides are checked. Half counts because people stutter by syllable -- 这这个,
    就就是, 然然后 -- and the whole-word test alone let those through, so the
    stage that exists to take repetitions out wrote one in.

    Its clause has something else left in it. A clause that is nothing but
    filler is deleted whole on purpose, and a word cannot be the content of a
    clause with no content: 嗯，那个。会议改到三点了。 must not come back as
    那个。会议改到三点了。

    Position is deliberately not consulted beyond that, and the reason is a
    trade rather than an absence. The signal is real: on human marks these words
    are filler 0.26 of the time at the start of a clause against 0.13 inside
    one. Wired in, it deletes 32 more characters a person marked and 98 a person
    kept -- three wrong for one right -- and it costs the commonest content use
    there is, the bare demonstrative 我要这个。. So the rule is refused for
    losing the exchange, not for having nothing behind it.

    The seven are not alike, and one of them is a coin. On the same marks 然后
    is filler 16 of 36 times, the highest of the seven, and it is the only one
    whose saved wrong deletions do not outnumber the right deletions it gives up
    (0.83, against 4.2 and higher for the rest). Dropping it from DOUBLE_DUTY
    moves precision 0.4879 to 0.4904 and recall 0.2934 to 0.3079 while the
    headline count of over-deleted characters goes 509 to 529. It is kept
    because content is the expensive side, and that is a judgement, not a
    measurement.
    """
    give_back: set[int] = set()
    handled: set[int] = set()
    edges = _word_edges(source)
    for word in DOUBLE_DUTY:
        width = len(word)
        start = source.find(word)
        while start != -1:
            end = start + width
            if set(range(start, end)) & handled:
                start = source.find(word, start + 1)
                continue
            # Claimed whether or not it is handed back. A shorter entry must not
            # reach inside a longer one at all: 就 sitting inside 就是就是就是
            # would restore a copy the longer word deliberately refused, and the
            # result is half a word.
            handled |= set(range(start, end))
            if (
                all(index in drop for index in range(start, end))
                and end < len(source)
                and end not in drop
                and source[end : end + width] != word
                and source[max(0, start - width) : start] != word
                and start in edges
                and end in edges
                and not _a_fragment_of_it_survives_to_the_left(source, drop, word, start)
                and _clause_has_other_survivors(source, drop, start, end)
            ):
                give_back |= set(range(start, end))
            start = source.find(word, start + 1)
    return give_back


def _a_fragment_of_it_survives_to_the_left(
    source: str, drop: set[int], word: str, start: int
) -> bool:
    """Whether the characters before this span are a half-said run at it.

    The whole-word guard beside this one catches 这个这个; people stutter by
    syllable, so what actually turns up is 这这个, 就就是, 然然后, 我我觉得 --
    the left neighbour is a prefix of the word, not the word. Handing the span
    back there writes the stutter into the output: an arm that had turned
    现这这个地步 into 现这地步 got 现这这个地步 instead, and this stage exists to
    take repetitions out, not to put them in. Neither reading is right; the one
    the user sees is the doubled one.

    Only when the fragment is still there. If the arm ate it too, nothing is
    left stranded and the span comes back as it should.
    """
    return any(
        source[start - size : start] == word[:size] and not set(range(start - size, start)) & drop
        for size in range(1, len(word))
        if start - size >= 0
    )


def _clause_has_other_survivors(source: str, drop: set[int], start: int, end: int) -> bool:
    """Whether anything besides ``source[start:end]`` survives between its pauses."""
    left = start
    while left > 0 and source[left - 1] not in PUNCTUATION:
        left -= 1
    right = end
    while right < len(source) and source[right] not in PUNCTUATION:
        right += 1
    return any(index not in drop for index in range(left, right) if not start <= index < end)


def keep_content(source: str, output: str) -> str:
    """``output``, with any double-duty word it deleted from a content slot put back.

    The single-arm path's share of ``content_words``. Unlike ``merge``, which
    renders its answer from the source either way, this one is editing text an
    arm already wrote, so it hands that text straight back unless it is sure
    what it is doing to it: nothing to give back, or an alignment that does not
    reproduce the arm's own output, and the arm's text goes through untouched.

    The second guard is not theoretical. ``dropped`` reads difflib's blocks,
    and difflib does not promise the longest common subsequence: 我一手机 out of
    我这个手机一是一个五G手机 aligns as 我 + 手机, losing the 一. Re-rendering
    from that drop set would quietly return a different sentence than the arm
    wrote, which is a worse failure than leaving 这个 deleted.
    """
    drop = dropped(source, output)
    latin = english_disfluency(source, set() if not any(_is_cjk(c) for c in source) else drop)
    give_back = content_words(source, drop) | negations_kept(source, drop)
    if not give_back and not latin:
        return output
    kept = "".join(char for index, char in enumerate(source) if index not in drop)
    if kept != output:
        return output
    settled = (set() if not any(_is_cjk(char) for char in source) else drop - give_back) | latin
    settled = spaces_follow_their_words(source, settled)
    return tidy("".join(char for index, char in enumerate(source) if index not in settled))


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
        # Both kinds are taken from every arm. Repetition used to come from arm
        # 0 alone, on the premise that the head could not see it at all; once the
        # repetition columns were wired that inverted, so the import is symmetric.
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
    # One copy of a stutter is protected from the word constraint, which cannot
    # tell it is looking at one: 土长土长期 has an arm asking for 土长 and jieba
    # reading 土/长期, so the constraint would hand the copy back and leave the
    # stutter in the text.
    protected: set[int] = set()
    for drop in drops:
        for copy in repetition_copies(source, drop):
            if stutter("".join(source[index] for index in sorted(copy))):
                protected |= copy
    settled = keep_one_copy(source, combined)
    combined = whole_words(source, settled - protected) | (settled & protected)
    combined = spaces_follow_their_words(source, combined)
    combined = snap_periods(source, combined)
    combined = duplicate_words(source, combined)
    # Last, on the settled deletion rather than on either arm. Both arms were
    # trained on the same lexicon, so they delete 这个 for the same wrong reason
    # and the veto reads that agreement as evidence -- there is no stage above
    # this one that can tell the difference.
    combined -= content_words(source, combined)
    # After the content give-back, not before: content_words can hand a word
    # back that carries the clause's only negation with it, and asking this
    # question of a drop set that is still moving answers about the wrong set.
    combined -= negations_kept(source, combined)
    # A piece with no Chinese in it is the rule's alone. The arms were trained
    # on Chinese and it shows: over 25 English probes they left every filled
    # pause standing and deleted the content word So. Keeping their opinion
    # there means keeping a deletion nobody can defend.
    if not any(_is_cjk(char) for char in source):
        combined = set()
    combined |= english_disfluency(source, combined)
    combined = spaces_follow_their_words(source, combined)
    return tidy("".join(char for index, char in enumerate(source) if index not in combined))


def nothing_but_filler(text: str) -> bool:
    """Whether the whole text is interjections and marks and nothing else.

    Deliberately only the interjections that are always filler -- 嗯 呃 啊 and
    their kin, which CS2W's annotators delete 95-100% of the time. The
    double-duty words are not here: 那个 alone is a real answer to "which one",
    and this decides whether the user gets an empty text box.
    """
    return bool(text.strip()) and all(
        char in ALWAYS_FILLER or char in _ANY_MARK or char.isspace() for char in text
    )


def has_content(text: str) -> bool:
    """Whether anything is left that a person would call words.

    Punctuation-only counts as empty: a text box holding a lone 。 is the same
    failure as one holding nothing.
    """
    return any("一" <= char <= "鿿" or char.isalnum() for char in text)


def consumed(source: str, output: str) -> float:
    """Share of ``source`` the output reached, matching greedily in order.

    Exact under the copy constraint, so it says how much of the input the decode
    got through before it stopped.
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
    # A ceiling, not a budget: under the copy constraint the output is a
    # subsequence of the input, so the input's own length is the real bound and
    # a fixed number is a length limit on the stage wearing a safety cap's
    # clothes. ``decode_budget`` keeps it as a floor.
    #
    # **Straightening it out changed nothing that could be measured**, and that
    # is the finding rather than the fix. The ungrouped arm of the sweep was run
    # once on each side of this change, same data, greedy decode: deleted 2111
    # against 2111, pieces failing FLOOR 108 against 108, over the same 32971
    # characters, both recalls equal to the last digit. So the 256 was never
    # what stopped those decodes.
    #
    # What stops them is the model ending the sequence on its own, and
    # thinker.py:73 has that measured on this checkpoint family one message at
    # a time: 395 tokens empty 0/8, 435 2/8, 472 3/8, 774 6/8, 977 8/8. SFT ran
    # at --max_seq_len 512. The pieces this stage loses fit that shape and not
    # the cap's -- ungrouped, the ones that came back empty averaged 1460
    # characters, about 1226 tokens, three times past where thinker.py's own
    # guard sits.
    #
    # Kept anyway, because ``group_longest`` is a knob now: turn it up and the
    # fixed number would start to bite for real.
    max_new_tokens: int = 256
    # Longest group the transcript is cut into before the model sees it. A knob
    # rather than a constant because it has never been answered: the one sweep
    # we ran scored arms on how little they deleted, so it measured which
    # setting deletes less, not which is better.
    group_longest: int = TRAINED_LENGTH
    # The two ways a piece can come back unpolished, counted apart because they
    # are reached by different roads and only one of them was visible.
    #
    # ``floored`` is the decode that stopped partway and failed the FLOOR test.
    # ``emptied`` is the decode that produced nothing at all -- and that one was
    # invisible, because ``_polish_batch`` substitutes the input for an empty
    # answer, after which ``consumed(piece, piece)`` is 1.0 and the FLOOR test
    # passes. The arm most likely to answer empty was therefore the arm that
    # reported the fewest fallbacks. It is the third instance tonight of a
    # counter that could not see the thing it counted.
    #
    # Not hypothetical: thinker.py:73 has this checkpoint family measured, one
    # user message at a time -- 395 tokens empty 0/8, 435 2/8, 472 3/8, 774
    # 6/8, 977 8/8. SFT ran at --max_seq_len 512.
    floored: int = 0
    floored_chars: int = 0
    emptied: int = 0
    emptied_chars: int = 0
    # How many pieces one decode may carry. Wide enough that a step is full,
    # narrow enough that one caller's long dictation does not hold everyone
    # else's: a batch runs until its longest member stops, so the cost of a
    # wide batch is paid by its shortest piece. Measured at 64 on real-length
    # pieces the wait became minutes; eight keeps the throughput and not that.
    max_batch: int = 8
    # The second arm and the threshold it answers at. Unset, the stage is
    # generate-tidy-done. Set, the two are merged by veto: the head's confidence
    # protects the generator's content rather than removing more of it, and
    # neither may veto a vocabulary filler or an exact repetition.
    tagger: Path | None = None
    tagger_backbone: Path | None = None
    tagger_threshold: float = 0.5
    # Marks the decode may emit even though the transcript did not. Off by
    # default, which is the copy constraint exactly as it has always been.
    #
    # The constraint says the output must be a subsequence of the transcript,
    # and that is what makes invention structurally impossible. It also, as a
    # side effect nobody chose, makes punctuation uneditable -- a mark can be
    # dropped but never moved, so the recogniser's punctuation is final. On the
    # training pool that costs nothing: every target there is already a
    # subsequence, because §18 built them that way. On real spontaneous speech
    # 87.7% of targets are unreachable, and every one of them is unreachable
    # *only* on punctuation -- the content is a subsequence in 100% of them.
    #
    # So this opens the one door and no other: content still comes from the
    # transcript, and the set below is closed and small.
    punctuation_insertable: bool = False
    # How much to add to a mark's logit before the argmax. Teacher-forced on
    # 101 positions where a human put a mark, the model ranks that mark in its
    # top three every single time and first only 20.8% of the time -- median
    # rank two, median probability 0.142. It learnt where marks go and greedy
    # decoding throws that away, because a mark loses to the next transcript
    # character by a small margin and argmax only keeps the winner. This is the
    # same knob the streaming-translation work calls a decode bias; it changes
    # no weights and leaves the copy constraint alone, because content still
    # comes only from the transcript.
    punctuation_bias: float = 0.0
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
    _insertable: Any = None
    #: Marks this decode emitted that the transcript did not offer. A counter
    #: rather than a flag: the question is how often, and zero is a real answer.
    inserted: int = 0

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
        pieces = group_sentences(transcript, self.group_longest)
        wanted = [piece for piece in pieces if piece.strip() and worth_polishing(piece)]
        if not wanted:
            return transcript
        # On a thread, for the reason `_transcribe` is: a blocking GPU call on
        # the service's only event loop stalls every other connection for the
        # length of the forward pass. Gathered rather than awaited in turn
        # because the arms do not depend on each other, though the gain is small
        # -- one card, both arms on it, so "concurrent" is mostly a queue.
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
            reached = consumed(piece, answer)
            if reached < FLOOR:
                # Counted here rather than inferred from the output, because a
                # piece that fell back is character-for-character identical to
                # one the model chose not to touch. The punctuation reading this
                # stage lost a whole night to was exactly that shape: a check
                # that could not see the thing it was counting.
                self.floored += 1
                self.floored_chars += len(piece)
            written = answer if reached >= FLOOR else piece
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
            else:
                # `merge` ends with the same rule. Without this branch a
                # deployment with no tagger keeps the whole defect: the
                # generator arm deletes 这个 on its own, and 14 of the 20
                # reported dictations break with one arm too.
                written = keep_content(piece, written)
            out.append(written)
        # Tidied over the joined text, not per piece: a piece boundary is a
        # sentence boundary, so a particle stranded at the start of one is only
        # visible once its neighbour is in front of it.
        polished = tidy("".join(out))
        # Never hand back an empty text box. An utterance that is all filler
        # gets deleted down to punctuation by both arms, which FLOOR cannot
        # catch: `consumed` measures how far the decode reached, not how much
        # survived. Whole-result level on purpose -- dropping one all-filler
        # sentence out of a longer dictation is this stage working.
        if transcript.strip() and not has_content(polished):
            # Two different things reach here and they want opposite answers.
            # If what was removed is nothing but vocabulary filler, empty is the
            # right answer and handing the transcript back is the defect: half a
            # second of fan noise recognises as 嗯。and the product wrote 嗯。
            # into the user's text box, while the same noise over two seconds
            # correctly wrote nothing. It also contradicts the rest of the
            # stage, which deletes 嗯 everywhere it is not the whole utterance.
            if nothing_but_filler(transcript):
                return ""
            return transcript
        return polished

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

        # Room for the piece itself plus what a decode may add on top of it:
        # an inserted mark has no character in the source to match, and the
        # stop token needs a step of its own.
        budget = decode_budget(max(len(ids) for ids in sources), self.max_new_tokens)

        produced: list[list[int]] = [[] for _ in pieces]
        # Advanced a token at a time instead of recomputed from the whole
        # output each step. Identical to subsequence_pointer while nothing is
        # inserted -- both walk the transcript greedily -- and the only version
        # that stays correct once something is: an inserted mark has nothing to
        # match, so the greedy scan would run to the end of the transcript and
        # take the rest of the sentence with it.
        pointer = [0] * len(pieces)
        insertable = self._insertable_ids()
        done = [False] * len(pieces)
        cache, step_in = None, padded
        with torch.no_grad():
            for _ in range(budget):
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
                    window = set(
                        reachable(
                            piece_source,
                            pointer[row],
                            self.lookahead,
                            self._filler_spans(),
                            self.protect_head,
                        )
                    )
                    ahead = window | insertable
                    ahead.add(stop)
                    keep = torch.tensor(sorted(ahead), device=logits.device, dtype=torch.long)
                    masked = torch.full_like(logits[row], float("-inf"))
                    masked[keep] = logits[row][keep]
                    if insertable and self.punctuation_bias:
                        # Only the marks the transcript did not already offer:
                        # a mark that is the next character is a copy, and
                        # tilting the scale for it would be tilting it for
                        # content that happens to be punctuation.
                        added = insertable - window
                        if added:
                            lift = torch.tensor(
                                sorted(added), device=logits.device, dtype=torch.long
                            )
                            masked[lift] += self.punctuation_bias
                    chosen = int(masked.argmax())
                    if chosen == stop:
                        done[row] = True
                    else:
                        produced[row].append(chosen)
                        # A token the transcript offered consumes it. A mark
                        # that only the insertable set offered consumes
                        # nothing, so the window does not move past content the
                        # model has not written yet.
                        #
                        # Counted here rather than inferred from the output,
                        # because an inserted 。 that also appears later in the
                        # transcript leaves the output a subsequence of it and
                        # the after-the-fact check reads zero insertions where
                        # there were many. Chinese transcripts are full of
                        # commas and full stops, so that check was never going
                        # to work.
                        if chosen not in window:
                            self.inserted += 1
                        if chosen in window:
                            at = pointer[row]
                            while at < len(piece_source) and piece_source[at] != chosen:
                                at += 1
                            pointer[row] = min(at + 1, len(piece_source))
                    nxt.append(chosen)
                if all(done):
                    break
                step_in = torch.tensor([[token] for token in nxt], device=self.device)
                mask = torch.cat([mask, torch.ones((len(pieces), 1), device=self.device)], dim=1)

        answers = []
        for piece, tokens in zip(pieces, produced, strict=True):
            text = self._tokeniser.decode(tokens, skip_special_tokens=True).strip()
            if not text:
                self.emptied += 1
                self.emptied_chars += len(piece)
            answers.append(text or piece)
        return answers

    def _insertable_ids(self) -> frozenset[int]:
        """Token ids for the marks, when the graded constraint is on.

        Every id whose decoded text is nothing but marks, not just the ids of
        the single characters: a tokeniser that has a token for 。」 would
        otherwise let a whole shape through the door unnoticed.
        """
        if not self.punctuation_insertable:
            return frozenset()
        if self._insertable is None:
            marks = PUNCTUATION | set(",;:.!?")
            found = set()
            for token, index in self._tokeniser.get_vocab().items():
                text = self._tokeniser.convert_tokens_to_string([token])
                if text and all(ch in marks for ch in text):
                    found.add(index)
            self._insertable = frozenset(found)
        return self._insertable

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
        budget = decode_budget(len(source), self.max_new_tokens)
        with torch.no_grad():
            for _ in range(budget):
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
