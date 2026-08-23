"""Put the speaker's own proper nouns back where the recogniser guessed at them.

SenseVoice has no hotword input -- its `generate` takes audio, a language and
nothing else -- so this is a pass over the finished transcript rather than a
bias on the decode. It matches by toneless pinyin: of the fifteen Chinese
proper nouns the recogniser got wrong on the probe corpus, thirteen came back
as an exact or near homophone, so a sound match reaches them and edit distance
is not needed.

The whole risk runs the other way. 部署 and 不熟 are one sound, 李工 and 理工
are one sound: a table holding either turns every ordinary sentence carrying
the other into nonsense. Two doors stand in the way.

Admission, at assembly. A word is refused when jieba's dictionary holds a
different word with the same sound at or above FREQUENCY_FLOOR. That rejects
部署 (部属, 1126), 断连 (锻炼, 1689) and 李工 (理工, 490); it admits 吕鑫 and
通义千问. Over the 65-sentence trap set that one door takes the damage from 22
sentences to 2, and both survivors are sentences the recogniser had already
got wrong before this stage ran.

The sentence, at request time. The dictionary only knows collisions between
whole words, so admission cannot see one that spans two. 吕鑫 passes it, and
then 心率心率的计算公式 in the held-out set offers 率心 across the seam --
same sound, no dictionary word involved. So a span is refused when it is a
dictionary word in its own right, or when a dictionary word crosses either of
its edges. That door is what takes the held-out set from one damaged sentence
to zero.

Tone was measured, not assumed. Matching on tone as well loses 通一千问 (一 is
yī against 义 yì) while admitting 断连, which then damages the ordinary 断联
sentences: same 5 recoveries, two more damaged. Toneless wins on both counts.

What comes out is the transcript with whole table entries spliced in at equal
character count. No character that was not already in the transcript or in the
table can appear -- which is what replaces the polish stage's subsequence
argument for this path.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import jieba
from pypinyin import lazy_pinyin

from mindsurf_omni.service.config import ConfigurationError

# How loud a rival has to be before it costs a word its place, and how loud a
# word has to be before it protects a span below. jieba's own counts, so the
# unit is that dictionary's corpus rather than this product's traffic.
#
# Swept over the probe corpus, the 65-sentence trap set and the 986 held-out
# transcripts, with the five words the probe asked for:
#
#      1..5   admits 1   recovered 3/14   damaged 0/65   held-out 0/986
#     10..100 admits 2   recovered 5/14   damaged 0/65   held-out 0/986
#      250    admits 2   recovered 5/14   damaged 0/65   held-out 1/986
#      500    admits 3   recovered 7/14   damaged 4/65   held-out 1/986
#     2000    admits 5   recovered 10/14  damaged 12/65  held-out 12/986
#
# A plateau from 10 to 100 with nothing to choose inside it. Below it 吕鑫 goes
# too, on 履新 at 8. Above it the doors fail one at a time and in that order:
# at 250 the sentence door stops seeing 心率 and lets 率心 through, at 500 李工
# is admitted and four ordinary 理工 sentences break.
FREQUENCY_FLOOR = 10

# How far either side of an edge to look for a word crossing it. jieba holds
# entries up to sixteen characters, but they are idioms and place names; the
# words this door exists to protect (熟悉, 联系, 锻炼身体) are two to four.
_LONGEST_WORD = 8

_CHINESE = re.compile(r"[一-鿿]+")


def build_table(words: Iterable[str]) -> dict[tuple[str, ...], str]:
    """Sounds mapped to the words they should be written as.

    Raises rather than dropping a word it will not take. A table where some
    entries silently do nothing is the worse failure: the operator configured
    a term, the service reported ready, and the term never appeared -- with no
    surface anywhere that says which half of the table is live.
    """
    jieba.initialize()
    table: dict[tuple[str, ...], str] = {}
    for word in words:
        if not _CHINESE.fullmatch(word) or len(word) < 2:
            raise ConfigurationError(
                f"MINDSURF_HOTWORDS carries {word!r}; this stage matches Chinese by pinyin, "
                "so an entry needs two or more Chinese characters. A Latin term would sit "
                "in the table and never match anything"
            )
        sound = _sound(word)
        if sound in table:
            raise ConfigurationError(
                f"MINDSURF_HOTWORDS carries both {table[sound]!r} and {word!r}, which sound "
                "exactly alike; there is no way to tell which one a transcript meant"
            )
        table[sound] = word

    # One pass over the dictionary rather than one per word: it holds half a
    # million entries and a two-hundred-word table would otherwise convert it
    # two hundred times.
    lengths = {len(word) for word in table.values()}
    rivals: dict[str, tuple[str, int]] = {}
    for other, frequency in jieba.dt.FREQ.items():
        if frequency < FREQUENCY_FLOOR or len(other) not in lengths:
            continue
        word = table.get(_sound(other))
        if word is not None and word != other and frequency > rivals.get(word, ("", 0))[1]:
            rivals[word] = (other, frequency)
    if rivals:
        word, (other, frequency) = max(rivals.items(), key=lambda entry: entry[1][1])
        raise ConfigurationError(
            f"MINDSURF_HOTWORDS carries {word!r}, which sounds exactly like {other!r} "
            f"(jieba frequency {frequency}); every transcript that says {other!r} would come "
            f"back saying {word!r}, so this entry costs more than it repairs. Drop it, or "
            "write the term some other way"
        )
    return table


def corrections(text: str, table: dict[tuple[str, ...], str]) -> list[tuple[int, int, str]]:
    """Where this transcript should be repaired, as (start, end, replacement).

    Non-overlapping and in order. Longest entries first, so that a table
    holding both a term and something that sounds like part of it repairs the
    term rather than half of it.
    """
    sizes = sorted({len(word) for word in table.values()}, reverse=True)
    boundaries = _boundaries(text)
    found: list[tuple[int, int, str]] = []
    taken: set[int] = set()
    for run in _CHINESE.finditer(text):
        sounds = lazy_pinyin(run.group())
        # pypinyin folds some characters into one entry, which would put every
        # index after it one place out. Rare, and skipping the run costs a
        # repair while guessing costs a wrong edit.
        if len(sounds) != len(run.group()):
            continue
        for size in sizes:
            for index in range(len(sounds) - size + 1):
                word = table.get(tuple(sounds[index : index + size]))
                start = run.start() + index
                if word is None or text[start : start + size] == word:
                    continue
                if not taken.isdisjoint(range(start, start + size)):
                    continue
                if not _replaceable(text, start, start + size, boundaries):
                    continue
                found.append((start, start + size, word))
                taken.update(range(start, start + size))
    return sorted(found)


def correct(text: str, table: dict[tuple[str, ...], str]) -> str:
    """The transcript with the table's words spliced in where they belong."""
    pieces, cursor = [], 0
    for start, end, word in corrections(text, table):
        pieces += [text[cursor:start], word]
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _sound(word: str) -> tuple[str, ...]:
    return tuple(lazy_pinyin(word))


def _boundaries(text: str) -> set[int]:
    """Character offsets where jieba puts a word boundary, ends included."""
    offsets = {0}
    cursor = 0
    for token in jieba.cut(text):
        cursor += len(token)
        offsets.add(cursor)
    return offsets


def _reads_as_words(text: str, start: int, end: int, boundaries: set[int]) -> bool:
    """Whether the segmenter already reads this span as ordinary words.

    The door the first version of this stage did not have, and the one that
    matters: it checked the span and the two edges, so 统一前文 -- 统一 and 前文,
    two everyday words that together sound like 通义千问 -- passed all three
    and 先统一前文的术语 came back as 先通义千问的术语. The word doing the damage
    was wholly inside the span, where nothing was looking.

    A span standing on two boundaries with a boundary inside it is two words
    the speaker said that happen to rhyme with the table's entry. A span
    standing on two boundaries with nothing inside is one word, and a rare one
    is still one: 履新 is in the dictionary at 8, below the admission floor,
    and 今天履新 must not become 今天吕鑫. What is left -- a span the segmenter
    cuts across, or one it made up (吕新, which jieba's own model emits as a
    name with no dictionary entry behind it) -- is where a misrecognition
    lives.
    """
    if start not in boundaries or end not in boundaries:
        return False
    if any(start < inside < end for inside in boundaries):
        return True
    return jieba.get_FREQ(text[start:end]) is not None


def _replaceable(text: str, start: int, end: int, boundaries: set[int]) -> bool:
    """Whether this span is the recogniser's mistake rather than ordinary text.

    A span that is itself a common word is what the speaker said -- 锻炼 in
    他每天都锻炼身体 sounds like 断连 and is not it. A span with a common word
    crossing one of its edges is a piece of one -- 不熟 inside 不熟悉, 断联
    inside 断联系, 率心 inside 心率心率.
    """
    frequency = jieba.get_FREQ(text[start:end])
    if frequency is not None and frequency >= FREQUENCY_FLOOR:
        return False
    if _reads_as_words(text, start, end, boundaries):
        return False
    return not _crossed_by_a_word(text, start) and not _crossed_by_a_word(text, end)


def _crossed_by_a_word(text: str, index: int) -> bool:
    for start in range(max(0, index - _LONGEST_WORD), index):
        for end in range(index + 1, min(len(text), index + _LONGEST_WORD) + 1):
            frequency = jieba.get_FREQ(text[start:end])
            # jieba's table carries every prefix of every word, so a miss means
            # nothing longer from this start can be a word either.
            if frequency is None:
                break
            if frequency >= FREQUENCY_FLOOR:
                return True
    return False
