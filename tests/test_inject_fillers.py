"""Injected filler, which is the only synthetic half of the polish training pairs."""

from __future__ import annotations

import random

from scripts.inject_fillers import BRIDGING_FILLERS, LEADING_FILLERS, inject, split_clauses


def test_clauses_keep_their_punctuation() -> None:
    """The comma is where the speaker hesitates, so it has to survive the split."""
    assert split_clauses("你好，今天天气怎么样？") == ["你好，", "今天天气怎么样？"]
    assert split_clauses("没有标点") == ["没有标点"]


def test_the_same_seed_injects_the_same_words() -> None:
    """Audio is synthesised once; a re-run that moves the filler orphans it."""
    text = "您想看什么类型的电影？比如浪漫爱情、惊悚恐怖、科幻冒险等等。"

    first = inject(text, random.Random("20260815:zh001"), 0.35, 3)
    second = inject(text, random.Random("20260815:zh001"), 0.35, 3)

    assert first == second


def test_every_recorded_injection_is_in_the_sentence() -> None:
    """The retention count is read off this record; a phantom entry deflates it."""
    for index in range(40):
        spoken, injections = inject(
            "你好，今天天气怎么样？我想出门。", random.Random(f"seed:{index}"), 0.9, 3
        )
        for item in injections:
            assert item["token"] in spoken
        assert len(injections) <= 3


def test_nothing_is_injected_at_rate_zero() -> None:
    """The clean arm has to be reachable through the same code path."""
    spoken, injections = inject("你好，今天天气怎么样？", random.Random(1), 0.0, 3)

    assert injections == []
    assert spoken == "你好，今天天气怎么样？"


def test_the_first_clause_gets_no_bridging_filler() -> None:
    """怎么说呢 between clauses is speech; at the very front it is a different sentence."""
    for index in range(60):
        spoken, injections = inject(
            "你好，今天天气怎么样？", random.Random(f"lead:{index}"), 1.0, 3
        )
        leading = [item for item in injections if item["clause"] == 0 and item["kind"] == "filler"]
        assert all(item["token"] in LEADING_FILLERS for item in leading)
        assert spoken.startswith(tuple(LEADING_FILLERS) + ("你好",))
    assert set(BRIDGING_FILLERS).isdisjoint(LEADING_FILLERS)


def test_some_filler_lands_inside_the_clause_not_only_in_front_of_it() -> None:
    """The first data round put every filler at a boundary and the model learned
    the boundary: front-of-sentence filler survived 5.4% of the time, filler
    further in 21.4%."""
    inside = 0
    for index in range(80):
        spoken, injections = inject(
            "阳台的花需要阳光和水分，所以要保持充足的水分。",
            random.Random(f"inside:{index}"),
            0.9,
            3,
        )
        for item in injections:
            if item.get("inside"):
                inside += 1
                # The clause still reads in order around the inserted word.
                assert item["token"] in spoken
    assert inside > 0


def test_an_inserted_word_never_swallows_the_clause() -> None:
    """Every character of the original has to survive the insertion."""
    text = "阳台的花需要阳光和水分，所以要保持充足的水分。"
    for index in range(40):
        spoken, _ = inject(text, random.Random(f"keep:{index}"), 0.9, 3)
        remaining = iter(spoken)
        assert all(character in remaining for character in text)


def test_the_pause_lands_between_words_not_inside_one() -> None:
    """微波然后炉 is a broken word, not a hesitation."""
    from scripts.inject_fillers import word_boundaries

    clause = "微波炉里的异味怎么去掉"

    cuts = word_boundaries(clause)

    assert cuts, "a clause this long has interior boundaries"
    assert all(2 <= cut <= len(clause) - 2 for cut in cuts)
    # 微波炉 is one word: no cut inside it.
    assert 1 not in cuts and 2 not in cuts
