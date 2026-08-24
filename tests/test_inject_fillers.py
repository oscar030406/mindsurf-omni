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


def test_the_repetition_share_is_a_knob() -> None:
    """Repetition is the half the filler-clearance line now fails on: the best
    arm clears 0.995 of vocabulary filler and 0.749 of repetition."""
    text = "阳台的花需要阳光和水分，所以要保持充足的水分。"

    def count(share: float) -> int:
        total = 0
        for index in range(60):
            _spoken, injections = inject(text, random.Random(f"share:{index}"), 0.9, 3, share)
            total += sum(1 for item in injections if item["kind"] == "repetition")
        return total

    assert count(0.0) == 0
    assert count(1 / 3) < count(1.0)


def test_repeats_are_shaped_the_way_people_make_them() -> None:
    """The injector used to make one shape, and the model learned one shape.

    Measured against CS2W's 3807 human repetitions, ours were 88% two
    characters and always adjacent where people are 58% one character and half
    of them separated by something. And the model is worst at exactly what it
    saw least: recall by shape is 0.755 adjacent against 0.426 gapped, with
    single-character repeats the low point of both (0.662 and 0.343).

    So the distribution is the thing under test, not any one draw.
    """
    import collections
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from inject_fillers import GAP_FILLERS, repeat_of

    rng = random.Random(20260824)
    # Characters chosen not to collide with the gap vocabulary, or stripping
    # the gap back off would eat the head and read the shape wrong. That
    # mistake was made once while writing this.
    clause = "明天下午开会讨论预算"
    sizes: collections.Counter[int] = collections.Counter()
    gapped = 0
    made = 0
    for _ in range(4000):
        out = repeat_of(clause, rng)
        if not out:
            continue
        made += 1
        core = out
        for gap in sorted(GAP_FILLERS, key=len, reverse=True):
            if core.endswith(gap):
                core = core[: -len(gap)]
                gapped += 1
                break
        sizes[min(len(core), 3)] += 1

    assert made > 3_000
    assert abs(sizes[1] / made - 0.58) < 0.04, "单字该占 58%"
    assert abs(sizes[2] / made - 0.31) < 0.04, "双字该占 31%"
    assert abs(sizes[3] / made - 0.11) < 0.04, "三字以上该占 11%"
    assert abs(gapped / made - 0.49) < 0.05, "中间隔开的该占 49%"


def test_a_repeat_never_comes_back_as_punctuation_alone() -> None:
    """A clause that is nothing but marks has no head to repeat."""
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from inject_fillers import repeat_of

    rng = random.Random(1)
    assert repeat_of("，。！", rng) == ""
    assert repeat_of("", rng) == ""
