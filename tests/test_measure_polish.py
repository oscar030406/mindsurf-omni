"""The two numbers that must be read together: cleaned, and still the same words."""

from __future__ import annotations

from scripts.measure_polish import CONTENT_KEPT, content_kept, filler_removed, invented


def test_a_model_that_deletes_content_scores_low_even_when_it_cleaned_well() -> None:
    """This is the failure the second criterion exists to catch."""
    target = "北京有很多著名的景点，比如故宫、天坛、颐和园"

    faithful = content_kept(target, "北京有很多著名的景点，比如故宫、天坛、颐和园")
    truncated = content_kept(target, "北京有很多著名的景点")

    assert faithful == 1.0
    assert truncated < CONTENT_KEPT


def test_invention_is_measured_against_the_original_not_the_transcript() -> None:
    """A polisher that adds a helpful sentence has changed what the user said."""
    target = "今天天气怎么样"

    assert invented(target, "今天天气怎么样") == 0.0
    assert invented(target, "今天天气怎么样？我可以帮你查一下天气预报") > 0.2


def test_filler_the_recogniser_already_dropped_is_not_credited_to_the_model() -> None:
    """Otherwise the score rises when the recogniser tidies, which is not our work."""
    row = {
        "source": "今天天气怎么样",  # the 那个 never reached the transcript
        "target": "今天天气怎么样",
        "injections": [{"kind": "filler", "token": "那个", "clause": 0}],
    }

    arrived, removed = filler_removed(row, "今天天气怎么样")

    assert (arrived, removed) == (0, 0)


def test_filler_that_arrived_and_was_removed_counts() -> None:
    row = {
        "source": "那个今天天气怎么样",
        "target": "今天天气怎么样",
        "injections": [{"kind": "filler", "token": "那个", "clause": 0}],
    }

    assert filler_removed(row, "今天天气怎么样") == (1, 1)
    assert filler_removed(row, "那个今天天气怎么样") == (1, 0)


def test_the_pointer_walks_the_input_as_the_output_deletes_from_it() -> None:
    """Deletion-only output is a subsequence, so position is recoverable from it."""
    from scripts.measure_polish import subsequence_pointer

    source = [10, 11, 12, 13, 14]

    assert subsequence_pointer(source, []) == 0
    assert subsequence_pointer(source, [10]) == 1
    # 11 and 12 deleted: the pointer jumps past them.
    assert subsequence_pointer(source, [10, 13]) == 4
    assert subsequence_pointer(source, [10, 11, 12, 13, 14]) == 5


def test_a_known_filler_can_be_stepped_over_whole_but_a_clause_cannot() -> None:
    """The narrow window protects content; the filler door removes filler.

    A single width cannot do both: wide enough for 你知道吧 is wide enough to
    drop a clause, which is what the unbounded arm did.
    """
    from scripts.measure_polish import reachable

    # 0 1 2 are a filler; 3.. is what was said.
    source = [40, 41, 42, 50, 51, 52, 53, 54, 55]
    narrow = reachable(source, 0, lookahead=2, fillers=())
    with_door = reachable(source, 0, lookahead=2, fillers=((40, 41, 42),))

    assert narrow == [40, 41]  # cannot reach past the filler
    assert 50 in with_door and 51 in with_door  # the door opens onto what follows
    assert 54 not in with_door  # but not onto the rest of the sentence


def test_no_lookahead_means_everything_ahead() -> None:
    from scripts.measure_polish import reachable

    assert reachable([1, 2, 3], 1, lookahead=0) == [2, 3]
