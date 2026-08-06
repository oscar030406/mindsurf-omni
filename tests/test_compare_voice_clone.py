"""The clustering, which is the part that decides the verdict."""

from __future__ import annotations

from scripts.compare_voice_clone import grouped_deltas, voice_of


def report(rows: dict[str, float]) -> dict[str, dict]:
    return {clip: {"similarity": value, "hit": True} for clip, value in rows.items()}


def test_the_clip_id_carries_its_own_voice() -> None:
    assert voice_of("uncle_fu/zh007") == "uncle_fu"
    assert voice_of("moon/zh000") == "moon"


def test_deltas_group_by_voice_not_by_clip() -> None:
    """Twenty clips of one voice are one cluster, not twenty independent rows."""
    candidate = report({"a/zh000": 0.7, "a/zh001": 0.6, "b/zh000": 0.5})
    reference = report({"a/zh000": 0.6, "a/zh001": 0.5, "b/zh000": 0.5})

    groups = grouped_deltas(candidate, reference, "similarity")

    assert sorted(groups) == ["a", "b"]
    assert len(groups["a"]) == 2 and len(groups["b"]) == 1


def test_only_shared_clips_are_paired() -> None:
    candidate = report({"a/zh000": 0.7, "a/zh999": 0.1})
    reference = report({"a/zh000": 0.6, "a/zh888": 0.9})

    groups = grouped_deltas(candidate, reference, "similarity")

    assert groups == {"a": [0.7 - 0.6]}


def test_clustering_changes_the_floor_enough_to_matter() -> None:
    """One voice moving a lot must not read as twelve voices agreeing.

    The per-row floor sees 20 rows; the clustered floor sees one cluster that
    moved and eleven that did not, which is what the data actually shows.
    """
    from mindsurf_omni.evaluation.metrics import compare_paired, compare_paired_clustered

    groups = {f"v{i}": [0.0] * 20 for i in range(11)}
    groups["moved"] = [0.30] * 20
    flat = [value for group in groups.values() for value in group]

    per_row = compare_paired(
        "clone_similarity", flat, lower_is_better=False, effect_of_interest=0.05
    )
    clustered = compare_paired_clustered(
        "clone_similarity", groups, lower_is_better=False, effect_of_interest=0.05
    )

    assert per_row != clustered
