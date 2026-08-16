def test_a_marked_character_is_deleted_and_the_rest_survives() -> None:
    """The conversion infers nothing: the target is the source minus what a
    person marked, in order."""
    import json
    import tempfile
    from pathlib import Path

    from scripts.cs2w_to_pairs import convert

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cs2w.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": 1,
                    "content": "千万不要不要有贪婪的意识",
                    "disfluency_label": "002200000000",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        rows, dropped = convert(path, "train")

    assert dropped == 0
    # Positions 2 and 3 are marked, which is the FIRST 不要 -- the annotator
    # marks the copy that was restarted, not the one that survived.
    assert rows[0]["source"] == "千万不要不要有贪婪的意识"
    assert rows[0]["target"] == "千万不要有贪婪的意识"
    assert rows[0]["split"] == "train"


def test_a_row_whose_labels_do_not_reach_the_end_is_dropped() -> None:
    """A short label string would misalign every position after the gap, which
    is a mislabelling no loss curve can show."""
    import json
    import tempfile
    from pathlib import Path

    from scripts.cs2w_to_pairs import convert

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "cs2w.jsonl"
        path.write_text(
            json.dumps(
                {"id": 1, "content": "一二三四", "disfluency_label": "00"}, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )

        rows, dropped = convert(path, "train")

    assert rows == []
    assert dropped == 1
