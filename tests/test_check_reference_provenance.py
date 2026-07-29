"""Provenance is declared, and these cover why it cannot be inferred instead.

Membership in the training data was checked when the chat holdout was adopted,
and it passed. The set was still unusable, because one of the arms had written
it -- and a model's own samples leave no trace in any dataset.

The natural next idea is to detect authorship by resemblance. It was built and
measured against the known case and it does not work: at temperature 0.7 a
model does not reproduce its own earlier reply, so the true author scored mean
similarity 0.291. What the search did return was short degenerate replies that
unrelated models produce identically. Hence the split below -- a verbatim
screen that only claims to catch a copied file, and a declared author that
carries the real weight.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_reference_provenance import (
    TOO_SHORT_TO_MEAN_ANYTHING,
    VERBATIM,
    provenance_path,
    similarity,
    verbatim_reuse,
)


def test_a_copied_file_is_caught() -> None:
    reference = [{"id": "a", "text": "北京有很多好玩的地方，故宫和长城都值得专门安排一天去看。"}]
    finding = verbatim_reuse(reference, {"copier": list(reference)})

    assert finding["copier"]["verdict"] == "COPIED"


def test_a_short_collision_is_not_evidence() -> None:
    """Several unrelated models answer this prompt with these five characters.

    Counting that as authorship is how a screen starts reporting noise -- and
    it was most of what the resemblance search returned on the real data.
    """
    short = "米饭更顶饱"
    assert len(short) < TOO_SHORT_TO_MEAN_ANYTHING

    finding = verbatim_reuse(
        [{"id": "a", "text": short}], {"unrelated": [{"id": "a", "text": short}]}
    )

    assert finding["unrelated"]["compared"] == 0
    assert finding["unrelated"]["verdict"] == "nothing long enough to compare"


def test_resampled_text_is_not_detectable_which_is_why_the_bar_is_verbatim() -> None:
    """The measured failure, kept as a test so nobody lowers the threshold.

    These two are the same model answering the same prompt twice at temperature
    0.7. They are unmistakably one author to a reader and score nowhere near
    the bar, which is the whole reason authorship has to be recorded instead.
    """
    first = "猫喜欢纸箱是因为躲在里面有安全感，四面有遮挡，它只需要留意一个方向。"
    again = "纸箱能给猫安全感，狭小的空间让它不必分心防备四周，还比较保暖。"

    assert similarity(first, again) < VERBATIM


def test_a_set_without_a_recorded_author_cannot_be_cleared(tmp_path: Path) -> None:
    reference = tmp_path / "refs.jsonl"
    reference.write_text(json.dumps({"id": "a", "text": "答案"}) + "\n", encoding="utf-8")

    assert not provenance_path(reference).is_file()
    assert provenance_path(reference).name == "refs.jsonl.provenance.json"
