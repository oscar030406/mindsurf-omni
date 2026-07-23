"""The lazy table must return exactly what the full table would.

A reader that is fast and wrong is worse than one that is slow and right, so
every test here compares against ``pq.read_table`` on the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mindsurf_omni.data.lazy_parquet import LazyParquetTable


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Several row groups, so boundary arithmetic is actually exercised."""
    rows = 250
    table = pa.table(
        {
            "conversations": pa.array(
                [json.dumps([{"role": "user", "content": f"第{i}行"}]) for i in range(rows)],
                type=pa.large_string(),
            ),
            "answer_audios": pa.array([[i, i + 1] for i in range(rows)]),
        }
    )
    path = tmp_path / "data.parquet"
    pq.write_table(table, path, row_group_size=37)  # deliberately not a divisor
    return path


def test_every_row_matches_the_full_table(dataset: Path) -> None:
    full = pq.read_table(dataset)
    lazy = LazyParquetTable(str(dataset))

    assert len(lazy) == full.num_rows
    assert set(lazy.column_names) == set(full.column_names)
    for index in range(full.num_rows):
        assert lazy["conversations"][index].as_py() == full["conversations"][index].as_py()
        assert lazy["answer_audios"][index].as_py() == full["answer_audios"][index].as_py()


def test_random_access_across_row_group_boundaries(dataset: Path) -> None:
    """The training loader shuffles, so reads jump between row groups."""
    full = pq.read_table(dataset)
    lazy = LazyParquetTable(str(dataset), cached_row_groups=2)

    # Ordering chosen to thrash a two-entry cache.
    for index in [0, 249, 37, 200, 36, 38, 100, 1, 248, 74]:
        assert lazy["conversations"][index].as_py() == full["conversations"][index].as_py()


def test_multiple_files_are_concatenated_in_order(dataset: Path, tmp_path: Path) -> None:
    """The dataset accepts a comma-separated list, and order decides indices."""
    second = tmp_path / "second.parquet"
    pq.write_table(
        pa.table(
            {
                "conversations": pa.array(["第二个文件"], type=pa.large_string()),
                "answer_audios": pa.array([[999]]),
            }
        ),
        second,
    )
    lazy = LazyParquetTable(f"{dataset},{second}")

    assert len(lazy) == 251
    assert lazy["conversations"][250].as_py() == "第二个文件"
    # Rows of the first file keep their original indices.
    assert json.loads(lazy["conversations"][0].as_py())[0]["content"] == "第0行"


def test_negative_and_out_of_range_indices(dataset: Path) -> None:
    full = pq.read_table(dataset)
    lazy = LazyParquetTable(str(dataset))

    assert lazy["conversations"][-1].as_py() == full["conversations"][249].as_py()
    with pytest.raises(IndexError):
        lazy["conversations"][250]
    with pytest.raises(IndexError):
        lazy["conversations"][-251]


def test_unknown_column_is_an_error_not_silence(dataset: Path) -> None:
    """The dataset probes for optional columns; a wrong answer would be silent."""
    lazy = LazyParquetTable(str(dataset))

    assert "image_bytes" not in lazy.column_names
    with pytest.raises(KeyError):
        lazy["image_bytes"][0]


def test_only_the_cached_row_groups_stay_resident(dataset: Path) -> None:
    """The whole point is bounded memory, so the cache must actually bound."""
    lazy = LazyParquetTable(str(dataset), cached_row_groups=2)

    for index in range(0, 250, 37):
        lazy["conversations"][index].as_py()

    assert len(lazy._local.cache) <= 2
