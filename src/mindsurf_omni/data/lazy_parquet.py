"""Read a parquet dataset by row group instead of materialising all of it.

MiniMind-O's dataset builds the whole table up front::

    pa.Table.from_batches(pq.ParquetFile(path).iter_batches())

That is fine for the mini splits and fatal for the full ones. sft_t2a.parquet
is 5.35 GB on disk and its parquet metadata reports 5.78 GB "uncompressed",
but that figure counts the *encoded* bytes inside row groups; materialised into
Arrow, with JSON conversation strings and audio code arrays expanded, it does
not fit in this machine's 23 GB and the loader is OOM-killed before the first
step.

The rows are read in random order, so streaming is not an option -- but they
are read one at a time, and a row group here averages 23 MB. Keeping a handful
of row groups resident covers the access pattern at a cost that rounds to
nothing.

This is a drop-in for the ``.table`` attribute, so the dataset's own indexing
code is untouched: it still writes ``self.table['conversations'][i].as_py()``.
"""

from __future__ import annotations

import bisect
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class _Column:
    """One column of a lazily-read table, indexable by global row number."""

    def __init__(self, table: LazyParquetTable, name: str) -> None:
        self._table = table
        self._name = name

    def __getitem__(self, index: int) -> Any:
        return self._table.value(self._name, index)


class LazyParquetTable:
    """Enough of ``pa.Table`` for a dataset that reads one row at a time.

    Not a general substitute: it supports column lookup, length, and the
    ``column_names`` attribute, which is what the access pattern needs.
    """

    # A row group larger than this defeats the point: reading one row means
    # materialising the whole group. sft_a2a.parquet ships as a single 6.1 GB
    # group holding all 414,024 rows, which OOM-killed a DataLoader worker
    # before a single step ran.
    LARGEST_SENSIBLE_ROW_GROUP_BYTES = 512 * 1024 * 1024

    def __init__(
        self,
        paths: str | list[str],
        cached_row_groups: int = 4,
        check_row_group_size: bool = True,
    ) -> None:
        if isinstance(paths, str):
            paths = [part.strip() for part in paths.split(",") if part.strip()]
        self._paths = [str(Path(path)) for path in paths]
        self._cache_size = max(1, cached_row_groups)

        # Row-group boundaries come from the footer, so this costs a seek per
        # file rather than a read of the data.
        self._offsets: list[int] = []
        self._groups: list[tuple[int, int]] = []  # (file index, row group index)
        total = 0
        self.column_names: list[str] = []
        for file_index, path in enumerate(self._paths):
            handle = pq.ParquetFile(path)
            if not self.column_names:
                self.column_names = list(handle.schema_arrow.names)
            for group_index in range(handle.metadata.num_row_groups):
                group = handle.metadata.row_group(group_index)
                if check_row_group_size and group.total_byte_size > (
                    self.LARGEST_SENSIBLE_ROW_GROUP_BYTES
                ):
                    handle.close()
                    raise ValueError(
                        f"{path} row group {group_index} holds {group.num_rows:,} rows "
                        f"in {group.total_byte_size / 1e9:.1f} GB. Reading one row "
                        f"materialises the whole group, so lazy reading buys nothing "
                        f"and a worker will be OOM-killed. Repartition it first: "
                        f"python scripts/repartition_parquet.py --input {path}"
                    )
                self._offsets.append(total)
                self._groups.append((file_index, group_index))
                total += group.num_rows
            handle.close()
        self._length = total

        # Opened per process: a parquet handle carries file state that does not
        # survive a fork, so every DataLoader worker needs its own.
        self._local = threading.local()

    def __len__(self) -> int:
        return self._length

    @property
    def num_rows(self) -> int:
        return self._length

    def __getitem__(self, name: str) -> _Column:
        return _Column(self, name)

    def _handles(self) -> list[pq.ParquetFile]:
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = [pq.ParquetFile(path, memory_map=True) for path in self._paths]
            self._local.handles = handles
            self._local.cache = OrderedDict()
        return handles  # type: ignore[no-any-return]

    def _row_group(self, slot: int) -> pa.Table:
        self._handles()
        cache: OrderedDict[int, pa.Table] = self._local.cache
        cached = cache.get(slot)
        if cached is not None:
            cache.move_to_end(slot)
            return cached

        file_index, group_index = self._groups[slot]
        table = self._local.handles[file_index].read_row_group(group_index)
        cache[slot] = table
        while len(cache) > self._cache_size:
            cache.popitem(last=False)
        return table

    @property
    def row_group_offsets(self) -> list[int]:
        """Row-group boundaries, for a sampler that wants to respect them."""
        return list(self._offsets)

    def value(self, column: str, index: int) -> Any:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"row {index} out of range for {self._length} rows")
        slot = bisect.bisect_right(self._offsets, index) - 1
        table = self._row_group(slot)
        if column not in table.column_names:
            raise KeyError(column)
        return table[column][index - self._offsets[slot]]


def _single_row_copy(path: str) -> str:
    """A one-row parquet with the same schema, cheap to materialise."""
    import tempfile

    handle = pq.ParquetFile(path)
    first = next(handle.iter_batches(batch_size=1))
    # delete=False and closed immediately: pyarrow writes by path, and on
    # Windows a file cannot be reopened while the handle is still held.
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as target:
        name = target.name
    pq.write_table(pa.Table.from_batches([first]), name)
    return name


def patch_omni_dataset(cached_row_groups: int = 4) -> None:
    """Make MiniMind-O's OmniDataset build a lazy table instead of a full one.

    Done by replacing ``__init__``'s table construction rather than by editing
    upstream: the dataset's own logic is the part we want to keep identical, so
    that a difference in behaviour cannot be blamed on our changes.
    """
    from dataset.omni_dataset import OmniDataset  # type: ignore[import-not-found]

    original = OmniDataset.__init__

    def patched(self: Any, data_path: str, *args: Any, **kwargs: Any) -> None:
        # The original reads the file to build its table, which is the whole
        # problem. Point it at a one-row copy so everything else it sets up --
        # tokenizer ids, special tokens, sampling probabilities -- happens
        # exactly as upstream wrote it, then swap in the lazy table.
        stub = _single_row_copy(data_path.split(",")[0].strip())
        try:
            original(self, stub, *args, **kwargs)
        finally:
            Path(stub).unlink(missing_ok=True)
        self.table = LazyParquetTable(data_path, cached_row_groups=cached_row_groups)

    OmniDataset.__init__ = patched  # type: ignore[method-assign]
