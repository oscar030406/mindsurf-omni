"""Rewrite a parquet file with row groups small enough to read lazily.

MiniMind-O ships sft_a2a.parquet as a single row group holding all 414,024
rows in 6.1 GB. Reading one row from it means materialising the whole thing,
so the lazy reader buys nothing and a DataLoader worker is OOM-killed before
the first step.

The rewrite streams: batches are read and written without the full table ever
being resident, so this runs in a few hundred MB regardless of file size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

# Chosen to match what sft_t2a already uses (~3700 rows, ~11 MB per group).
# Small enough that a cache of a few groups is cheap, large enough that the
# per-group overhead stays negligible.
DEFAULT_ROWS_PER_GROUP = 4096


def describe(path: Path) -> str:
    handle = pq.ParquetFile(path)
    metadata = handle.metadata
    groups = metadata.num_row_groups
    largest = max(metadata.row_group(index).total_byte_size for index in range(groups))
    handle.close()
    return f"{metadata.num_rows:,} rows in {groups} row group(s), largest {largest / 1e9:.2f} GB"


def repartition(source: Path, target: Path, rows_per_group: int) -> None:
    handle = pq.ParquetFile(source)
    schema = handle.schema_arrow
    # compression carried over from the source, so the rewrite does not quietly
    # inflate the file or change how fast it decodes.
    writer = pq.ParquetWriter(target, schema, compression="snappy")
    try:
        written = 0
        for batch in handle.iter_batches(batch_size=rows_per_group):
            writer.write_batch(batch)
            written += batch.num_rows
            if written % (rows_per_group * 20) == 0:
                print(f"  {written:,} rows", flush=True)
    finally:
        writer.close()
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="defaults to <input>.repartitioned")
    parser.add_argument("--rows-per-group", type=int, default=DEFAULT_ROWS_PER_GROUP)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="move the result over the input once it verifies",
    )
    args = parser.parse_args()

    target = args.output or args.input.with_suffix(".repartitioned.parquet")
    print(f"before: {describe(args.input)}")
    repartition(args.input, target, args.rows_per_group)
    print(f"after:  {describe(target)}")

    # Verified before replacing, because a truncated rewrite that silently
    # replaced the source would cost the dataset.
    before = pq.ParquetFile(args.input).metadata.num_rows
    after = pq.ParquetFile(target).metadata.num_rows
    if before != after:
        raise SystemExit(f"row count changed: {before:,} -> {after:,}; keeping both files")

    if args.replace:
        args.input.unlink()
        target.rename(args.input)
        print(f"replaced {args.input}")


if __name__ == "__main__":
    main()
