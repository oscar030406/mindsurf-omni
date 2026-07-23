"""Shuffle within and between row groups, instead of across the whole file.

Reading the full table costs more memory than this machine has; reading rows
lazily in a globally shuffled order costs 0.16 s per row, because nearly every
read misses the cache and pulls a whole 23 MB row group to serve one row. Both
are measured, and both are unusable.

The way out is to stop fighting the layout. Shuffle the order of row groups,
shuffle the rows inside each, and read groups one at a time: every group is
decompressed once per epoch instead of once per row, and memory stays at a
couple of groups.

**This is not a global shuffle, and the difference matters.** Rows that sit in
the same row group always appear near each other within an epoch. That is
acceptable *here* only because the upstream file is already shuffled -- if the
data were sorted by speaker, language, or length, this sampler would hand the
model a biased batch every time and the loss curve would not show it. Check the
file's ordering before reusing this.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence


class RowGroupShuffleSampler:
    """Yield indices grouped by row group, shuffled at both levels.

    ``buffer_groups`` sets how many row groups are merged before shuffling.
    One means rows never cross a group boundary; higher values mix further at
    the cost of holding more groups resident.
    """

    def __init__(
        self,
        offsets: Sequence[int],
        total_rows: int,
        seed: int = 0,
        buffer_groups: int = 2,
    ) -> None:
        if buffer_groups < 1:
            raise ValueError("buffer_groups must be at least 1")
        self._offsets = list(offsets)
        self._total = total_rows
        self._seed = seed
        self._buffer_groups = buffer_groups
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """A different order each epoch; without this every epoch is identical."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._total

    def _bounds(self, slot: int) -> tuple[int, int]:
        start = self._offsets[slot]
        end = self._offsets[slot + 1] if slot + 1 < len(self._offsets) else self._total
        return start, end

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self._seed + self._epoch)
        order = list(range(len(self._offsets)))
        rng.shuffle(order)

        for position in range(0, len(order), self._buffer_groups):
            window = order[position : position + self._buffer_groups]
            indices: list[int] = []
            for slot in window:
                start, end = self._bounds(slot)
                indices.extend(range(start, end))
            rng.shuffle(indices)
            yield from indices
