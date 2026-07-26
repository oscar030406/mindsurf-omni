"""Launch MiniMind-O's trainer with the data path this machine can afford.

Two substitutions, applied before the trainer runs, so upstream's training
logic stays byte-identical and any difference in results cannot be blamed on
our edits:

* the dataset reads row groups on demand instead of materialising the table,
  which otherwise exceeds this machine's memory and is OOM-killed before the
  first step;
* the shuffle respects row-group boundaries, because a globally shuffled read
  over a lazy table costs 0.16 s per row -- every read misses the cache and
  pulls 23 MB to serve one row.

Measured on sft_t2a.parquet (1,248,923 rows):

    full materialisation      OOM-killed
    lazy + global shuffle     0.164  s/row
    lazy + row-group shuffle  0.00003 s/row, 0.65 GB peak

The compromise is stated in row_group_sampler: this is not a global shuffle,
and it is only sound because the upstream file is already shuffled.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def force_thinker_shape(intermediate_size: int, num_key_value_heads: int) -> None:
    """Build the Thinker at our base's shape, and refuse silently to not.

    The trainer exposes only --hidden_size and --num_hidden_layers, so the rest
    of the config comes from MiniMind's defaults: GQA with four key-value heads
    and a feed-forward width derived from hidden_size. Our base is MHA with a
    wider feed-forward, and the loader uses strict=False -- so without this,
    forty weight tensors are skipped for shape mismatch, left at random init,
    and training proceeds while reporting the wrong parameter count.

    That failure is silent and expensive: the run looks healthy and the base
    contributed almost nothing.

    Patched at MiniMindConfig rather than at each construction site, because
    the Talker is built from its own MiniMindConfig that takes only
    hidden_size. Leaving that one narrow while the Thinker widens breaks the
    seeding step, which copies the Thinker's upper layers into the Talker.
    """
    from model.model_minimind import MiniMindConfig  # type: ignore[import-not-found]

    original = MiniMindConfig.__init__
    # A grafted checkpoint carries upstream's Talker, built at upstream's shape.
    # Widening it here would make those 20 tensors mismatch, and the loader
    # skips mismatched tensors *silently* -- the graft would evaporate and the
    # run would train a freshly copied Talker while looking healthy. So when the
    # Talker comes from elsewhere, the override applies to the Thinker only.
    #
    # Discriminated by class, not by call site: OmniConfig is the Thinker's
    # config, while the Talker builds a plain MiniMindConfig with hidden_size
    # and use_moe alone.
    thinker_only = os.environ.get("MINDSURF_TALKER_SHAPE", "").strip() == "upstream"

    def patched(self: Any, *args: Any, **kwargs: Any) -> None:
        if not (thinker_only and type(self) is MiniMindConfig):
            kwargs.setdefault("intermediate_size", intermediate_size)
            kwargs.setdefault("num_key_value_heads", num_key_value_heads)
        original(self, *args, **kwargs)

    MiniMindConfig.__init__ = patched  # type: ignore[method-assign]
    if thinker_only:
        print(
            "[mindsurf] MINDSURF_TALKER_SHAPE=upstream: Talker keeps upstream's shape, "
            "so a grafted checkpoint loads instead of being silently skipped",
            flush=True,
        )


def verify_base_loaded(minimind_root: Path) -> None:
    """Fail loudly if the checkpoint did not actually go into the model."""
    from trainer import trainer_utils  # type: ignore[import-not-found]

    original = trainer_utils.init_omni_model

    def patched(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        thinker = sum(p.numel() for p in model.thinker.parameters())
        if thinker < 80_000_000:
            raise SystemExit(
                f"the Thinker has {thinker:,} parameters, but our base has 89,864,448 -- "
                "the checkpoint did not load into this shape"
            )
        print(f"[mindsurf] thinker {thinker:,} parameters", flush=True)
        return result

    trainer_utils.init_omni_model = patched


def install(minimind_root: Path, buffer_groups: int = 2, cached_row_groups: int = 3) -> None:
    sys.path.insert(0, str(minimind_root))
    os.chdir(minimind_root / "trainer")

    force_thinker_shape(intermediate_size=3584, num_key_value_heads=8)
    verify_base_loaded(minimind_root)

    import torch

    from mindsurf_omni.data.lazy_parquet import LazyParquetTable, patch_omni_dataset
    from mindsurf_omni.data.row_group_sampler import RowGroupShuffleSampler

    patch_omni_dataset(cached_row_groups=cached_row_groups)

    # The trainer shuffles with torch.randperm(len(train_ds)) and feeds the
    # result to SkipBatchSampler. Replacing randperm for the one call whose
    # length matches the dataset keeps that structure intact while changing the
    # order it produces.
    original_randperm = torch.randperm
    state: dict[str, object] = {"epoch": -1}

    def randperm(n: int, *args: object, **kwargs: object) -> torch.Tensor:
        dataset = state.get("dataset")
        if dataset is None or n != len(dataset):  # type: ignore[arg-type]
            return original_randperm(n, *args, **kwargs)  # type: ignore[arg-type]
        state["epoch"] = int(state["epoch"]) + 1  # type: ignore[arg-type]
        sampler = RowGroupShuffleSampler(
            dataset.table.row_group_offsets,  # type: ignore[union-attr]
            n,
            seed=42,
            buffer_groups=buffer_groups,
        )
        sampler.set_epoch(int(state["epoch"]))  # type: ignore[arg-type]
        print(f"[mindsurf] row-group shuffle, epoch {state['epoch']}", flush=True)
        return torch.tensor(list(sampler), dtype=torch.long)

    torch.randperm = randperm  # type: ignore[assignment]

    # The dataset is constructed inside the trainer, so it is captured on the
    # way past rather than passed in.
    from dataset.omni_dataset import OmniDataset  # type: ignore[import-not-found]

    constructed = OmniDataset.__init__

    def capture(self: object, *args: object, **kwargs: object) -> None:
        constructed(self, *args, **kwargs)  # type: ignore[arg-type]
        if isinstance(getattr(self, "table", None), LazyParquetTable):
            state["dataset"] = self

    OmniDataset.__init__ = capture  # type: ignore[method-assign]


def main() -> None:
    minimind_root = Path(
        os.environ.get("MINIMIND_O_ROOT", Path.home() / "omni" / "minimind-o")
    ).resolve()
    install(
        minimind_root,
        buffer_groups=int(os.environ.get("MINDSURF_BUFFER_GROUPS", "2")),
        cached_row_groups=int(os.environ.get("MINDSURF_CACHED_GROUPS", "3")),
    )
    sys.argv = [str(minimind_root / "trainer" / "train_sft_omni.py"), *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
