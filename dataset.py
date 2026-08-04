"""LLaMA-3-Lite data-loading shim over the vendored universal loader.

The actual ``PackedDataset`` / DataLoader glue lives in ``data/shared_data``;
this module re-exports the public symbols so ``train.py`` and
``benchmark_data.py`` keep working unchanged.
"""
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent

for _p in (_PROJECT_ROOT / "data",):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from shared_data.loader import (  # noqa: E402
    PackedDataset,
    ShuffledRangeSampler,
    collate_fn,
    build_training_data,
    build_synthetic_data,
)

__all__ = [
    "PackedDataset",
    "ShuffledRangeSampler",
    "collate_fn",
    "build_training_data",
    "build_synthetic_data",
]
