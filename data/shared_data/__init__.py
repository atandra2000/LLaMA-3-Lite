"""In-tree ``shared_data`` package for LLaMA-3-Lite.

Mirrors the public surface imported by ``dataset.py`` and
``data/prepare_data.py``: ``PackedDataset``, ``ShuffledRangeSampler``,
``collate_fn``, ``build_synthetic_data``, ``build_training_data``.
"""
from .loader import (
    PackedDataset,
    ShuffledRangeSampler,
    collate_fn,
    build_synthetic_data,
    build_training_data,
)

__all__ = [
    "PackedDataset",
    "ShuffledRangeSampler",
    "collate_fn",
    "build_synthetic_data",
    "build_training_data",
]
