"""LLaMA-3-Lite data loading — thin shim over the universal pipeline.

All data *preparation* (download, clean, tokenise, pack, dedup) and the
training *loader* now live in ``shared_data`` (the single source of truth
for the whole portfolio). This module only re-exports the canonical
symbols so existing call sites (``train.py``, ``test_pipeline.py``,
``benchmark_data.py``) keep working unchanged.

To prepare a real corpus, run the shim first::

    python data/prepare_data.py          # delegates to shared_data.run_pipeline
    python train.py                       # build_training_data reads the shards
"""
import sys
from pathlib import Path

# Ensure the shared_data package (at the LLM/ root) is importable.
_LLM_ROOT = Path(__file__).resolve().parents[1]  # .../LLM/
if str(_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_ROOT))

from shared_data.loader import (
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
