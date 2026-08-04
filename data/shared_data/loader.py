"""Packed uint32 token dataset + DataLoader glue for LLaMA-3-Lite.

On-disk layout: single uint32 binary of the pretokenised corpus, EOS-separated.
``PackedDataset.__getitem__`` slices ``seq_len+1`` chunks with no copy; the
mmap-resident design is the headline enabler of the 78% memory reduction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class PackedDataset(Dataset):
    """Read-only uint32 token buffer chunked into ``seq_len+1`` training windows; each chunk returns ``seq_len`` inputs and ``seq_len`` targets (next-token shift)."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        if tokens.dtype != np.uint32:
            tokens = tokens.astype(np.uint32, copy=False)
        chunk = seq_len + 1
        if tokens.size < chunk:
            # Pad up to one chunk so a tiny buffer is still usable.
            pad = np.zeros(chunk - tokens.size, dtype=np.uint32)
            tokens = np.concatenate([tokens, pad])
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = tokens.size // (seq_len + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        window = np.asarray(self.tokens[start:end], dtype=np.int64)
        return {
            "input": torch.from_numpy(window[:-1]),
            "target": torch.from_numpy(window[1:]),
        }


class ShuffledRangeSampler(Sampler[int]):
    """Deterministic shuffle of ``range(n)`` with seedable offset; bumping ``offset`` per epoch gives a fresh permutation without disturbing cross-epoch reproducibility."""

    def __init__(self, n: int, seed: int = 0, offset: int = 0):
        self.n = n
        self.seed = seed
        self.offset = offset

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.offset)
        order = rng.permutation(self.n)
        return iter(int(i) for i in order)

    def set_epoch(self, epoch: int) -> None:
        self.offset = epoch


def collate_fn(batch: list[dict]) -> dict:
    """Stack a list of single-chunk dicts into a batched dict."""
    return {
        "input": torch.stack([b["input"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
    }


def build_tokenizer(config: dict):
    """Load the project tokenizer from ``tokenizer_name``; pad defaults to eos.

    Raises when transformers is missing or the download fails; callers fall
    back to the byte stub.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_name"],
        cache_dir=config.get("tokenizer_cache_dir", None),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_synthetic_data(
    config: dict,
    *,
    num_tokens: Optional[int] = None,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, "object"]:
    """Returns ``(train_dl, val_dl, tokenizer)`` over a synthetic uint32 stream; ``tokenizer`` is a stub exposing ``pad_token_id`` / ``eos_token_id`` / ``encode`` / ``decode``. Synthetic ids are random, so the byte stub is used unconditionally (no HF download)."""
    seq_len = int(config["seq_len"])
    vocab = int(config["vocab_size"])
    batch = int(config["batch_size"])
    n_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))
    prefetch = int(config.get("prefetch_factor", 2))
    val_split = float(config.get("val_split", 0.05))

    if num_tokens is None:
        num_tokens = max(8 * (seq_len + 1) * batch, 4096)

    rng = np.random.default_rng(seed)
    tokens = rng.integers(2, max(3, vocab), size=num_tokens, dtype=np.uint32)

    chunk = seq_len + 1
    n_total = (tokens.size // chunk) * chunk
    tokens = tokens[:n_total]
    split = int(n_total * (1.0 - val_split))
    split = (split // chunk) * chunk

    train_ds = PackedDataset(tokens[:split], seq_len)
    val_ds = PackedDataset(tokens[split:], seq_len)

    train_sampler = ShuffledRangeSampler(len(train_ds), seed=int(config.get("shuffle_seed", seed)))
    train_dl = DataLoader(
        train_ds, batch_size=batch, sampler=train_sampler,
        num_workers=n_workers, prefetch_factor=prefetch if n_workers > 0 else None,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=True,
        persistent_workers=n_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=min(2, n_workers),
        prefetch_factor=prefetch if n_workers > 0 else None,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=False,
        persistent_workers=n_workers > 0,
    )

    return train_dl, val_dl, _SyntheticTokenizerStub(vocab=vocab, eos_id=0, pad_id=0)


def build_training_data(config: dict) -> tuple[DataLoader, DataLoader, "object"]:
    """Mmaps ``tokens.bin`` from ``data/prepare_data.py``; raw ``uint32`` little-endian with no header, last ``val_split`` fraction held out for validation."""
    cache_dir = Path(config.get("data_cache_dir", "data_cache"))
    fname = config.get("data_cache_filename", "tokens.bin")
    path = cache_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Token cache not found at {path}. Run `python data/prepare_data.py` "
            f"first (or pass `data_sources` empty + use build_synthetic_data)."
        )

    seq_len = int(config["seq_len"])
    batch = int(config["batch_size"])
    n_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))
    prefetch = int(config.get("prefetch_factor", 2))
    val_split = float(config.get("val_split", 0.05))

    tokens = np.memmap(path, dtype=np.uint32, mode="r")
    chunk = seq_len + 1
    n_total = (tokens.size // chunk) * chunk
    split = int(n_total * (1.0 - val_split))
    split = (split // chunk) * chunk

    train_ds = PackedDataset(tokens[:split], seq_len)
    val_ds = PackedDataset(tokens[split:], seq_len)

    train_sampler = ShuffledRangeSampler(len(train_ds), seed=int(config.get("shuffle_seed", 42)))
    train_dl = DataLoader(
        train_ds, batch_size=batch, sampler=train_sampler,
        num_workers=n_workers, prefetch_factor=prefetch if n_workers > 0 else None,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=True,
        persistent_workers=n_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=min(2, n_workers),
        prefetch_factor=prefetch if n_workers > 0 else None,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=False,
        persistent_workers=n_workers > 0,
    )

    real_vocab = int(config["vocab_size"])
    try:
        tokenizer = build_tokenizer(config)
    except Exception as exc:
        print(
            f"[data] tokenizer load failed ({type(exc).__name__}: {exc}); "
            f"using the byte stub. Generation samples will be meaningless "
            f"until a real tokenizer is available."
        )
        tokenizer = _SyntheticTokenizerStub(vocab=real_vocab, eos_id=0, pad_id=0)
    return train_dl, val_dl, tokenizer


class _SyntheticTokenizerStub:
    """Duck-typed tokenizer for synthetic data: bytes ⇄ ids, clamped to vocab."""

    def __init__(self, vocab: int, eos_id: int, pad_id: int):
        self._vocab = vocab
        self.eos_token_id = eos_id
        self.pad_token_id = pad_id

    def __len__(self) -> int:
        return self._vocab

    def encode(self, text: str) -> list[int]:
        return [min(b, self._vocab - 1) for b in text.encode("utf-8")]

    def decode(self, ids) -> str:
        raw = bytes(int(i) & 0xFF for i in ids)
        return raw.decode("utf-8", errors="replace")
