import hashlib
import os
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer

_LLM_ROOT = Path(__file__).resolve().parents[2]  # .../LLM/
if str(_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_ROOT))

_TOKEN_DTYPE = np.uint32


class UniversalShardDataset(Dataset):
    """Dataset over the universal-shard corpus produced by ``shared_data``."""

    def __init__(self, seq_len: int, *, data_root=None):
        from shared_data.common import set_data_root as _set_root
        from shared_data.shard_reader import (
            load_manifest,
            open_shard_memmaps,
        )

        if data_root is not None:
            _set_root(Path(data_root))

        self.seq_len = int(seq_len)
        self.manifest = load_manifest()
        self.shards = open_shard_memmaps(self.manifest)
        if not self.shards:
            raise FileNotFoundError(
                f"No shards found under {data_root or 'data/'}. "
                "Run the universal pipeline first: "
                "python data/prepare_data.py --stage pretrain"
            )

        self._cache_idx = -1
        self._cache_arr = None

        offsets = np.cumsum([0] + [s.n_tokens for s in self.shards])
        self._shard_offsets = offsets  # len = n_shards + 1
        total_tokens = int(offsets[-1])
        self.n_chunks = max(0, total_tokens // (self.seq_len + 1))

    def _get_window(self, chunk_idx: int) -> np.ndarray:
        """Return the ``(seq_len+1)``-token window for ``chunk_idx``."""
        import bisect
        start = chunk_idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        shard_idx = bisect.bisect_right(self._shard_offsets, start) - 1
        local_start = start - int(self._shard_offsets[shard_idx])
        if local_start + self.seq_len + 1 <= self.shards[shard_idx].n_tokens:
            arr = self.shards[shard_idx].mmap
            return np.array(arr[local_start: local_start + self.seq_len + 1],
                            copy=True)
        out = np.empty(self.seq_len + 1, dtype=np.uint32)
        filled = 0
        pos = start
        while filled < self.seq_len + 1:
            shard_idx = bisect.bisect_right(self._shard_offsets, pos) - 1
            sh = self.shards[shard_idx]
            local_pos = pos - int(self._shard_offsets[shard_idx])
            take = min(self.seq_len + 1 - filled, sh.n_tokens - local_pos)
            out[filled:filled + take] = sh.mmap[local_pos: local_pos + take]
            filled += take
            pos += take
        return out

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        chunk = self._get_window(int(idx))
        return {
            'input': torch.from_numpy(chunk[:-1]).long(),
            'target': torch.from_numpy(chunk[1:]).long(),
        }


class PackedDataset(Dataset):
    """Packed dataset backed by memory-mapped uint32 file on disk."""

    def __init__(self, mmap_array: np.ndarray, seq_len: int, eos_id: int,
                 indices: np.ndarray | None = None):
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.data = mmap_array
        self.n_chunks = len(self.data) // (seq_len + 1)
        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
            if len(self.indices) > self.n_chunks:
                raise ValueError(f"indices length {len(self.indices)} exceeds n_chunks {self.n_chunks}")
        else:
            self.indices = np.arange(self.n_chunks, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        chunk_idx = int(self.indices[idx])
        start = chunk_idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = np.array(self.data[start:end], copy=True)
        return {
            'input': torch.from_numpy(chunk[:-1]).long(),
            'target': torch.from_numpy(chunk[1:]).long(),
        }


class ShuffledRangeSampler(Sampler):
    """Deterministic, resumable shuffled sampler building permutation once."""

    def __init__(self, n_chunks: int, seed: int = 42, offset: int = 0):
        self.n_chunks = n_chunks
        self.seed = seed
        self.offset = offset
        rng = np.random.default_rng(seed)
        self.indices = rng.permutation(n_chunks)

    def __iter__(self):
        for i in range(self.offset, len(self.indices)):
            yield int(self.indices[i])

    def __len__(self):
        return len(self.indices) - self.offset


def collate_fn(batch):
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return {'input': inputs, 'target': targets}


def build_tokenizer(config):
    """Load LLaMA 3 tokenizer with pad_token set to eos_token."""
    cache_dir = config.get('tokenizer_cache_dir', None)
    tokenizer = AutoTokenizer.from_pretrained(
        config['tokenizer_name'],
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _doc_hash(token_ids, n_hash_tokens: int) -> bytes:
    """Hash first n_hash_tokens for exact-dedup (tokenizer-version-independent)."""
    head = token_ids[:n_hash_tokens]
    return hashlib.sha256(np.ascontiguousarray(head).tobytes()).digest()


def _doc_passes_filter(text: str, filter_str: str, mode: str) -> bool:
    """Apply text-level filter. Modes: 'word' (whole-word) or 'substring'."""
    if mode == 'word':
        import re
        pattern = r'(?<![A-Za-z0-9_])' + re.escape(filter_str) + r'(?![A-Za-z0-9_])'
        return bool(re.search(pattern, text, flags=re.IGNORECASE))
    return filter_str in text.lower()


def _has_lang_field(example, languages) -> bool:
    """Check if example's language field matches requested languages."""
    lang = example.get('language', example.get('lang', None))
    if lang is None:
        return False
    return lang in languages


def _build_source_streams(config, sources, probs):
    """Populate sources and probs from config['data_sources'], expanding multi-source entries."""
    for name, cfg in config['data_sources'].items():
        weight = cfg.get('weight', 0.0)
        if weight <= 0:
            continue

        sub_sources = cfg.get('sources', None)
        if sub_sources is None:
            sub_sources = [cfg['source']]

        sub_weight = weight / len(sub_sources)

        for sub in sub_sources:
            if ':' in sub and not sub.startswith('http'):
                source_name, split_name = sub.split(':', 1)
            else:
                source_name, split_name = sub, cfg.get('split', 'train')

            try:
                ds = load_dataset(source_name, streaming=True, split=split_name)
            except Exception as exc:
                print(f"[data] WARNING: failed to load {source_name} ({split_name}): {exc}")
                continue

            if 'languages' in cfg:
                langs = cfg['languages']
                ds = ds.filter(lambda x, _langs=langs: _has_lang_field(x, _langs))

            if 'filter' in cfg:
                filt = cfg['filter']
                mode = cfg.get('filter_mode', 'word')
                ds = ds.filter(
                    lambda x, _f=filt, _m=mode: _doc_passes_filter(
                        x.get('text', ''), _f, _m
                    )
                )

            sources.append(ds)
            probs.append(sub_weight)


def _normalize_probs(probs):
    """Normalize weights to sum to 1.0. Raises if all weights are zero."""
    total = sum(probs)
    if total <= 0:
        raise ValueError(f"All data source weights are zero or negative (sum={total}).")
    return [p / total for p in probs]


def _stream_to_disk(config, tokenizer):
    """Stream, tokenize, dedup, and write uint32 tokens to disk; returns (train_arr, val_arr, metadata)."""
    cache_dir = Path(config.get('data_cache_dir', 'data_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / config.get('data_cache_filename', 'tokens.bin')
    meta_path = cache_path.with_suffix('.meta.txt')

    target_tokens = config.get('target_tokens', 4_000_000_000)
    min_doc_tokens = config.get('min_doc_tokens', 16)
    max_doc_tokens = config.get('max_doc_tokens', 8192)
    hash_size = config.get('dedup_hash_bytes', 256)
    do_dedup = config.get('dedup', True)
    do_shuffle = config.get('shuffle_documents', True)
    seed = config.get('shuffle_seed', 42)
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    if (config.get('reuse_data_cache', True) and cache_path.exists()
            and meta_path.exists()):
        try:
            with open(meta_path) as f:
                total_tokens = int(f.read().strip())
            mmap = np.memmap(cache_path, dtype=_TOKEN_DTYPE, mode='r', shape=(total_tokens,))
            val_split = config.get('val_split', 0.05)
            seq_len_plus_1 = config.get('seq_len', 2048) + 1
            split_token = _align_split_to_docs_and_chunks(
                mmap, target_pos=int(total_tokens * (1.0 - val_split)),
                eos_id=eos_id, seq_len_plus_1=seq_len_plus_1,
            )
            train_arr = mmap[:split_token]
            val_arr = mmap[split_token:]
            print(f"[data] Reusing cache: {total_tokens:,} tokens, "
                  f"train={len(train_arr):,}, val={len(val_arr):,}")
            return train_arr, val_arr, {
                'reused': True, 'total_tokens': total_tokens,
                'train_tokens': len(train_arr), 'val_tokens': len(val_arr),
            }
        except Exception as exc:
            print(f"[data] Cache read failed ({exc}), re-streaming...")

    sources, probs = [], []
    _build_source_streams(config, sources, probs)
    if not sources:
        raise RuntimeError("No data sources could be loaded. Check config['data_sources'].")
    probs = _normalize_probs(probs)
    print(f"[data] Interleaving {len(sources)} sources with normalized probs: "
          f"{[f'{p:.3f}' for p in probs]}")

    mixed = interleave_datasets(sources, probabilities=probs, seed=seed)

    if do_shuffle:
        mixed = mixed.shuffle(buffer_size=10_000, seed=seed)

    seen_hashes: set[bytes] = set()
    buf_capacity = int(target_tokens * 1.1)
    buf = np.zeros(buf_capacity, dtype=_TOKEN_DTYPE)
    write_pos = 0
    total_seen = 0
    total_kept = 0
    dropped_short = 0
    dropped_dup = 0

    def _write_doc(ids):
        nonlocal write_pos, total_kept
        n = len(ids)
        if write_pos + n > buf_capacity:
            return False
        buf[write_pos:write_pos + n] = ids
        write_pos += n
        total_kept += n
        return True

    for example in mixed:
        if write_pos >= target_tokens:
            break
        text = example.get('text', None)
        if not text:
            continue
        total_seen += 1

        enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_doc_tokens)
        ids = enc['input_ids']
        if len(ids) < min_doc_tokens:
            dropped_short += 1
            continue

        if do_dedup:
            h = _doc_hash(ids, hash_size)
            if h in seen_hashes:
                dropped_dup += 1
                continue
            seen_hashes.add(h)

        doc = np.empty(len(ids) + 2, dtype=_TOKEN_DTYPE)
        doc[0] = bos_id
        doc[1:-1] = np.asarray(ids, dtype=_TOKEN_DTYPE)
        doc[-1] = eos_id

        if not _write_doc(doc):
            break

    buf = buf[:write_pos]
    np.array(buf, dtype=_TOKEN_DTYPE).tofile(cache_path)
    with open(meta_path, 'w') as f:
        f.write(str(len(buf)))

    print(f"[data] Streamed {total_seen:,} docs, kept {total_kept:,} tokens "
          f"({write_pos:,} packed tokens). "
          f"Dropped: {dropped_short:,} short, {dropped_dup:,} duplicate.")
    print(f"[data] Cache written to {cache_path} ({cache_path.stat().st_size / 1e9:.2f} GB)")

    mmap = np.memmap(cache_path, dtype=_TOKEN_DTYPE, mode='r', shape=(len(buf),))
    val_split = config.get('val_split', 0.05)
    seq_len_plus_1 = config.get('seq_len', 2048) + 1
    split_token = _align_split_to_docs_and_chunks(
        mmap,
        target_pos=int(len(buf) * (1.0 - val_split)),
        eos_id=eos_id,
        seq_len_plus_1=seq_len_plus_1,
    )
    train_arr = mmap[:split_token]
    val_arr = mmap[split_token:]
    return train_arr, val_arr, {
        'reused': False, 'total_tokens': len(buf),
        'train_tokens': len(train_arr), 'val_tokens': len(val_arr),
    }


def _align_split_to_docs_and_chunks(mmap, target_pos, eos_id, seq_len_plus_1,
                                     search_window=100_000):
    """Align split to document boundary (after EOS) and chunk boundary."""
    end = min(target_pos + search_window, len(mmap))
    for i in range(target_pos, end):
        if int(mmap[i]) == eos_id:
            chunk_aligned = (i // seq_len_plus_1) * seq_len_plus_1
            return chunk_aligned
    return (target_pos // seq_len_plus_1) * seq_len_plus_1


def build_training_data(config):
    """Build training and validation dataloaders with disk-backed token cache."""
    tokenizer = build_tokenizer(config)
    seq_len = config['seq_len']

    train_arr, val_arr, meta = _stream_to_disk(config, tokenizer)
    if not meta.get('reused', False):
        print(f"[data] Cache built: train={meta['train_tokens']:,} tokens, "
              f"val={meta['val_tokens']:,} tokens")

    eos_id = tokenizer.eos_token_id
    train_dataset = PackedDataset(train_arr, seq_len, eos_id)
    val_dataset = PackedDataset(val_arr, seq_len, eos_id)

    num_workers = config.get('num_workers', 4)
    prefetch_factor = config.get('prefetch_factor', 4)
    pin_memory = config.get('pin_memory', True)
    seed = config.get('shuffle_seed', 42)
    train_sampler = ShuffledRangeSampler(train_dataset.n_chunks, seed=seed, offset=0)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False,
    )

    return train_dataloader, val_dataloader, tokenizer


def build_synthetic_data(config):
    """Generate synthetic data for smoke testing without downloading datasets."""
    tokenizer = build_tokenizer(config)
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    seq_len = config['seq_len']

    rng = random.Random(42)
    vocab_size = len(tokenizer)
    total_tokens = 100_000
    doc_len = 200
    docs = []
    for _ in range(total_tokens // doc_len):
        doc = [bos_id] + [rng.randint(0, vocab_size - 1) for _ in range(doc_len - 2)] + [eos_id]
        docs.extend(doc)

    buf = np.array(docs, dtype=_TOKEN_DTYPE)
    val_split = config.get('val_split', 0.05)
    split = int(len(buf) * (1.0 - val_split))
    chunk = seq_len + 1
    split = (split // chunk) * chunk
    train_arr = buf[:split]
    val_arr = buf[split:]

    train_dataset = PackedDataset(train_arr, seq_len, eos_id)
    val_dataset = PackedDataset(val_arr, seq_len, eos_id)

    train_sampler = ShuffledRangeSampler(train_dataset.n_chunks, seed=42, offset=0)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader, tokenizer
