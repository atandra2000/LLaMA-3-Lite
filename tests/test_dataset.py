"""Tests for ``dataset.py``.

Most tests use synthetic in-memory token buffers so they don't depend on
HuggingFace network access or a real tokenizer. The tokenizer-dependent
helpers are tested separately via ``build_synthetic_data`` (which the README
itself advertises as the offline smoke test) — but only when the tokenizer
can actually be loaded; otherwise those tests are skipped.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import make_token_stream

import dataset as ds


# --------------------------------------------------------------------------- #
# Helpers / pure functions
# --------------------------------------------------------------------------- #
class TestDocHash:
    def test_deterministic(self):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        h1 = ds._doc_hash(ids, n_hash_tokens=5)
        h2 = ds._doc_hash(ids, n_hash_tokens=5)
        assert h1 == h2

    def test_only_head_is_hashed(self):
        """Documents that differ only beyond n_hash_tokens hash the same."""
        head = np.array([10, 20, 30], dtype=np.uint32)
        a = np.concatenate([head, [999]])
        b = np.concatenate([head, [1]])
        assert ds._doc_hash(a, 3) == ds._doc_hash(b, 3)

    def test_short_doc_pads_with_available_tokens(self):
        # No crash when doc is shorter than n_hash_tokens.
        ids = np.array([1, 2], dtype=np.uint32)
        h = ds._doc_hash(ids, n_hash_tokens=16)
        assert isinstance(h, bytes)
        # And equals hashing the full short array.
        assert h == hashlib.sha256(ids.tobytes()).digest()


class TestDocFilter:
    def test_word_mode_whole_word(self):
        assert ds._doc_passes_filter("the code is great", "code", "word")
        # 'code' as substring of 'codec' must NOT match in word mode.
        assert not ds._doc_passes_filter("a codec file", "code", "word")

    def test_word_mode_case_insensitive(self):
        assert ds._doc_passes_filter("PYTHON is fun", "python", "word")

    def test_word_mode_word_boundary(self):
        assert ds._doc_passes_filter("def foo():\n  pass", "def", "word")
        assert not ds._doc_passes_filter("define foo()", "def", "word")

    def test_substring_mode(self):
        assert ds._doc_passes_filter("a codec here", "code", "substring")
        # Substring mode lowercases both sides.
        assert ds._doc_passes_filter("SomeTHING", "something", "substring")


class TestHasLangField:
    def test_language_key(self):
        assert ds._has_lang_field({"language": "Python"}, ["Python"])
        assert not ds._has_lang_field({"language": "Python"}, ["Go"])

    def test_lang_key(self):
        # Some datasets use 'lang' instead of 'language'.
        assert ds._has_lang_field({"lang": "Rust"}, ["Rust"])

    def test_missing_key(self):
        assert not ds._has_lang_field({"text": "x"}, ["Python"])


class TestNormalizeProbs:
    def test_normalizes_to_one(self):
        out = ds._normalize_probs([0.5, 0.3, 0.2])
        assert abs(sum(out) - 1.0) < 1e-12
        assert out == pytest.approx([0.5, 0.3, 0.2])

    def test_unnormalized_input(self):
        out = ds._normalize_probs([2, 2, 4])
        assert out == pytest.approx([0.25, 0.25, 0.5])

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            ds._normalize_probs([0.0, 0.0])

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            ds._normalize_probs([-1.0, -2.0])


# --------------------------------------------------------------------------- #
# Split alignment (no tokenizer needed)
# --------------------------------------------------------------------------- #
class TestAlignSplit:
    def test_returns_chunk_aligned_eos_position(self):
        seq_len = 32
        seq_len_plus_1 = seq_len + 1
        # Build a buffer where EOS appears at position 60.
        buf = np.zeros(200, dtype=np.uint32)
        buf[60] = 99      # eos id
        target_pos = 50   # before the EOS
        split = ds._align_split_to_docs_and_chunks(
            buf, target_pos=target_pos, eos_id=99,
            seq_len_plus_1=seq_len_plus_1, search_window=100,
        )
        # Should align to (60 // 33) * 33 = 33.
        assert split == (60 // seq_len_plus_1) * seq_len_plus_1
        assert split % seq_len_plus_1 == 0

    def test_falls_back_to_chunk_aligned_when_no_eos_in_window(self):
        seq_len = 32
        buf = np.zeros(500, dtype=np.uint32)  # no EOS at all
        target_pos = 100
        split = ds._align_split_to_docs_and_chunks(
            buf, target_pos=target_pos, eos_id=99,
            seq_len_plus_1=seq_len + 1, search_window=10,
        )
        # No EOS in the 10-token window -> chunk-align target_pos itself.
        assert split == (target_pos // (seq_len + 1)) * (seq_len + 1)

    def test_split_within_buffer_bounds(self):
        seq_len = 16
        buf = np.zeros(100, dtype=np.uint32)
        buf[80] = 99
        split = ds._align_split_to_docs_and_chunks(
            buf, target_pos=70, eos_id=99, seq_len_plus_1=seq_len + 1,
            search_window=100,
        )
        assert 0 <= split < len(buf)


# --------------------------------------------------------------------------- #
# PackedDataset (no tokenizer needed — uses raw uint32 arrays)
# --------------------------------------------------------------------------- #
class TestPackedDataset:
    def test_n_chunks_correct(self):
        seq_len = 8
        data = np.arange(50, dtype=np.uint32)   # 50 tokens -> 50 // 9 = 5 chunks
        ds_ = ds.PackedDataset(data, seq_len=seq_len, eos_id=0)
        assert ds_.n_chunks == 50 // (seq_len + 1)
        assert len(ds_) == ds_.n_chunks

    def test_input_target_shifted_by_one(self):
        seq_len = 4
        # 5 tokens per chunk: [0,1,2,3,4] -> input [0,1,2,3], target [1,2,3,4]
        data = np.arange(15, dtype=np.uint32)   # 3 chunks
        ds_ = ds.PackedDataset(data, seq_len=seq_len, eos_id=0)
        item = ds_[0]
        assert torch.equal(item["input"], torch.arange(0, 4, dtype=torch.long))
        assert torch.equal(item["target"], torch.arange(1, 5, dtype=torch.long))

    def test_indices_subset(self):
        seq_len = 4
        data = np.arange(15, dtype=np.uint32)  # 3 chunks
        idx = np.array([2, 0, 1], dtype=np.int64)
        ds_ = ds.PackedDataset(data, seq_len=seq_len, eos_id=0, indices=idx)
        assert len(ds_) == 3
        # __getitem__ uses the supplied indices order.
        out = ds_[0]
        # chunk_idx = indices[0] = 2 -> tokens [10..14]
        assert out["input"][0].item() == 10

    def test_indices_too_long_raises(self):
        seq_len = 4
        data = np.arange(15, dtype=np.uint32)   # 3 chunks
        idx = np.arange(5, dtype=np.int64)      # 5 > 3
        with pytest.raises(ValueError):
            ds.PackedDataset(data, seq_len=seq_len, eos_id=0, indices=idx)

    def test_getitem_returns_long_tensors(self):
        seq_len = 4
        data = np.arange(10, dtype=np.uint32)
        ds_ = ds.PackedDataset(data, seq_len=seq_len, eos_id=0)
        item = ds_[0]
        assert item["input"].dtype == torch.long
        assert item["target"].dtype == torch.long
        assert item["input"].shape == (seq_len,)
        assert item["target"].shape == (seq_len,)

    def test_copies_data_no_view(self):
        """__getitem__ must copy so callers can mutate without corrupting
        the mmap (the production code uses mode='r' mmap, so this is moot in
        practice, but the contract is documented as a copy)."""
        seq_len = 4
        data = np.arange(10, dtype=np.uint32)
        ds_ = ds.PackedDataset(data, seq_len=seq_len, eos_id=0)
        item = ds_[0]
        item["input"][0] = 999
        # Next fetch should NOT see the mutation.
        item2 = ds_[0]
        assert item2["input"][0].item() != 999


class TestCollate:
    def test_stacks_inputs_and_targets(self):
        batch = [
            {"input": torch.arange(4), "target": torch.arange(1, 5)},
            {"input": torch.arange(4) * 10, "target": torch.arange(1, 5) * 10},
        ]
        out = ds.collate_fn(batch)
        assert out["input"].shape == (2, 4)
        assert out["target"].shape == (2, 4)
        assert torch.equal(out["input"][1], torch.arange(0, 40, 10))


# --------------------------------------------------------------------------- #
# ShuffledRangeSampler
# --------------------------------------------------------------------------- #
class TestShuffledRangeSampler:
    def test_length_and_contents(self):
        sampler = ds.ShuffledRangeSampler(n_chunks=10, seed=7)
        idxs = list(sampler)
        assert len(idxs) == 10
        assert sorted(idxs) == list(range(10))

    def test_deterministic_with_seed(self):
        a = list(ds.ShuffledRangeSampler(20, seed=42))
        b = list(ds.ShuffledRangeSampler(20, seed=42))
        assert a == b

    def test_different_seeds_differ(self):
        a = list(ds.ShuffledRangeSampler(50, seed=1))
        b = list(ds.ShuffledRangeSampler(50, seed=2))
        assert a != b

    def test_offset_skips_first(self):
        sampler = ds.ShuffledRangeSampler(n_chunks=10, seed=42, offset=3)
        full = list(ds.ShuffledRangeSampler(10, seed=42))
        assert list(sampler) == full[3:]
        assert len(sampler) == 7


# --------------------------------------------------------------------------- #
# build_synthetic_data — requires the LLaMA-3 tokenizer (network/disk).
# Skipped if the tokenizer can't be loaded offline.
# --------------------------------------------------------------------------- #
def _tokenizer_available() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B",
                                       local_files_only=True)
        return True
    except Exception:
        return False


_HAS_TOKENIZER = _tokenizer_available()


@pytest.mark.skipif(not _HAS_TOKENIZER,
                    reason="LLaMA-3 tokenizer not cached locally")
class TestBuildSyntheticData:
    def test_returns_dataloaders_and_tokenizer(self, tiny_config, tmp_path,
                                                monkeypatch):
        # Avoid writing into the real repo dir.
        tiny_config = {**tiny_config,
                       "data_cache_dir": str(tmp_path / "dc"),
                       "batch_size": 2}
        # build_synthetic_data downloads nothing — it generates random token
        # IDs directly — so this is safe to run offline *if* the tokenizer is
        # cached. It still needs the tokenizer for vocab size + EOS/BOS IDs.
        train, val, tok = ds.build_synthetic_data(tiny_config)
        assert tok is not None
        assert len(train) > 0
        assert len(val) > 0
        batch = next(iter(train))
        assert batch["input"].shape[1] == tiny_config["seq_len"]
        assert batch["input"].dtype == torch.long
        assert batch["target"].shape == batch["input"].shape