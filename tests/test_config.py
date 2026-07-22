"""Tests for ``config.py``."""
from __future__ import annotations

import pytest

from config import get_config


REQUIRED_KEYS = {
    "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "d_ff",
    "vocab_size", "seq_len", "rope_theta", "rms_norm_eps",
    "batch_size", "gradient_accumulation", "max_steps", "learning_rate",
    "min_lr", "warmup_steps", "weight_decay", "max_grad_norm",
    "beta1", "beta2", "eps",
    "compile_model", "compile_mode",
    "gradient_checkpointing", "use_chunked_cross_entropy",
    "use_z_loss", "z_loss_weight", "qknorm", "use_ema", "ema_decay",
    "tf32", "cudnn_benchmark", "cuda_alloc_conf",
    "data_sources", "num_workers", "prefetch_factor", "pin_memory",
    "target_tokens", "data_cache_dir",
    "data_cache_filename", "reuse_data_cache", "shuffle_documents",
    "shuffle_seed", "dedup", "dedup_hash_bytes", "min_doc_tokens",
    "max_doc_tokens", "tokenizer_name", "tokenizer_cache_dir",
    "val_interval", "val_max_batches", "val_split",
    "generation_interval", "generation_max_tokens",
    "generation_temperature", "generation_top_k",
    "model_folder", "model_filename", "checkpoint_interval",
    "keep_last_n_checkpoints", "async_checkpoint", "preload",
    "wandb_project", "wandb_entity", "wandb_tags", "log_interval",
    "optimizer",
    # Triton dispatch keys (opt-in; force-back by default — see
    # documentation/triton_kernels.md and AGENTS.md §Hard rules).
    "cross_entropy_impl", "rmsnorm_impl", "swiglu_impl",
}


class TestGetConfig:
    def test_returns_dict(self):
        cfg = get_config()
        assert isinstance(cfg, dict)

    def test_has_all_required_keys(self, full_config):
        missing = REQUIRED_KEYS - set(full_config.keys())
        assert not missing, f"config is missing keys: {sorted(missing)}"

    def test_no_extra_unknown_keys(self, full_config):
        extra = set(full_config.keys()) - REQUIRED_KEYS
        assert extra == set(), (
            f"config has keys not covered by tests: {sorted(extra)}. "
            f"Either add tests or extend REQUIRED_KEYS."
        )

    @pytest.mark.parametrize("key,expected", [
        ("d_model", 1024), ("n_layers", 16), ("n_heads", 8),
        ("n_kv_heads", 4), ("head_dim", 128), ("d_ff", 4096),
        ("vocab_size", 128000), ("seq_len", 2048),
    ])
    def test_known_values(self, full_config, key, expected):
        assert full_config[key] == expected

    def test_gqa_heads_divide_evenly(self, full_config):
        assert full_config["n_heads"] % full_config["n_kv_heads"] == 0
        assert full_config["n_heads"] // full_config["n_kv_heads"] >= 1

    def test_data_source_weights_positive(self, full_config):
        weights = [s["weight"] for s in full_config["data_sources"].values()]
        assert all(w > 0 for w in weights), weights
        assert sum(weights) > 0
        assert 0.5 < sum(weights) <= 1.0 + 1e-9

    def test_learning_rate_schedule_invariants(self, full_config):
        assert 0 < full_config["min_lr"] < full_config["learning_rate"]
        assert 0 < full_config["warmup_steps"] < full_config["max_steps"]
        assert full_config["weight_decay"] >= 0
        assert full_config["max_grad_norm"] > 0
