"""Tests for ``config.py``."""
from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    cleanup_old_checkpoints,
    get_config,
    get_weights_file_path,
    latest_weights_file_path,
)


REQUIRED_KEYS = {
    "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "d_ff",
    "vocab_size", "seq_len", "rope_theta", "rms_norm_eps", "dropout",
    "tie_embeddings", "bias",
    "batch_size", "gradient_accumulation", "max_steps", "learning_rate",
    "min_lr", "warmup_steps", "weight_decay", "max_grad_norm",
    "beta1", "beta2", "eps",
    "dtype", "use_flash_attention", "compile_model",
    "gradient_checkpointing", "use_chunked_cross_entropy",
    "tf32", "cudnn_benchmark", "cuda_alloc_conf",
    "data_sources", "num_workers", "prefetch_factor", "pin_memory",
    "document_packing", "target_tokens", "data_cache_dir",
    "data_cache_filename", "reuse_data_cache", "shuffle_documents",
    "shuffle_seed", "dedup", "dedup_hash_bytes", "min_doc_tokens",
    "max_doc_tokens", "tokenize_batch_size", "tokenizer_name",
    "tokenizer_type", "tokenizer_cache_dir",
    "val_interval", "val_max_batches", "val_split",
    "generation_interval", "generation_max_tokens",
    "generation_temperature", "generation_top_k",
    "model_folder", "model_filename", "checkpoint_interval",
    "keep_last_n_checkpoints", "async_checkpoint", "preload",
    "wandb_project", "wandb_entity", "wandb_tags", "log_interval",
    "top_k", "temperature",
    "optimizer", "lr_scheduler", "warmup_style",
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


class TestGetWeightsFilePath:
    def test_constructs_expected_path(self, full_config):
        path = get_weights_file_path(full_config, step=5000)
        assert path.endswith(f"{full_config['model_filename']}_step_5000.pt")
        assert path.startswith(full_config["model_folder"])

    def test_step_zero(self, full_config):
        path = get_weights_file_path(full_config, step=0)
        assert path.endswith("_step_0.pt")


class TestLatestWeightsFilePath:
    def test_returns_none_when_folder_missing(self, full_config, tmp_path):
        full_config = {**full_config, "model_folder": str(tmp_path / "nope")}
        assert latest_weights_file_path(full_config) is None

    def test_returns_none_when_empty(self, full_config, tmp_path):
        full_config = {**full_config, "model_folder": str(tmp_path)}
        assert latest_weights_file_path(full_config) is None

    def test_picks_single_checkpoint(self, full_config, tmp_path):
        (tmp_path / "llama3-515M_step_100.pt").touch()
        full_config = {**full_config, "model_folder": str(tmp_path),
                       "model_filename": "llama3-515M"}
        result = latest_weights_file_path(full_config)
        assert result is not None
        assert result.endswith("llama3-515M_step_100.pt")

    def test_picks_highest_step_not_lexical_max(self, full_config, tmp_path):
        """Regression: lexical sort would pick step_9.pt over step_10.pt."""
        steps = [1, 2, 9, 10, 11, 20]
        for s in steps:
            (tmp_path / f"llama3-515M_step_{s}.pt").touch()
        full_config = {**full_config, "model_folder": str(tmp_path),
                       "model_filename": "llama3-515M"}
        result = latest_weights_file_path(full_config)
        assert result is not None
        assert result.endswith("llama3-515M_step_20.pt"), (
            f"latest_weights_file_path appears to use lexical sort; "
            f"got {Path(result).name}"
        )


class TestCleanupOldCheckpoints:
    def test_noop_when_folder_missing(self, full_config, tmp_path):
        full_config = {**full_config, "model_folder": str(tmp_path / "nope")}
        cleanup_old_checkpoints(full_config, current_step=1000)

    def test_keeps_last_n_and_removes_rest(self, full_config, tmp_path):
        for s in [100, 200, 300, 400, 500]:
            (tmp_path / f"llama3-515M_step_{s}.pt").write_bytes(b"x")
        full_config = {**full_config, "model_folder": str(tmp_path),
                       "model_filename": "llama3-515M",
                       "keep_last_n_checkpoints": 2}
        cleanup_old_checkpoints(full_config, current_step=500)
        remaining = sorted(tmp_path.glob("*.pt"))
        remaining_steps = [int(p.stem.split("_step_")[-1]) for p in remaining]
        assert remaining_steps == [400, 500]

    def test_keeps_all_when_fewer_than_n(self, full_config, tmp_path):
        for s in [100, 200]:
            (tmp_path / f"llama3-515M_step_{s}.pt").write_bytes(b"x")
        full_config = {**full_config, "model_folder": str(tmp_path),
                       "model_filename": "llama3-515M",
                       "keep_last_n_checkpoints": 5}
        cleanup_old_checkpoints(full_config, current_step=200)
        assert len(list(tmp_path.glob("*.pt"))) == 2