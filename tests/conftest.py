"""Shared pytest fixtures and helpers for the LLaMA-3-Lite test suite."""
from __future__ import annotations

import importlib
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

# Inject a minimal `wandb` stub into sys.modules when the real package is
# not installed, so that `train.py`'s module-level `import wandb` succeeds
# on a CPU/Mac dev box and the test `monkeypatch.setattr(wandb, "log", ...)`
# has a real attribute to patch. The stub records calls in-memory; tests
# assert against `wandb._calls["log"]` if needed (the live test uses
# monkeypatch.setattr on top, which still works).
if "wandb" not in sys.modules:
    try:
        importlib.import_module("wandb")
    except ImportError:
        _wandb_stub = types.ModuleType("wandb")
        _calls: dict[str, list] = {"log": [], "init": [], "finish": []}

        def _log(*args, **kwargs):
            _calls["log"].append((args, kwargs))

        def _init(*args, **kwargs):
            _calls["init"].append((args, kwargs))
            return None

        def _finish():
            _calls["finish"].append(True)

        class _Table:
            def __init__(self, columns=None):
                self.columns = columns or []
                self.rows: list = []

            def add_data(self, *values):
                self.rows.append(values)

        _wandb_stub.log = _log
        _wandb_stub.init = _init
        _wandb_stub.finish = _finish
        _wandb_stub.Table = _Table
        _wandb_stub._calls = _calls
        sys.modules["wandb"] = _wandb_stub


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default=None,
        choices=("cpu", "cuda"),
        help="Force the device used by tests (default: cpu unless --run-gpu).",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked `gpu` (skipped by default).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu and a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def device(request) -> torch.device:
    """Default device used by tests (honors --device; defaults to cpu)."""
    requested = request.config.getoption("--device")
    if requested is None:
        requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return torch.device(requested)


@pytest.fixture(scope="session")
def dtype(device: torch.device) -> torch.dtype:
    """Use float32 on CPU for numerical exactness; bf16 only on GPU."""
    return torch.float32 if device.type == "cpu" else torch.bfloat16


@pytest.fixture
def seed_everything():
    """Seed torch / numpy / python RNG and return the seed used."""
    def _seed(seed: int = 1234):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return seed
    return _seed


@pytest.fixture(scope="session")
def full_config() -> dict:
    """The production config straight from config.get_config()."""
    from config import get_config
    return get_config()


@pytest.fixture
def tiny_config() -> dict:
    """A small, CPU-friendly config for fast tests."""
    return {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 16,
        "d_ff": 128,
        "vocab_size": 256,
        "seq_len": 32,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "batch_size": 4,
        "gradient_accumulation": 1,
        "max_steps": 10,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 2,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "optimizer": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "compile_model": False,
        "gradient_checkpointing": False,
        "use_chunked_cross_entropy": True,
        "tf32": False,
        "cudnn_benchmark": False,
        "data_sources": {},
        "num_workers": 0,
        "pin_memory": False,
        "target_tokens": 4096,
        "data_cache_dir": "data_cache_test",
        "data_cache_filename": "tokens_test.bin",
        "reuse_data_cache": False,
        "shuffle_documents": True,
        "shuffle_seed": 42,
        "dedup": True,
        "dedup_hash_bytes": 16,
        "min_doc_tokens": 4,
        "max_doc_tokens": 64,
        "tokenizer_name": "NousResearch/Meta-Llama-3-8B",
        "tokenizer_cache_dir": None,
        "val_interval": 1000,
        "val_max_batches": 2,
        "val_split": 0.1,
        "generation_interval": 1000,
        "generation_max_tokens": 8,
        "generation_temperature": 0.8,
        "generation_top_k": 20,
        "model_folder": "weights_test",
        "model_filename": "tiny",
        "checkpoint_interval": 1000,
        "keep_last_n_checkpoints": 2,
        "async_checkpoint": False,
        "preload": None,
        "wandb_project": "test",
        "wandb_entity": None,
        "wandb_tags": ["test"],
        "log_interval": 1,
    }


@pytest.fixture
def tiny_model(tiny_config, device, dtype, seed_everything):
    """Build a tiny Transformer on the requested device/dtype."""
    from model import build_transformer
    seed_everything(1234)
    model = build_transformer(
        vocab_size=tiny_config["vocab_size"],
        d_model=tiny_config["d_model"],
        n_layers=tiny_config["n_layers"],
        n_heads=tiny_config["n_heads"],
        n_kv_heads=tiny_config["n_kv_heads"],
        head_dim=tiny_config["head_dim"],
        d_ff=tiny_config["d_ff"],
        max_seq_len=tiny_config["seq_len"],
        rope_theta=tiny_config["rope_theta"],
        rms_norm_eps=tiny_config["rms_norm_eps"],
        gradient_checkpointing=tiny_config["gradient_checkpointing"],
    ).to(device=device, dtype=dtype)
    return model


@pytest.fixture
def weights_dir(tmp_path, monkeypatch) -> Path:
    """Redirect config['model_folder'] into a tmp dir so tests don't pollute the repo."""
    d = tmp_path / "weights"
    d.mkdir()
    return d


def make_token_stream(num_tokens: int, vocab_size: int, seq_len: int,
                      eos_id: int = 0, bos_id: int = 1, seed: int = 42) -> np.ndarray:
    """Build a synthetic uint32 token buffer packed with BOS..EOS documents."""
    rng = np.random.default_rng(seed)
    doc_len = max(8, seq_len // 2)
    out: list[int] = []
    while len(out) < num_tokens:
        out.append(bos_id)
        body = rng.integers(2, max(3, vocab_size), size=doc_len - 2).tolist()
        out.extend(body)
        out.append(eos_id)
    return np.asarray(out[:num_tokens], dtype=np.uint32)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())