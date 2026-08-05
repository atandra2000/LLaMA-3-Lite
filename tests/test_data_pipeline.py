"""Regression tests for the data-pipeline wiring (audit C1/C2).

Covers the two broken documented commands found by docs/AUDIT.md:
- ``benchmark_data.py`` crashed on a stale ``eos_id`` kwarg (C1).
- nothing produced ``data_cache/tokens.bin`` from the workspace shards (C2).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _write_fake_shards(tmp_path: Path, n_shards: int = 3,
                       tokens_per_shard: int = 10_000) -> Path:
    """Write ``shards/shard_*.bin`` + ``shards/manifest.json`` like the
    workspace pack stage does; returns the shards dir."""
    shards_dir = tmp_path / "data" / "shards"
    shards_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    shards = []
    for i in range(n_shards):
        data = rng.integers(2, 1000, size=tokens_per_shard, dtype=np.uint32)
        (shards_dir / f"shard_{i:05d}.bin").write_bytes(data.tobytes())
        shards.append({
            "index": i,
            "path": f"shard_{i:05d}.bin",
            "n_tokens": tokens_per_shard,
            "sha256": "",
            "n_eos": 0,
        })
    manifest = {
        "version": "1.0.0",
        "vocab_size": 128_000,
        "eos_token_id": 128_009,
        "dtype": "uint32",
        "shard_size_tokens": tokens_per_shard,
        "total_tokens": n_shards * tokens_per_shard,
        "shard_count": n_shards,
        "shards_dir": "data/shards",
        "shards": shards,
        "sources": {},
    }
    (shards_dir / "manifest.json").write_text(json.dumps(manifest))
    return shards_dir


class TestConcatShardsToCache:
    def test_concatenates_in_manifest_order(self, tmp_path):
        from data.prepare_data import concat_shards_to_cache

        shards_dir = _write_fake_shards(tmp_path, n_shards=3)
        cache = tmp_path / "data_cache" / "tokens.bin"

        rc = concat_shards_to_cache(
            shards_dir, shards_dir / "manifest.json", cache)

        assert rc == 0
        assert cache.exists()
        got = np.fromfile(cache, dtype=np.uint32)
        # Byte-exact concat of the three shards in index order.
        expect = np.concatenate([
            np.fromfile(shards_dir / f"shard_{i:05d}.bin", dtype=np.uint32)
            for i in range(3)
        ])
        assert got.size == expect.size
        assert np.array_equal(got, expect)
        # No leftover temp file after the atomic rename.
        assert not cache.with_name(cache.name + ".tmp").exists()

    def test_missing_manifest_raises(self, tmp_path):
        from data.prepare_data import concat_shards_to_cache

        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        with pytest.raises(SystemExit):
            concat_shards_to_cache(
                shards_dir, shards_dir / "manifest.json",
                tmp_path / "tokens.bin")

    def test_empty_manifest_lists_no_shards(self, tmp_path):
        from data.prepare_data import concat_shards_to_cache

        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        manifest = shards_dir / "manifest.json"
        manifest.write_text(json.dumps({"shards": []}))
        with pytest.raises(SystemExit):
            concat_shards_to_cache(
                shards_dir, manifest, tmp_path / "tokens.bin")

    def test_cache_mmaps_cleanly(self, tmp_path):
        """The concat output must satisfy the loader's mmap contract."""
        from data.prepare_data import concat_shards_to_cache

        shards_dir = _write_fake_shards(tmp_path)
        cache = tmp_path / "data_cache" / "tokens.bin"
        concat_shards_to_cache(shards_dir, shards_dir / "manifest.json", cache)

        mm = np.memmap(cache, dtype=np.uint32, mode="r")
        assert mm.size == 3 * 10_000
        assert int(mm[0]) >= 0


class TestBenchmarkData:
    def test_runs_end_to_end(self):
        """`python benchmark_data.py` must not crash (audit finding C1)."""
        proc = subprocess.run(
            [sys.executable, str(ROOT / "benchmark_data.py"),
             "--steps", "2", "--batch_size", "4", "--seq_len", "64",
             "--num_workers", "0"],
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, (
            f"benchmark_data.py failed:\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        assert "throughput" in proc.stdout

    def test_json_mode(self):
        import json as _json

        proc = subprocess.run(
            [sys.executable, str(ROOT / "benchmark_data.py"),
             "--steps", "1", "--batch_size", "2", "--seq_len", "32",
             "--num_workers", "0", "--json"],
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        metrics = _json.loads(proc.stdout)
        assert metrics["steps"] == 1
        assert metrics["tokens_per_step"] == 64
