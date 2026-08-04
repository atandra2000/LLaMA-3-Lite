# Data Pipeline — LLaMA-3-Lite

> This project consumes the **universal 8.0B-token LLM data pipeline**
> shared by the LLM projects in the CoreProjects portfolio. The *loader*
> (mmap + DataLoader glue) is vendored in-tree at `data/shared_data/`
> so the repo is self-contained at runtime; the *preparation pipeline*
> (download → clean → tokenize → pack) is delegated to the workspace-level
> `LLM/shared_data` package.

---

## What lives in this repo

```
data/
├── prepare_data.py     ← thin shim → delegates to shared_data.prepare_data.run_pipeline
├── shared_data/        ← VENDORED LOADER ONLY (2 files: __init__.py, loader.py)
│   ├── __init__.py     ← re-exports the loader surface
│   └── loader.py       ← PackedDataset, ShuffledRangeSampler, collate_fn,
│                         build_synthetic_data, build_tokenizer, build_training_data
└── DATA_PIPELINE.md    ← this file
```

The vendored copy is intentionally **loader-only** (~7 KB). The full
preparation pipeline (`prepare_data.py`, `dedup.py`, `quality_filter.py`,
`shard_writer.py`, `manifest.py`, `config/`, `scripts/`, `documentation/`)
lives at the workspace level: `LLM/shared_data/`. `data/prepare_data.py`
imports it via `sys.path` — it resolves the workspace copy (with the
project root + `data/` on the path) and raises a clear error if the
workspace package is missing.

## Quick start

```bash
# Full pipeline (download → clean → tokenize → pack) — needs the workspace
# LLM/shared_data package importable from this machine
python3 data/prepare_data.py --stage pretrain

# Skip download (re-use an existing corpus)
python3 data/prepare_data.py --stage pretrain --skip-download

# Re-pack only (after a config change)
python3 data/prepare_data.py --stage pretrain \
    --skip-download --skip-clean --skip-tokenize
```

The pipeline writes shards under `LLM/shared_data`'s `DATA_ROOT`; the
project's `data_cache/tokens.bin` (the single uint32 file the vendored
loader mmaps) is produced by the packing stage. Running `python train.py`
without the cache falls back to synthetic data with a warning — see
[`docs/guides/quickstart.md`](../docs/guides/quickstart.md).

## Tokenizer used by LLaMA-3-Lite

| Field | Value |
|---|---|
| Family | LLaMA-3 BPE (Meta) |
| Vocab size | 128,000 (config minimum; real tokenizer is 128,256 → `max(config['vocab_size'], len(tokenizer))` in `train.py`) |
| EOS id | 128,009 (`<\|eot_id\|>`) |
| PAD id | 128,002 (falls back to EOS in `data/shared_data/loader.py:build_tokenizer`) |

## Loader contract (vendored, in-tree)

`data/shared_data/loader.py` exposes exactly:

| Symbol | Role |
|---|---|
| `PackedDataset` | read-only uint32 buffer sliced into `seq_len+1` windows, shift-by-1 input/target pairs, no copy |
| `ShuffledRangeSampler` | deterministic seed+offset permutation; `set_epoch` for resumable reshuffles |
| `collate_fn` | stacks chunk dicts into `[B, S]` tensors |
| `build_synthetic_data` | random-id corpus for smoke tests / first runs (byte-stub tokenizer, no HF download) |
| `build_tokenizer` | real LLaMA-3 tokenizer via `transformers.AutoTokenizer` (pad→eos fallback) |
| `build_training_data` | mmaps `data_cache/tokens.bin`, splits train/val on chunk boundaries, returns loaders + tokenizer |

## Updating the vendored loader

The workspace pipeline may evolve. To refresh the vendored loader copy:

```bash
rsync -a LLM/shared_data/loader.py LLM/LLaMA-3-Lite/data/shared_data/loader.py
```

(The workspace package itself is the canonical implementation; do not
vendor the full pipeline into this repo unless the workspace dependency is
being removed deliberately.)

## References

- Workspace canonical pipeline: `LLM/shared_data/README.md`
- Mixture spec: `LLM/shared_data/config/mixture.yaml`
- Data config: `LLM/shared_data/config/data_config.yaml`
- Per-module deep-dives: `LLM/shared_data/documentation/`
- This project's docs: `docs/reference/data.md` (code tour),
  `docs/theory/data-engineering.md` (theory)
