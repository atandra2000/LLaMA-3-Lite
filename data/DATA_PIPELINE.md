# Data Pipeline — LLaMA-3-Lite

> This project uses the **universal 8.0B-token LLM data pipeline** shared
> by all 5 LLM projects in the CoreProjects portfolio.
> The pipeline is **vendored in-tree** at `data/shared_data/` so the repo
> is fully self-contained on a fresh clone.

---

## Quick start

```bash
# Full pipeline (download → clean → tokenize → pack)
python3 data/prepare_data.py --stage pretrain

# Skip download (re-use an existing corpus)
python3 data/prepare_data.py --stage pretrain --skip-download

# Re-pack only (after a config change)
python3 data/prepare_data.py --stage pretrain \
    --skip-download --skip-clean --skip-tokenize
```

## Tokenizer used by LLaMA-3-Lite

| Field | Value |
|---|---|
| Family | LLaMA-3 BPE (Meta) |
| Vocab size | **128,000** |
| EOS id | **128,009** (`<\|eot_id\|>`) |
| PAD id | 128,002 |

> The shards produced by **LLaMA-3-Lite** and **GPT-OSS-Lite** are
> **bit-identical** (both use LLaMA-3 BPE) and can be shared verbatim
> between the two projects.

## What this shim does

`data/prepare_data.py` is a thin wrapper that:

1. Invokes `shared_data.prepare_data.run_pipeline(...)` using the
   universal LLaMA-3 defaults (no project-local data config override
   is needed).
2. Runs `download_raw → clean → tokenize → pack_shards` as separate
   subprocesses (so a crash in one stage doesn't poison the next).

## Vendored copy

`data/shared_data/` is a **verbatim copy** of the workspace-level
`LLM/shared_data/` package. The shim resolves it first (via
`sys.path.insert(0, <project>/data)`) and falls back to the workspace
copy if the vendored copy is missing.

**Vendored size:** ~160 KB · 24 source files.

## The canonical reference

The authoritative documentation for the pipeline lives at the workspace
level: `LLM/shared_data/README.md` (and the per-module deep-dives in
`LLM/shared_data/documentation/`). See that file for:

- The 5-stage pipeline diagram
- The 8.0B-token mixture specification
- The shard format (`uint32`, 50M tokens, EOS-separated)
- The manifest schema
- Performance numbers and atomicity invariants

## Data mix (8.0B tokens, Chinchilla-optimal for ~400M-param models)

| Source | Weight | Tokens |
|---|---:|---:|
| FineWeb-Edu (HuggingFaceFW) | 0.50 | 4.00 B |
| FineWeb (HuggingFaceFW) | 0.20 | 1.60 B |
| the-stack-python (bigcode) | 0.15 | 1.20 B |
| OpenMathInstruct-2 (nvidia) | 0.10 | 0.80 B |
| arxiv (cdv) | 0.05 | 0.40 B |
| **Total** | **1.00** | **8.00 B** |

## Layout

```
data/
├── prepare_data.py              ← this project's shim
├── shared_data/                 ← vendored universal pipeline
│   ├── __init__.py
│   ├── common.py                ← paths, atomic IO, hashing
│   ├── config.py                ← PIPELINE_VERSION, UNIVERSAL_TOTAL_TOKENS
│   ├── config/
│   │   ├── mixture.yaml
│   │   └── data_config.yaml
│   ├── scripts/                 ← stage entry points
│   ├── quality_filter.py
│   ├── dedup.py
│   ├── shard_writer.py
│   ├── shard_reader.py
│   ├── manifest.py
│   ├── prepare_data.py          ← orchestrator
│   └── documentation/
└── ...
```

## Why a vendored copy?

When LLaMA-3-Lite is cloned standalone, the workspace-level
`LLM/shared_data/` is **not present**. Vendoring makes the repo fully
runnable on its own. The shim's `sys.path` lookup is:

```python
# in data/prepare_data.py
sys.path.insert(0, <project_root>/data)        # vendored copy first
sys.path.insert(0, <workspace_root>/LLM)       # workspace copy as fallback
```

## Updating the vendored copy

The workspace-level pipeline may receive updates. To refresh the
vendored copy in any project, from the workspace root:

```bash
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='README.md' \
    LLM/shared_data/  LLM/LLaMA-3-Lite/data/shared_data/
```

The 5 vendored copies are kept **bit-identical**.

## References

- Workspace-level canonical docs: `LLM/shared_data/README.md`
- Workspace-level mixture spec: `LLM/shared_data/config/mixture.yaml`
- Workspace-level data config:  `LLM/shared_data/config/data_config.yaml`
- Per-module deep-dives:       `LLM/shared_data/documentation/`
