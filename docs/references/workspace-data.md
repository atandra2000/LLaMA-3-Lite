# Workspace data pipeline (`LLM/shared_data`) — reference

> This page documents the **universal corpus pipeline** that produces the
> training data LLaMA-3-Lite consumes. It lives outside this repo — at
> `LLM/shared_data/` (the workspace root's sibling of `LLM/LLaMA-3-Lite`)
> — and is shared by all five LLM projects. The authoritative in-workspace
> documentation is `LLM/shared_data/README.md`; this page is the
> project-side view: what the pipeline produces, where it puts it, and how
> it connects to this repo's vendored loader.

## Why a shared pipeline

Every LLM project used to prepare its own corpus — its own downloads, its
own shard format, its own manifest schema. That wasted roughly 27 hours of
duplicate download/clean work per project and produced incompatible shards.
The workspace pipeline fixes this: **one** mixture, one dedup, one shard
format, one manifest schema, consumed by all five projects. Only the
tokenizer is per-project (each project has its own vocab and EOS id), and
only LLaMA-3-Lite and GPT-OSS-Lite share the LLaMA-3 BPE — their shards
are bit-identical.

## Data-root resolution

The pipeline resolves its data root once, at import time
(`LLM/shared_data/common.py`):

```
LLM_DATA_ROOT (env)  >  $PWD/data
```

Run `python data/prepare_data.py` from this repo's root and the trees land
under `LLaMA-3-Lite/data/`; set `LLM_DATA_ROOT` to a shared directory and
every project reads the same shards. `data/prepare_data.py --data-root
PATH` overrides for a single invocation (it calls
`shared_data.common.set_data_root` before running).

## Directory layout

Under the data root:

| Path | Contents |
|---|---|
| `raw/<src>/data.jsonl` | Streaming downloads from HuggingFace (JSONL) |
| `clean/<src>/data.jsonl` | Quality-filtered + SHA-256-deduped documents |
| `tokens/<src>/data.bin` | Per-source `uint32` token streams, EOS-separated (one `TokenStream` per source) |
| `shards/shard_*.bin` | Packed, training-ready shards (50 M tokens each) |
| `shards/manifest.json` | Full provenance (schema below) |
| `state/<stage>_<source>.json` | Resumable per-stage state |
| `config/` | `mixture.yaml` + `data_config.yaml` (copied for provenance) |

## The five stages

Each stage is a subprocess (`python -m shared_data.scripts.<stage>`), so a
crash in one stage never loses the others' work. All stages are idempotent
and resumable via `state/`.

| # | Stage | Input → output | Notes |
|---|---|---|---|
| 0 | `train_tokenizer` (optional) | trains a custom BPE under `data/tokenizer/` | HyMo only; LLaMA-3-Lite uses the stock LLaMA-3 tokenizer |
| 1 | `download_raw` | HF sources → `raw/<src>/data.jsonl` | streaming, per-source; ~1.3 TB at full scale |
| 2 | `clean` | `raw/` → `clean/<src>/data.jsonl` | 6 cheap text filters + SHA-256 exact dedup (256 hash buckets + Bloom filter) |
| 3 | `tokenize` | `clean/` → `tokens/<src>/data.bin` | per-source `uint32` stream, EOS appended after every document |
| 4 | `pack_shards` | `tokens/` → `shards/shard_*.bin` + `shards/manifest.json` | round-robin interleave across sources; atomic writes; SHA-256-verified; manifest validated before save |

The shim's bridge stage (`data/prepare_data.py:concat_shards_to_cache`)
then concatenates the shards into the flat `tokens.bin` the vendored
loader mmaps — see [data-reference.md](data-reference.md).

## The canonical mixture

`LLM/shared_data/config/mixture.yaml` — 8.0 B tokens, 7 sources, weights
sum to 1.0 (validated at load time):

| id | dataset (config) | weight | tokens (×8.0e9) | role |
|---|---|---|---|---|
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` (sample-10BT) | 0.40 | 3.20 B | quality-gated web backbone |
| `dclm-baseline` | `mlfoundations/dclm-baseline-1.0` | 0.15 | 1.20 B | DCLM-curated web |
| `the-stack-v2-python` | `bigcode/the-stack-v2` (Python) | 0.15 | 1.20 B | code, reasoning lever |
| `the-stack-v2-jupyter` | `bigcode/the-stack-v2` (JupyterNotebook) | 0.05 | 0.40 B | notebook code+prose |
| `openmath` | `nvidia/OpenMathInstruct-2` | 0.10 | 0.80 B | problem + worked solution |
| `arxiv` | `cdv/arxiv-classification` | 0.10 | 0.80 B | long documents, long context |
| `cosmopedia` | `HuggingFaceTB/cosmopedia` | 0.05 | 0.40 B | synthetic educational prose |

Web is 55%, code + math ≈ 30%, and the mix is tuned so no shard is
dominated by one source: `pack_shards` round-robin interleaves documents
one at a time across sources. OpenMath documents concatenate the problem
and its generated solution with a `"\n\n### Solution\n\n"` separator.

## The shard format

`shards/shard_NNNNN.bin` is a flat contiguous buffer of token ids:

- **dtype:** `uint32` (4 bytes/token; safe to 4.29 B vocab — covers the
  8 B-token corpus as 32 GB on disk)
- **size:** 50,000,000 tokens ≈ 190 MB per shard
- **EOS:** every document boundary is marked with the configured EOS id;
  documents are **never split across shards**
  (`cross_document_boundary_ok: false` in `data_config.yaml`)
- **atomicity:** written to `.tmp`, `os.fsync`, `os.replace` on success
- **verification:** re-read after writing; the SHA-256 goes into the
  manifest

A training window can safely cross an EOS — it is just a regular token —
without leaking semantic context between unrelated documents, because the
shards were EOS-separated at pack time.

## The manifest schema

`shards/manifest.json` (v1.0.0) records full provenance and is validated
before save:

| Field | Meaning |
|---|---|
| `vocab_size` / `eos_token_id` / `pad_token_id` / `tokenizer_name` | tokenizer contract that produced the ids |
| `dtype` / `shard_size_tokens` / `total_tokens` / `shard_count` | physical layout |
| `shards_dir` | relative to the data root (usually `data/shards`) |
| `shards[]` | per shard: `index`, `path` (relative to `shards_dir`), `n_tokens`, `sha256`, `n_eos` |
| `sources{}` | per source: `target_tokens`, `actual_tokens`, `n_docs`, `n_dedup_dropped`, `shard_count` |
| `config_hash` / `mixture_hash` | SHA-256 of the two YAMLs, for reproducibility |
| `created_utc` | pipeline run timestamp |

`Manifest.validate()` enforces shard count, total tokens, EOS coverage,
and per-source mix; the pack stage aborts on any issue.

## How LLaMA-3-Lite consumes it

The chain, end to end:

```
mixture.yaml + data_config.yaml
   │  python data/prepare_data.py --stage pretrain
   ▼
LLM/shared_data pipeline (5 stages, subprocesses)
   │
   ▼
data/shards/shard_*.bin + data/shards/manifest.json   (32 GB uint32)
   │  data/prepare_data.py:concat_shards_to_cache
   ▼
data_cache/tokens.bin  (flat uint32, manifest order)
   │  data/shared_data/loader.py:build_training_data
   ▼
np.memmap → PackedDataset windows → train/val DataLoaders
```

Key facts for this repo:

- The pipeline itself never emits `tokens.bin` — the project shim's
  bridge stage does. `--skip-pack` leaves nothing to concatenate.
- The vendored `data/shared_data/` package is a **loader only** (2 files);
  it cannot run any pipeline stage and never imports the workspace
  package.
- `data/prepare_data.py` is the only file that crosses into the
  workspace. Without `LLM/shared_data` importable it exits with a
  `SystemExit` guidance message rather than silently preparing a
  different corpus.
- LLaMA-3-Lite's tokenizer contract is pinned by the shim:
  `LLAMA3_TOKENIZER_NAME = "llama3"`, `LLAMA3_VOCAB_SIZE = 128_000`,
  `LLAMA3_EOS_TOKEN_ID = 128_009`, `LLAMA3_PAD_TOKEN_ID = 128_002`.
