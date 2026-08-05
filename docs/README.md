# LLaMA-3-Lite — Documentation Index

The documentation is organized in four parts plus this index: **concepts** (`docs/concepts/`, theory and architecture built from first principles), **references** (`docs/references/`, code-keyed walkthroughs), **guides** (`docs/guides/`, how-to and operations), and a top-level
[training.md](training.md) covering the applied training pipeline, memory
stack, and data pipeline. Every citation to code uses symbol anchors (`<module>.py:<symbol>`), verified by `tests/test_doc_refs.py` — stale references fail CI. This page is the nav map: it also folds in the file→doc map that used to live in `CODE_MAP.md`.

## Corpus size

Measured 2026-08-05 (`wc -w` over the working tree; refreshed after the audit follow-up: new `workspace-data.md` reference, the benchmark reference section, and the EMA-internals canonical section).

| Track | Files | Words |
|-------|------:|------:|
| `docs/concepts/` | 4 | 70,397 |
| `docs/references/` | 4 | 28,880 |
| `docs/guides/` | 4 | 8,767 |
| `docs/training.md` | 1 | 7,850 |
| `docs/README.md` | 1 | 658 |
| **`docs/` total** | **14** | **116,552** |
| Top-level (`README.md`, `AGENTS.md`, `SKILLS.md`) | 3 | 5,259 |
| **Repo-wide total** | **17** | **121,811** |

## Learning paths

- **Beginner** (what is this model, how does it work):
  [guides/quickstart.md](guides/quickstart.md) →
  [concepts/attention-and-positional.md](concepts/attention-and-positional.md)
  (the decoder, attention, RoPE) →
  [concepts/architecture-components.md](concepts/architecture-components.md)
  (RMSNorm, SwiGLU, chunked CE, z-loss) →
  [references/model-reference.md](references/model-reference.md)
- **Intermediate** (train it, understand the numerics):
  [concepts/training-and-memory.md](concepts/training-and-memory.md) →
  [training.md](training.md) →
  [references/model-reference.md](references/model-reference.md) (config
  section) →
  [concepts/data-and-kernels.md](concepts/data-and-kernels.md)
- **Expert** (memory engineering, kernels, tests):
  [training.md](training.md) (memory-stack section) →
  [concepts/data-and-kernels.md](concepts/data-and-kernels.md) →
  [references/data-reference.md](references/data-reference.md) →
  [references/training-reference.md](references/training-reference.md) →
  [guides/troubleshooting.md](guides/troubleshooting.md)

## Concepts track (`docs/concepts/`) — from-scratch concept building

| Doc | Audience | Core topics |
|-----|----------|-------------|
| [attention-and-positional.md](concepts/attention-and-positional.md) | beginner | LM task, decoder-only design, residual stream, pre-norm, data flow, 513.8M anatomy; scaled dot-product, √d_k, causal mask, MHA, GQA, Flash Attention 2; why positions, absolute/relative/RoPE families, θ=500K, NTK/YaRN |
| [architecture-components.md](concepts/architecture-components.md) | intermediate | RMSNorm vs LayerNorm, pre-norm placement, QK-norm; FFN math, SwiGLU, fused gate+up; CE, chunked CE + equivalence proof, z-loss + gradient |
| [training-and-memory.md](concepts/training-and-memory.md) | intermediate | AdamW math, decay partitioning, cosine + warmup; FP32/BF16/TF32, why no GradScaler; full 92→20 GB derivation, allocator, memmap; activation-memory math, recompute tradeoff; RNG state, checkpoint round-trip; token math, Chinchilla, W&B metrics |
| [data-and-kernels.md](concepts/data-and-kernels.md) | intermediate→expert | mixture, packing, dedup, shuffling, memmap layout; Triton model, online softmax, atomic_add, autograd.Function, HAS_TRITON gating |

## References track (`docs/references/`) — code-keyed walkthroughs

| Doc | Walks through |
|-----|---------------|
| [model-reference.md](references/model-reference.md) | `model.py` — every block, tensor-shape trace, param budget; `model.py:RoPE` implementation deep dive; every key in `config.py:get_config` |
| [training-reference.md](references/training-reference.md) | test strategy, fixtures (`tests/conftest.py`), markers, per-file walkthroughs, the e2e GPU script, running the suites |
| [data-reference.md](references/data-reference.md) | `data/shared_data/loader.py` + `data/prepare_data.py` shim; `build_tokenizer`, stub, vocab/special-token contract; `kernels/*.py` — signatures, launch configs, fallbacks |
| [workspace-data.md](references/workspace-data.md) | the universal `LLM/shared_data` pipeline (stages, mixture, shard format, manifest schema, data-root rules) that produces the corpus |

## Guides (`docs/guides/`)

- [learning-paths.md](guides/learning-paths.md) — audience-specific reading orders
- [quickstart.md](guides/quickstart.md) — first run, synthetic fallback, real data, resume
- [troubleshooting.md](guides/troubleshooting.md) — OOM, compile, triton, data-cache, checkpoint issues
- [glossary.md](guides/glossary.md) — notation and acronyms

## Top-level training doc

- [training.md](training.md) — the applied training pipeline: `train.py`
  walkthrough (loop, sampling, validation, checkpointing, EMA), the eight-technique memory stack (92 GB → 20 GB derivation), and the data pipeline (vendored loader + workspace `LLM/shared_data` preparation).

## File→doc map (formerly `docs/CODE_MAP.md`)

The old `CODE_MAP.md` was replaced by this hand-maintained map (the generator script was removed in the 2026-08-04 doc cleanup). `tests/test_doc_refs.py` (CI) verifies every citation resolves.

| Module | Where documented |
|--------|------------------|
| `model.py` | [model-reference.md](references/model-reference.md) (full walkthrough), [attention-and-positional.md](concepts/attention-and-positional.md) (attention/RoPE theory), [architecture-components.md](concepts/architecture-components.md) (norms, FFN, loss) |
| `config.py` | [model-reference.md](references/model-reference.md) (config section), [glossary.md](guides/glossary.md) (config-key glossary) |
| `train.py` | [training.md](training.md) (full walkthrough), [quickstart.md](guides/quickstart.md), [troubleshooting.md](guides/troubleshooting.md) |
| `data/prepare_data.py`, `data/shared_data/loader.py` | [data-reference.md](references/data-reference.md) (code tour), [training.md](training.md) (data-pipeline section), [data-and-kernels.md](concepts/data-and-kernels.md) (theory) |
| `LLM/shared_data` (workspace pipeline) | [workspace-data.md](references/workspace-data.md) (stages, manifest, shards), [data-reference.md](references/data-reference.md) (the shim + bridge stage) |
| `dataset.py` | [data-reference.md](references/data-reference.md) (re-export shim note) |
| `kernels/*.py` | [data-reference.md](references/data-reference.md) (kernel reference), [data-and-kernels.md](concepts/data-and-kernels.md) (kernel programming theory) |
| `tests/*` | [training-reference.md](references/training-reference.md) (strategy, fixtures, markers), plus per-doc test citations |
| `benchmark_data.py` | [quickstart.md](guides/quickstart.md) |

## Top-level docs (not in this folder)

- [`../README.md`](../README.md) — public overview: features, quick start, configuration, hardware.
- [`../AGENTS.md`](../AGENTS.md) — project subagent, hard rules, memory-stack table.
- [`../SKILLS.md`](../SKILLS.md) — project-scoped operational skills.
