# LLaMA-3-Lite Documentation Index

The documentation is organized in three tracks plus this index. Every
citation to code uses symbol anchors (`<module>.py:<symbol>`), verified by
`tests/test_doc_refs.py` — stale references fail CI. See
[`docs_expansion_plan.md`](docs_expansion_plan.md) for the expansion
blueprint and [`CODE_MAP.md`](CODE_MAP.md) for the symbol ↔ doc ↔ test
map.

## Corpus size

Measured 2026-08-04 (`wc -w` over the working tree).

| Track | Files | Words |
|-------|------:|------:|
| `docs/theory/` | 14 | 72,536 |
| `docs/reference/` | 9 | 34,102 |
| `docs/guides/` | 4 | 8,663 |
| Meta (`docs/README.md`, `CODE_MAP.md`, `docs_expansion_plan.md`) | 3 | 4,732 |
| **`docs/` total** | **30** | **120,033** |
| Top-level + `data/DATA_PIPELINE.md` (`README.md`, `AGENTS.md`, `SKILLS.md`) | 4 | 5,322 |
| **Repo-wide total** | **34** | **125,355** |

## Learning paths

- **Beginner** (what is this model, how does it work):
  [guides/quickstart.md](guides/quickstart.md) →
  [theory/transformers-from-scratch.md](theory/transformers-from-scratch.md)
  → [theory/attention.md](theory/attention.md) →
  [theory/normalization.md](theory/normalization.md) →
  [theory/feedforward.md](theory/feedforward.md) →
  [theory/loss-functions.md](theory/loss-functions.md) →
  [reference/model.md](reference/model.md)
- **Intermediate** (train it, understand the numerics):
  [theory/positional-encoding.md](theory/positional-encoding.md) →
  [theory/optimization.md](theory/optimization.md) →
  [theory/mixed-precision.md](theory/mixed-precision.md) →
  [theory/gradient-checkpointing.md](theory/gradient-checkpointing.md) →
  [theory/data-engineering.md](theory/data-engineering.md) →
  [reference/training.md](reference/training.md) →
  [reference/config.md](reference/config.md)
- **Expert** (memory engineering, kernels, tests):
  [theory/memory-engineering.md](theory/memory-engineering.md) →
  [theory/kernel-programming.md](theory/kernel-programming.md) →
  [reference/kernels.md](reference/kernels.md) →
  [reference/tests.md](reference/tests.md) →
  [guides/troubleshooting.md](guides/troubleshooting.md)

## Theory track (`docs/theory/`) — from-scratch concept building

| Doc | Audience | Core topics |
|-----|----------|-------------|
| [transformers-from-scratch.md](theory/transformers-from-scratch.md) | beginner | LM task, decoder-only, residual stream, pre-norm, data flow, 513.8M anatomy |
| [attention.md](theory/attention.md) | beginner | scaled dot-product, √d_k, causal mask, MHA, GQA, Flash Attention 2 |
| [positional-encoding.md](theory/positional-encoding.md) | beginner | why positions, absolute/relative/RoPE families, θ=500K, NTK/YaRN |
| [normalization.md](theory/normalization.md) | intermediate | LayerNorm vs RMSNorm, pre-norm placement, QK-norm |
| [feedforward.md](theory/feedforward.md) | intermediate | FFN math, SwiGLU, fused gate+up |
| [loss-functions.md](theory/loss-functions.md) | intermediate | CE, chunked CE + equivalence proof, z-loss + gradient |
| [optimization.md](theory/optimization.md) | intermediate | AdamW math, decay partitioning, cosine + warmup |
| [mixed-precision.md](theory/mixed-precision.md) | intermediate | FP32/BF16/TF32, why no GradScaler, autocast scoping |
| [gradient-checkpointing.md](theory/gradient-checkpointing.md) | intermediate | activation memory math, recompute tradeoff |
| [memory-engineering.md](theory/memory-engineering.md) | expert | full 92→20 GB derivation, allocator, memmap |
| [kernel-programming.md](theory/kernel-programming.md) | expert | Triton model, online softmax, atomic_add, autograd.Function |
| [data-engineering.md](theory/data-engineering.md) | intermediate | mixture, packing, dedup, shuffling, memmap layout |
| [reproducibility.md](theory/reproducibility.md) | intermediate | RNG state, checkpoint round-trip, deterministic shuffle |
| [scaling-and-metrics.md](theory/scaling-and-metrics.md) | intermediate | token math, Chinchilla, loss curves, W&B metrics |

## Reference track (`docs/reference/`) — code-keyed walkthroughs

| Doc | Walks through |
|-----|---------------|
| [model.md](reference/model.md) | `model.py` — every block, tensor-shape trace, param budget |
| [training.md](reference/training.md) | `train.py` — sampling, generation, validation, checkpointing, the loop |
| [data.md](reference/data.md) | `data/shared_data/loader.py` + `data/prepare_data.py` shim |
| [tokenizer.md](reference/tokenizer.md) | `build_tokenizer`, stub, vocab/special-token contract |
| [rope.md](reference/rope.md) | `model.py:RoPE` implementation deep dive |
| [memory-stack.md](reference/memory-stack.md) | the 8-technique stack summary (full math in theory/memory-engineering.md) |
| [kernels.md](reference/kernels.md) | `kernels/*.py` — signatures, launch configs, fallbacks |
| [config.md](reference/config.md) | every key in `config.py:get_config` |
| [tests.md](reference/tests.md) | test strategy, fixtures, markers, CI |

## Guides (`docs/guides/`)

- [learning-paths.md](guides/learning-paths.md) — audience-specific reading orders
- [quickstart.md](guides/quickstart.md) — first run, synthetic fallback, real data, resume
- [troubleshooting.md](guides/troubleshooting.md) — OOM, compile, triton, data-cache, checkpoint issues
- [glossary.md](guides/glossary.md) — notation and acronyms

## Top-level docs (not in this folder)

- [`../README.md`](../README.md) — public overview: features, quick start, configuration, hardware.
- [`../AGENTS.md`](../AGENTS.md) — project subagent, hard rules, memory-stack table.
- [`../SKILLS.md`](../SKILLS.md) — project-scoped operational skills.
- [`../data/DATA_PIPELINE.md`](../data/DATA_PIPELINE.md) — the prepare_data shim and vendored loader reality.
