# LLaMA-3-Lite Documentation Index

This folder extends the project's top-level reference material with
extracted rationale that previously lived as inline code comments.
For the deep first-principles architectural walkthrough, see
[`../architecture.md`](../architecture.md) (1,234 lines) — the files
below reference it rather than duplicating it.

## Reference docs

| File | Scope |
|------|-------|
| [`model_architecture.md`](model_architecture.md) | Block-by-block explanation of `model.py` (InputEmbedding, RoPE, RMSNorm, GQA, SwiGLU, DecoderBlock, Transformer, `chunked_cross_entropy`, `build_transformer`). References `../architecture.md` for theory. |
| [`training.md`](training.md) | Line-by-line walkthrough of `train.py` — LR scheduler, sampling, generation, validation, checkpointing, async I/O, the training loop, design decisions. |
| [`data_prep.md`](data_prep.md) | The streaming → tokenize → dedup → pack → memmap pipeline in `dataset.py`. Sources, filters, cache layout, split alignment, `PackedDataset`, samplers, dataloader construction. |
| [`rope.md`](rope.md) | Deep dive on the `RoPE` class — math, frequency schedule, why θ=500K, interaction with GQA and Flash-Attention-2. |
| [`tokenizer.md`](tokenizer.md) | LLaMA-3 BPE tokenizer (128K vocab), special tokens, encoding/decoding, streaming tokenization, uint32 storage layout. |
| [`memory_stack.md`](memory_stack.md) | The 7-technique memory stack that yields the headline **78% peak memory reduction (92 GB → 20 GB)**. |

## Authoritative top-level docs (not in this folder)

- [`../architecture.md`](../architecture.md) — 1,234-line first-principles
  architecture reference. The canonical source for theory; the
  `docs/model_architecture.md` file is keyed to its line numbers.
- [`../AGENTS.md`](../AGENTS.md) — project subagent definition, hard rules,
  and the memory-stack table (mirrored in `memory_stack.md`).
- [`../SKILLS.md`](../SKILLS.md) — project-scoped skill index.
- [`../README.md`](../README.md) — project README (features, quick start,
  configuration, hardware requirements).