# Learning Paths — How to Read the LLaMA-3-Lite Docs

> Audience: all levels. This guide is the navigational entry point for the
> documentation tree. It does not teach any model topic itself; it tells you
> which doc to read next depending on what you already know and what you want
> out of the codebase.

The docs are organized into three tracks: **concepts** (`docs/concepts/`,
concept building from first principles), **references** (`docs/references/`,
code-keyed walkthroughs of specific files), and **guides** (`docs/guides/`,
navigation and operations). This file and the
[index](../README.md) form the meta layer on top of them.

There are three reading paths. Each assumes the previous one, and each is a
table of steps in order: start at the top, read the docs in sequence, and you
end with a working mental model of the whole stack. Every concepts doc is
self-contained (it builds its own prerequisites), so skipping a step is
usually survivable — but the "what you will know" column describes what the
step contributes, and later steps assume it.

---

## Beginner path — What this model is and how it works

Who this is for: you have basic Python and PyTorch tensor fluency, know what
next-token prediction is, and want to understand the LLaMA-3-Lite architecture
from the ground up. No transformer background required.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [quickstart.md](quickstart.md) | How to get a training loop running on synthetic data, how to build the real 8B-token corpus, and how to resume from a checkpoint. |
| 2 | [transformers-from-scratch.md](../concepts/attention-and-positional.md) | The language-modeling task, why LLaMA is decoder-only, the residual-stream view of a block, pre-norm vs post-norm, and the full data flow at this project's shapes (`[96, 2048, 1024]`). The 513.8M-parameter anatomy: 251.7M non-embedding parameters and what the 128K-wide LM head costs. |
| 3 | [attention.md](../concepts/attention-and-positional.md) | Scaled dot-product attention from first principles: why `softmax(QK^T / sqrt(d_k))`, why the causal mask, multi-head attention, grouped-query attention (`n_rep=2`), and how Flash Attention 2 makes it O(S) memory. |
| 4 | [normalization.md](../concepts/architecture-components.md) | Why deep nets need normalization, LayerNorm vs RMSNorm (dropping the mean), pre-norm residual placement, and QK-norm. |
| 5 | [feedforward.md](../concepts/architecture-components.md) | The FFN block, why the hidden layer is 4× `d_model`, SwiGLU gating, and the fused `gate_up_proj`. |
| 6 | [loss-functions.md](../concepts/architecture-components.md) | Cross-entropy for language modeling, `ignore_index=-100` semantics, why the logits are chunked (50 GB does not fit), the proof that chunked CE equals dense CE, and z-loss. |
| 7 | [model.md](../references/model-reference.md) | The code tour: every block in `model.py` with tensor-shape traces and the parameter budget, tying the six theory docs to the actual implementation. |

---

## Intermediate path — Train it and understand the numerics

Who this is for: you have completed the beginner path (or equivalent) and now
want to understand how this model is actually trained: the optimizer, the
number formats, the memory tricks, the data pipeline, and the knobs in the
config.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [positional-encoding.md](../concepts/attention-and-positional.md) | Why attention is permutation-invariant without positions, the absolute/relative/rotary families, the RoPE math, what the θ=500K frequency schedule controls, and the NTK/YaRN long-context extensions. |
| 2 | [optimization.md](../concepts/training-and-memory.md) | AdamW math, why weight decay applies only to 2D+ parameters, why β₂=0.95, FP32 master moments, gradient clipping at 1.0, and the warmup-then-cosine schedule from 3e-4 to 3e-5. |
| 3 | [mixed-precision.md](../concepts/training-and-memory.md) | FP32/FP16/BF16/TF32 bit layouts and ranges, why BF16 training needs no GradScaler, `torch.autocast` scoping rules, and `torch.set_float32_matmul_precision('high')`. |
| 4 | [gradient-checkpointing.md](../concepts/training-and-memory.md) | The `O(L·B·S·d)` activation memory math, the recompute-vs-memory tradeoff, `use_reentrant=False`, and how checkpointing interacts with the chunked head loss and CUDA graphs. |
| 5 | [data-engineering.md](../concepts/data-and-kernels.md) | The real data path (vendored loader plus the workspace shim), the 8B-token mixture, document packing with EOS separators, dedup, `ShuffledRangeSampler` determinism and epoch wrap, and the uint32 memmap layout. |
| 6 | [training.md](../training.md) | The `train.py` tour: the training loop, validation every `val_interval` steps, generation sampling (top-k/top-p, temperature), checkpointing, and the EMA shadow model. |
| 7 | [config.md](../references/model-reference.md) | Every key in `config.py:get_config`, grouped by concern, with defaults, consumers, and interactions — the map of every knob you met in steps 1–6. |
| 8 | [reproducibility.md](../concepts/training-and-memory.md) | RNG state theory, what the checkpoint round-trip stores and why it restores bit-identical runs, and how seed + offset keep the shuffle deterministic across resumes. |
| 9 | [scaling-and-metrics.md](../concepts/training-and-memory.md) | The run-sizing math (42,000 steps × 96 × 2048 = 8.26B tokens), the Chinchilla context, expected loss/perplexity trajectories, and what every W&B-logged metric means. |

---

## Expert path — Memory engineering, kernels, and tests

Who this is for: you have completed the intermediate path and want to
understand the systems layer: where every byte goes, how the fused Triton
kernels work, what the test suite defends, and how to debug the hard failures.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [memory-engineering.md](../concepts/training-and-memory.md) | The full 92 GB → 20 GB derivation, per component: model state, activations under gradient checkpointing, chunked logits, FA2 KV, memmap residency, and the CUDA caching allocator. |
| 2 | [kernel-programming.md](../concepts/data-and-kernels.md) | The Triton model of computation (grid, `tl.arange`, masks, `tl.constexpr`), and the three kernel patterns: RMSNorm row-reduce, SwiGLU elementwise fuse, and chunked CE with online softmax and `atomic_add`. |
| 3 | [kernels.md](../references/data-reference.md) | The code tour of `kernels/*.py`: per-kernel signatures, launch configs, the `autograd.Function` wrappers, fallback semantics, and the opt-in `ENABLE_TRITON_KERNELS=1` gating. |
| 4 | [tests.md](../references/training-reference.md) | The test strategy (unit/equivalence/smoke/e2e), the fixture reference (`tiny_config`, `tiny_model`, `make_token_stream`), markers (`gpu`, `numeric`, `smoke`), and how to run each suite. |
| 5 | [troubleshooting.md](troubleshooting.md) | The FAQ: OOM at batch 96, CUDA-graph capture stalls, Triton import failures on CPU/Mac, data-cache and `len(tokenizer)` errors, checkpoint corruption, and epoch-wrap `StopIteration`. |

---

## Prerequisite map

| Path | Assumes | Entry point |
|------|---------|-------------|
| Beginner | Python, basic PyTorch tensors, the idea of next-token prediction | [quickstart.md](quickstart.md) |
| Intermediate | The full beginner path: residual-stream model view, GQA, RMSNorm, SwiGLU, chunked CE, and the `model.py` tour | [positional-encoding.md](../concepts/attention-and-positional.md) |
| Expert | The full intermediate path: optimizer, numerics, data pipeline, config surface | [memory-engineering.md](../concepts/training-and-memory.md) |

No path has a hard time estimate; each step's doc states its own audience in
its first lines, and every theory doc is written to be self-contained, so a
step you already know can be skipped safely. Within a path the order matters:
later steps reference the concepts (not just the file names) of earlier ones.

### Off-path docs

These docs serve specific needs and plug into the paths as needed rather than
forming their own sequence:

- [glossary.md](glossary.md) — the shared vocabulary of the whole tree; read it first if any acronym or symbol stops you.
- [data.md](../references/data-reference.md) and [tokenizer.md](../references/data-reference.md) — the code tours behind the data-engineering step; read alongside intermediate step 5.
- [rope.md](../references/model-reference.md) — the RoPE implementation deep dive; read alongside intermediate step 1.
- [memory-stack.md](../training.md) — the eight-technique summary table; read alongside expert step 1.

---

## Cheat sheet — Which doc answers which question

| Question | Doc |
|----------|-----|
| How do I run training for the first time, or with real data? | [quickstart.md](quickstart.md) |
| How does a transformer actually work, end to end? | [transformers-from-scratch.md](../concepts/attention-and-positional.md) |
| Why is the LM head chunked, and is chunked CE identical to dense CE? | [loss-functions.md](../concepts/architecture-components.md) |
| Why does attention need positions at all, and what is RoPE? | [positional-encoding.md](../concepts/attention-and-positional.md) |
| What does every config key mean and which keys matter? | [config.md](../references/model-reference.md) |
| Why is the run sized at 42,000 steps / 8.26B tokens, and what do the W&B curves mean? | [scaling-and-metrics.md](../concepts/training-and-memory.md) |
| Why is there no GradScaler, and what runs in BF16 vs FP32? | [mixed-precision.md](../concepts/training-and-memory.md) |
| Where does the memory actually go, and how do we get from 92 GB to 20 GB? | [memory-engineering.md](../concepts/training-and-memory.md) |
| What do the Triton kernels do, and when are they used? | [kernel-programming.md](../concepts/data-and-kernels.md) then [kernels.md](../references/data-reference.md) |
| How do I resume a run bit-identically? | [reproducibility.md](../concepts/training-and-memory.md) |
| What does the test suite cover and how do I run it? | [tests.md](../references/training-reference.md) |
| Training OOMs, Triton won't import, the cache is missing, or a checkpoint is corrupt? | [troubleshooting.md](troubleshooting.md) |
| What does `B`, `S`, `d`, `n_kv`, or any other acronym mean? | [glossary.md](glossary.md) |
| Where is a given code symbol documented? | [docs/README.md](../README.md) (file→doc map) |
