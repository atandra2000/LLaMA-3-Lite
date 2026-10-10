# How to read this book

This portal is a book about building a LLaMA-3-style decoder from scratch in raw
PyTorch. It has eleven chapters in four parts. Read it
in order, or jump to the part you need.

Every citation names a symbol (`model.py:RoPE`), never a line number, so the
references survive edits. `tests/test_doc_refs.py` resolves each one and fails CI
when a citation goes stale.

## The parts

| Part | What it covers |
|------|----------------|
| **Part I: Foundations** | What the model is, and the three components that do the arithmetic: normalization, the feed-forward block, and the loss. |
| **Part II: The data** | How 8.25B tokens become a memory-mapped array of ids, and the kernels that keep one step fast. |
| **Part III: The training run** | The optimizer, the precision policy, the memory stack, and the loop that ties them together. |
| **Part IV: Reference** | Code-keyed walkthroughs. Open these when you need a signature or a config key, not a narrative. |

## Where to start

- **New to the model.** Read Part I in order.
  [Attention and positional encoding](concepts/attention-and-positional.md) is
  the longest chapter, and it assumes no prior reading.
- **You want to run it.** Start with the [Overview](../README.md) for hardware
  and configuration, then read
  [The training loop, end to end](training.md).
- **You want the numerics.** Read
  [Training, memory, and numerical stability](concepts/training-and-memory.md),
  which derives the 92 GB to 20 GB memory stack term by term.

## Visual tour

The [Illustrated systems guide](diagrams/atlas/index.html) is five standalone
interactive maps: system ownership, the decoder, data preparation and
consumption, the training loop, and all eight optimization techniques. Each map
carries tensor shapes, memory arithmetic, and source links. The
[artifact and browser evidence](diagrams/atlas/RECEIPTS.md) records what was
verified for each map.

## Which source file is documented where

| Module | Chapter | Key symbols |
|--------|---------|-------------|
| `model.py` | [Model, RoPE, and config](references/model-reference.md) (full walkthrough), [Attention and positional encoding](concepts/attention-and-positional.md) (attention and RoPE theory), [Architecture components](concepts/architecture-components.md) (norms, FFN, loss) | `model.py:RoPE`, `model.py:chunked_head_cross_entropy_with_z` |
| `config.py` | [Model, RoPE, and config](references/model-reference.md) (every key) | `config.py:get_config` |
| `train.py` | [The training loop, end to end](training.md) (full walkthrough), [Training, memory, and numerical stability](concepts/training-and-memory.md) (optimizer and memory stack) | `train.py:train_model`, `train.py:load_checkpoint` |
| `data/prepare_data.py`, `data/shared_data/loader.py` | [Data, tokenizer, and kernels](references/data-reference.md) (code tour), [The training loop, end to end](training.md) (data path), [The data pipeline](concepts/data-and-kernels.md) (theory) | `data/prepare_data.py:main`, `data/shared_data/loader.py:build_tokenizer` |
| `LLM/shared_data` (workspace pipeline) | [The shared data pipeline](references/workspace-data.md) (stages, manifest, shards), [Data, tokenizer, and kernels](references/data-reference.md) (the shim and the bridge stage) | `data/prepare_data.py:concat_shards_to_cache` |
| `dataset.py` | [Data, tokenizer, and kernels](references/data-reference.md) (re-export shim) | — |
| `kernels/*.py` | [Data, tokenizer, and kernels](references/data-reference.md) (kernel reference), [The data pipeline](concepts/data-and-kernels.md) (kernel programming theory) | `kernels/rmsnorm_triton.py:triton_rmsnorm` |
| `tests/*` | [The test suite](references/training-reference.md) (strategy, fixtures, markers) | `tests/conftest.py` |
| `benchmark_data.py` | [Data, tokenizer, and kernels](references/data-reference.md) (the benchmark harness) | — |

The repository README is Chapter 1 of this book, not a separate page.
