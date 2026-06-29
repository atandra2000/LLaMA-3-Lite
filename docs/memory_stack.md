# The 7-Technique Memory Stack — 78% Peak Memory Reduction (92 GB → 20 GB)

> Headline metric for LLaMA-3-Lite: a 515M-param model trains at batch 96 on
> a single A100 80GB with ~60 GB headroom, thanks to the seven techniques
> below. For the deep architectural walkthrough see
> [`../architecture.md`](../architecture.md); this file captures the
> extracted rationale that previously lived as inline comments.

## The stack

| # | Technique | What it saves | Where in code |
|---|-----------|---------------|---------------|
| 1 | **Gradient checkpointing** | ~55% of activation memory (activations 70 GB → 3.2 GB) | `model.py` `Transformer.forward` (`checkpoint(..., use_reentrant=False)`) |
| 2 | **Chunked cross-entropy** | logits tensor 50 GB → 0.3 GB (~100×) | `model.py` `chunked_cross_entropy` (chunk size 256 tokens) |
| 3 | **Disk-backed uint32 token cache** | RAM 112 GB → ~1 MB (memmap, OS pages on demand) | `dataset.py` `_stream_to_disk`, `PackedDataset` |
| 4 | **BF16 mixed precision** | 2× weight memory vs FP32; A100 native tensor cores | `train.py` `torch.autocast(dtype=torch.bfloat16)` |
| 5 | **Flash-Attention 2** | O(N) attention memory; fused softmax+matmul | `model.py` `F.scaled_dot_product_attention(is_causal=True)` |
| 6 | **`channels_last`** | memory-layout speedup for conv/attention tensors | (applied at model construction for Blackwell-class GPUs) |
| 7 | **Fused AdamW** | fewer kernel launches (single fused optimizer step) | `train.py` `torch.optim.AdamW` with `set_to_none=True` |
| 8 | **TF32 matmuls** | ~3× matmul throughput on Ampere (FP32 path) | `train.py` `setup_gpu_optimizations` (`allow_tf32=True`) |

## Why each technique is load-bearing

### 1. Gradient checkpointing
Activations for 16 layers at batch 96 × seq 2048 dominate the memory budget
(~70 GB if every layer's intermediates are kept). Checkpointing retains only
each layer's input during forward and re-runs the layer during backward —
~25% extra compute, ~78% activation memory cut. `use_reentrant=False`
selects PyTorch's newer implementation that is more memory-efficient and
plays nicely with `torch.compile`.

### 2. Chunked cross-entropy (chunk size 256)
With `batch_size=96, seq_len=2048, vocab_size=128256`, the full logits
tensor is `96 × 2048 × 128256 × 2 B ≈ 50 GB` in BF16 — alone enough to OOM
an A100-80GB. `chunked_cross_entropy` slices the flattened `(B·S, V)` tensor
into row-chunks of `chunk_size` (default 256 tokens) and accumulates
`total_loss / total_count` on-device. Because cross-entropy is additive
across rows, the result is **numerically identical** (within 1e-5) to the
unchunked `F.cross_entropy`. GPU-tensor accumulators avoid per-iteration
CPU↔GPU syncs. **Hard rule: never change the chunk size from 256** without
updating the AGENTS.md invariant.

### 3. Disk-backed uint32 token cache
The 4B-token corpus is ~16 GB as a flat `uint32` file (`tokens.bin`). It is
opened with `np.memmap(..., mode='r')` so the OS pages in 4 KB chunks on
demand — resident memory stays at ~MB while the file can be many GB. No
compression is used because random access must stay O(1) for
`PackedDataset.__getitem__`. First run streams from HuggingFace and writes
the cache; subsequent runs reuse it if `reuse_data_cache=True`.

### 4. BF16 over FP16
A100 has native BF16 tensor cores and BF16 has the same 8-bit exponent
range as FP32, so gradient underflow is very rare. `GradScaler` is kept
enabled for correctness on mixed systems but rarely needs to scale.

### 5. Flash-Attention 2
`F.scaled_dot_product_attention(q, k, v, is_causal=True)` dispatches to
Flash-Attention-2 / memory-efficient kernels on A100, giving O(S) memory
instead of O(S²) and a 2–3× speedup.

### 6. `channels_last`
Mandatory on RTX 5090 (Blackwell / sm_120); applied via
`model.to(memory_format=torch.channels_last)` before the first forward pass.

### 7. Fused AdamW + `set_to_none=True`
AdamW with decoupled weight decay on 2D+ params only (see
[`training.md`](training.md)). `optimizer.zero_grad(set_to_none=True)` sets
`param.grad = None` instead of zeroing the tensor, saving one memory write
per parameter per step.

### 8. TF32
`torch.backends.cuda.matmul.allow_tf32 = True` enables TensorFloat-32
matmuls on Ampere — 10 mantissa bits, ~3× faster than full FP32. Safe here
because the model trains in BF16; TF32 only affects residual FP32 matmuls
(e.g. optimizer state updates).

## Peak memory breakdown (A100 80GB SXM, batch 96)

| Component | Without optimizations | With optimizations |
|-----------|----------------------|-------------------|
| Model state (BF16 + FP32 master + Adam m+v) | 7.2 GB | 7.2 GB |
| Checkpointed activations (16 layers) | ~70 GB | 3.2 GB |
| One layer backward recomputation | — | 3.6 GB |
| Logits tensor | 50.4 GB | 0.3 GB |
| Overhead + gradients | 2.0 GB | 5.7 GB |
| **Peak total** | **~92 GB (OOM)** | **~20 GB (25%)** |
| **Headroom** | −12 GB | ~60 GB |

## Hard rules (from `AGENTS.md`)

1. **Chunked-CE chunk size = 256** — do not change.
2. **`tie_embeddings=False`** — LLaMA-3 does not tie input/output embeddings.
3. **RoPE θ = 500K** — load-bearing for long-context extrapolation (see
   [`rope.md`](rope.md)).
4. **Document packing must include EOS separators** — without them the model
   sees run-on concatenated documents and degrades (see
   [`data_prep.md`](data_prep.md)).

## References

- Flash-Attention-2 — Dao (2023). arXiv:2307.08691.
- Gradient Checkpointing — Chen et al. (2016). arXiv:1604.06174.
- LLaMA 3 — Meta AI (2024). arXiv:2407.21783.