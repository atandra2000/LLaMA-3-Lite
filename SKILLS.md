# SKILLS.md — LLaMA-3-Lite

> Skills for the memory-optimized LLaMA-3-style model. The headline anchor
> is **78% peak memory reduction (92 GB → 20 GB)**.

---

## Skill 1: Tune chunked cross-entropy

`chunked_cross_entropy(logits, labels, chunk_size=256)` lives in `model.py`.
The trick: compute `log_softmax(logits)` and `nll_loss` in FP32 over one
chunk at a time, never materializing the full `[B, T, V]` tensor.

| Chunk | Peak logits mem | Throughput |
|-------|-----------------|------------|
| 64    | ~0.07 GB        | baseline   |
| 256   | ~0.3 GB         | baseline   |
| 1024  | ~1.2 GB         | +5%        |
| 4096  | ~5 GB           | +10%       |
| full  | ~50 GB          | OOM        |

**Default 256 is the sweet spot** for the 1× A100 80GB target. Increase
only if you also enable gradient checkpointing on the chunk compute.

## Skill 2: Switch to/from disk-backed token cache

In `config.py`:
```python
"use_disk_cache": True,     # default — 112 GB → ~1 MB RAM
"disk_cache_path": "data/cache/tokens.bin",
```

The cache is mmap-backed uint32. Build it once:
```bash
python dataset.py --build_cache
```

Then load it lazily on each `__getitem__`. The cache must be **regenerated**
whenever you change the source mixture or tokenizer.

## Skill 3: Add a new data source

In `config.py:data_sources`, add:
```python
"the_stack_rust": {"weight": 0.05, "source": "bigcode/the-stack",
                   "split": "train", "languages": ["Rust"]},
```

Then **re-build the cache** and **re-validate dedup**. The SHA-256 dedup
runs on raw text, so existing tokens get re-hashed.

## Skill 4: Tune RoPE for long-context extension

Default `rope_theta = 500_000.0` (LLaMA-3 base).

For 8K → 32K extension:
```python
# Architecture: NTK-aware scaling
"rope_theta": 1_000_000.0,   # or apply YaRN-style factor
"rope_factor": 1.0,          # >1.0 enables NTK scaling
```

For >128K: combine with YaRN attention scaling (`attention_temperature`).
Refer to the `docs/rope.md` deep-dive.

## Skill 5: Resume training from a checkpoint

```bash
python train.py --resume checkpoints/llama3_step_30000.pt
```

The resume path restores:
- model weights
- optimizer state (AdamW moments)
- LR scheduler state
- RNG state (Python + NumPy + PyTorch)
- W&B run ID

This is **full reproducibility** — re-running gives identical loss curves.

## Skill 6: Profile memory before scaling

```python
torch.cuda.reset_peak_memory_stats()
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss = model(batch)
loss.backward()
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"peak={peak_gb:.1f} GB")
```

Expected at batch 96 + grad-ckpt + chunked CE: **~20 GB peak**.

## Skill 7: Verify FA2 is active

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = attn(q, k, v)
```

If this raises on a non-FA2 device, fall back:
```python
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    out = attn(q, k, v)
```

## Skill 8: Tune stability knobs (z-loss, QK-norm, EMA)

Three optional stability features were added in the 2026-07-15 refactor. All
default to safe values; all can be disabled independently.

| Flag | Default | Disable by | Notes |
|---|---|---|---|
| `use_z_loss` | True | False | Z-loss on output logits; prevents late-run collapse. |
| `z_loss_weight` | 1e-4 | 0.0 | Gemma2 default. Higher = stronger bound, slightly slower convergence. |
| `qknorm` | True | False | QK-norm on attention; prevents attention logit growth. ~16 KB params. |
| `use_ema` | True | False | EMA for val + generation. +2 GB peak memory. |
| `ema_decay` | 0.999 | 0.0 | Standard for 42K-step runs. Use 0.9999 for >100K steps. |

**Rule of thumb:** at 515M params / 42K steps / 1× A100, all defaults are
safe. To ablate, set one to off/zero per run; expect < 0.05 perplexity
difference between adjacent settings.

**How to inspect EMA at a checkpoint:**
```python
from train import EMA
ema = EMA(model, decay=0.999)
# shadow is a dict[str, Tensor] of FP32 copies keyed by param name.
print(list(ema.shadow.keys())[:5])
```

## Skill 9: Resume with EMA

Pre-2026-07-15 checkpoints have no `ema_state_dict` key. When loaded with the
new code, EMA simply starts fresh from the live weights — no crash, no
warning. The first few validation runs after a resume will use a "young" EMA
that hasn't converged; val loss will look slightly worse than expected for
~1K steps, then stabilise.

To force a cold start, set `use_ema=False` for the first 5K steps of any
resume, then flip it on.

## Pitfalls
- **`tie_embeddings=False`** — do not enable it; the LLaMA-3 paper
  deliberately unties and so should you.
- **`GradScaler` is not used** — BF16 has FP32's exponent range, so the
  scaler is pure overhead (and historically could false-skip valid updates
  on inf-numerical-noise in the unscale step). On Volta/Turing with FP16,
  wrap the loss in `torch.amp.GradScaler()` manually.
- **`channels_last`** is for *vision* — LLMs are 2D matmul-bound, layout
  doesn't help. Skip it.
- **EOS token** must exist in the LLaMA-3 tokenizer vocab (it does, id
  `128009`). Document packing requires it as a separator.
- **`use_chunked_cross_entropy`** must be `True` or you OOM.
- **Z-loss + QK-norm interact**: both are present, both are off by default
  via `z_loss_weight=0` / `qknorm=False`. Running with both off recovers
  the original recipe exactly (within FP32↔BF16 round-trip noise).

