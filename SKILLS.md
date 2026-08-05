---
name: llama3-lite-skills
description: Project-scoped operational skills for LLaMA-3-Lite — chunked CE tuning, disk-backed cache, data sources, RoPE, resume, memory profiling, FA2 verification, stability knobs, EMA.
metadata:
  type: skills
---

# SKILLS.md — LLaMA-3-Lite

> Skills for the memory-optimized LLaMA-3-style model. The headline anchor
> is **78% peak memory reduction (92 GB → 20 GB)**.

---

## Skill 1: Tune chunked LM-head cross-entropy

`chunked_head_cross_entropy_with_z(hidden, head_weight, targets, chunk_size=256)`
lives in `model.py`. The trick: the training path returns final hidden states
(`model(x, return_hidden=True)`) and the loss computes the output projection
in `chunk_size`-row slices, each inside `torch.utils.checkpoint` — the full
`[B, T, V]` logits tensor is never materialized, and only one chunk's logits
are alive at a time.

| Chunk | Peak logits mem | Throughput |
|-------|-----------------|------------|
| 64    | ~0.07 GB        | baseline   |
| 256   | ~0.3 GB         | baseline   |
| 1024  | ~1.2 GB         | +5%        |
| 4096  | ~5 GB           | +10%       |
| dense head | ~50 GB      | OOM        |

**Default 256 is the sweet spot** for the 1× A100 80GB target. Increase
only if you also enable gradient checkpointing on the chunk compute.

## Skill 2: Switch to/from disk-backed token cache

In `config.py`:
```python
"data_cache_dir": "data_cache",          # mmap-backed uint32 corpus
"data_cache_filename": "tokens.bin",
```

The cache is mmap-backed uint32. Build it by running the workspace pipeline:
```bash
python data/prepare_data.py
```

Note the shim produces `data/shards/shard_*.bin` + `manifest.json` (concatenating the shards in manifest order yields exactly the flat stream the loader wants); the workspace pipeline does not itself write `data_cache/tokens.bin` today — place or link the concatenated shard stream there (or add a stage) before the real-data path can run. See the "missing cache" pitfall in
`docs/references/data-reference.md`.

The loader (`data/shared_data/loader.py::build_training_data`) mmaps it on
each run; missing cache → `train.py` falls back to synthetic data with a
warning. The cache must be **regenerated** whenever you change the source
mixture or tokenizer.

## Skill 3: Add a new data source

Edit the **canonical mixture**, not `config.py` — the `data_sources` dict in
`config.py:get_config` is vestigial (nothing consumes it; only
`tests/test_config.py` validates it). The real recipe lives in the workspace:
`LLM/shared_data/config/mixture.yaml` (shared by all five LLM projects):

```yaml
# in LLM/shared_data/config/mixture.yaml, sources: list
- id: the-stack-v2-rust
  dataset: bigcode/the-stack-v2
  config: Rust
  split: train
  text_field: content
  weight: 0.05
  lang: rust
```

Weights must sum to 1.0 (validated at load time). Then **re-build the
shards/cache** and **re-validate dedup**. The SHA-256 dedup runs on raw text,
so existing tokens get re-hashed.

## Skill 4: Tune RoPE for long-context extension

Default `rope_theta = 500_000.0` (LLaMA-3 base) — load-bearing: AGENTS.md
rule 5 (reducing it to 10K cuts context quality dramatically).

The only RoPE knob this repo implements is `rope_theta` itself
(`config.py:get_config`). There is **no** `rope_factor` / NTK-scaling key and
no `attention_temperature` / YaRN scaling — the model trains on plain RoPE
with the 500K base and `max_seq_len = 2048`; the 500K base is its
extrapolation headroom (see `docs/concepts/attention-and-positional.md`,
"Beyond the training window"). For true NTK/YaRN scaling you would add the
frequency rescaling to `model.py:RoPE` first.

## Skill 5: Resume training from a checkpoint

Set `"preload": true` in `config.py`, or pass `config['preload']` via a
small wrapper; checkpoints live in `weights/` (`model_filename`):

```python
from train import load_checkpoint
load_checkpoint(model, optimizer, scheduler, config, device, ema=ema)
```

The resume path restores:
- model weights
- optimizer state (AdamW moments)
- LR scheduler state
- RNG state (Python + NumPy + PyTorch + CUDA)
- EMA shadow (when present)

This is **full reproducibility** — re-running gives identical loss curves.

## Skill 6: Profile memory before scaling

```python
# illustrative
torch.cuda.reset_peak_memory_stats()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    hidden = model(batch, return_hidden=True)
    loss = chunked_head_cross_entropy_with_z(
        hidden.view(-1, hidden.size(-1)), model.output_proj.weight,
        targets.view(-1), chunk_size=cfg["ce_chunk_size"])
loss.backward()
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"peak={peak_gb:.1f} GB")
```

Expected at batch 96 + grad-ckpt + chunked head CE: **~20 GB peak**.

## Skill 7: Verify FA2 is active

```python
# illustrative
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
# illustrative
from torch.optim.swa_utils import AveragedModel
# train.py builds it as AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
# shadow weights are FP32 copies under ema.module (same named params).
print(list(ema.module.state_dict().keys())[:5])
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
- **Chunked head CE is unconditional** — the training path never materializes
  the full logits tensor; `ce_chunk_size` only tunes the per-chunk slice.
- **Z-loss + QK-norm interact**: both default ON
  (`use_z_loss=True`, `qknorm=True`). Setting `z_loss_weight=0` /
  `qknorm=False` recovers the base recipe.

