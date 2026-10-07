# LLaMA-3-Lite pretrain runbook — single RTX 4090, 24 GB

How to run the 8B-token pretrain of `llama3-515M` on one RunPod RTX 4090
and not lose days of work to a preventable failure.

Written 2026-10-07. Nothing here has been executed on a GPU yet. Every
timing figure is a labelled guess until the pilot in §5.5 reports a
measurement.

---

## 1. What is being run

| Item | Value |
|---|---|
| Model | `llama3-515M`, 513.8M params, d_model 1024, 16 layers, 8 heads / 4 KV, d_ff 4096 |
| Tokenizer | `NousResearch/Meta-Llama-3-8B`, `len(tokenizer)` 128,256 |
| Sequence | 2048 |
| Recipe | 196,608 tokens per optimizer step, 42,000 optimizer steps, ~8.26B tokens |
| Corpus | `gdrive:llm-corpus/tok_llama3/shards/` — 157 shards, 7,824,922,237 tokens, 29.15 GiB |
| Optimizer | AdamW, lr 3e-4 cosine to 3e-5, warmup 2,000, wd 0.1, EMA 0.999 |
| Precision | BF16 autocast + TF32, gradient checkpointing, chunked CE (256) |
| Logging | Weights & Biases project `langgpt-llama3-pretrain` |

`config.py:get_config` is the single source of truth. The table above
mirrors it; if they disagree, config wins.

## 2. Why micro-batch 48 x accum 2, not batch 96

The repo's headline is a derived peak of 92 GB → 20 GB. That derivation
is honest about its own soft spots: the 20 GB figure assumes BF16
activation storage, while autocast keeps the residual stream in FP32,
which pushes the same math to ~26 GB. `torch.compile(mode='reduce-overhead')`
reserves extra memory during CUDA-graph capture, unquantified.

A 24 GB card cannot absorb 26 GB. `config.py` therefore splits the batch:

```
batch_size: 48            # micro-batch, this is the DataLoader batch
gradient_accumulation: 2  # 2 micro-batches per optimizer step
```

Effective batch stays 96, tokens per optimizer step stay 196,608, and the
derived strict peak falls to roughly 17 GB. The docs' "batch 96" claim is
still true at the effective level.

## 3. Bugs found while preparing this run

### 3.1 `step` counted micro-batches — on the real training path

`train.py` counted `step` per micro-batch while every metric formula
assumed `step` counted optimizer steps. With `gradient_accumulation == 1`
the two agree, so nothing was wrong until now. Setting accum to 2 would
have:

- reported `tokens_seen` at twice the real count,
- reported `tokens_per_sec` at twice the real throughput,
- stopped the run at 2.06B tokens instead of 8.26B, because `max_steps`
  counts loop iterations.

Fixed 2026-10-07: `step` now counts optimizer steps, and the inner loop
runs `gradient_accumulation` micro-batches. Every interval (`val_interval`,
`checkpoint_interval`, `generation_interval`) stays in optimizer-step
units and needs no rescaling.

### 3.2 The workspace loader aligned the split on the wrong id

Two different `shared_data` packages exist and they are not the same code:

| Package | Path | Used by | Data format |
|---|---|---|---|
| workspace | `LLM/shared_data/` | `data/prepare_data.py`, the shared test suite | `shards/` + `manifest.json` |
| vendored | `LLaMA-3-Lite/data/shared_data/` | **`train.py`** | flat `data_cache/tokens.bin` |

The workspace loader resolved the train/val split separator from
`tokenizer.eos_token_id`. The corpus is packed with `<|eot_id|>` (id
**128009**), while the tokenizer's own `eos_token_id` is
`<|end_of_text|>` (id **128001**) — a different token. Searching for
128001 in a buffer that only holds 128009 finds no boundary, so the split
silently fell back to chunk alignment and broke the documented "split on a
document boundary" invariant.

Fixed 2026-10-07 in `LLM/shared_data/loader.py`: the separator now comes
from `manifest.eos_token_id`, which is the id the packer wrote. Regression
test `shared_data/tests/test_loader.py` (`TestCorpusSeparatorId`) fails on
the old code and passes on the new.

**This does not change the LLaMA-3-Lite training path.** The vendored
loader that `train.py` imports splits `tokens.bin` at a pure chunk
boundary and never promised document alignment. Its val leakage is
bounded by one document (≤ 8192 tokens) out of ~391M val tokens. Left
alone.

### 3.3 `num_workers=0` could not start

`prefetch_factor` was passed to `DataLoader` unconditionally in the
workspace loader. PyTorch rejects it when `num_workers=0`. Fixed
2026-10-07. The vendored loader already guarded it.

### 3.4 A stale cache would be trained on silently

`config.reuse_data_cache` is `True`, so an existing `data_cache/tokens.bin`
is used as-is and the rebuild is skipped. A leftover file from a smoke
test would have been trained on for days. `doctor.py` now compares the
cache token count against the corpus manifest and fails on any mismatch.

## 4. Never train on synthetic data by accident

`train.py` used to catch `FileNotFoundError` from the data loader and
silently fall back to `build_synthetic_data`. A wrong cache path would
then burn days of GPU on random tokens with a healthy loss curve.

The fallback is now opt-in. Without `ALLOW_SYNTHETIC_DATA=1` a missing
corpus raises instead. Do not set that variable on the pod.

## 5. Order of operations

Run these in order. Do not skip the pilot.

### 5.1 Provision the pod (not done yet)

| Setting | Value |
|---|---|
| GPU | 1x RTX 4090 24 GB, dedicated (not community / not shared) |
| Disk | 200 GB |
| System RAM | 16 GB minimum, 32 GB comfortable |
| Template | PyTorch 2.x + CUDA 12.x |
| Persistence | Network volume or attached disk for `weights/` and `data_cache/` |
| Access | SSH key only. No public ports. |

### 5.2 Pull the code and the corpus

```bash
git clone <this repo> /workspace/LLaMA-3-Lite
cd /workspace/LLaMA-3-Lite

# 29.15 GiB of llama3 shards from Google Drive
export LLM_DATA_ROOT=/workspace/llm_corpus
rclone copy gdrive:llm-corpus/tok_llama3 "$LLM_DATA_ROOT" --transfers 8 --checkers 8
```

`LLM_DATA_ROOT` must be the parent of `shards/`. The bridge step below
reads `$LLM_DATA_ROOT/shards/manifest.json`.

### 5.3 Environment

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets numpy wandb
wandb login
```

### 5.4 Build the training cache

`train.py` does not read the shards. It memmaps one flat file,
`data_cache/tokens.bin`. Build it from the shards:

```bash
export LLM_DATA_ROOT=/workspace/llm_corpus
python data/prepare_data.py --skip-download --skip-clean --skip-tokenize --skip-pack
```

The flags are correct even though nothing was downloaded: the shards are
already packed on disk, and this step only runs the final shard → flat
cache concatenation. It streams shard by shard, so RAM stays flat.

Output: `data_cache/tokens.bin`, 7,824,922,237 tokens, **31.3 GB**.

Delete any pre-existing `data_cache/tokens.bin` first. `reuse_data_cache`
is `True` and will otherwise skip this step.

### 5.5 Preflight

```bash
python scripts/doctor.py --checksums
```

This verifies: GPU and VRAM against the derived peak, BF16 support,
`LLM_DATA_ROOT`, manifest identity (tokenizer `llama3`, token count,
shard count), every shard's sha256, document separators counted from the
bytes, the `data_cache/tokens.bin` token count against the manifest, free
disk, system RAM, and W&B credentials.

**Exit code must be 0.** A single FAIL means stop.

`--checksums` reads all 29 GiB and takes a few minutes. Run it at least
once after the download, and again before any resume that follows a crash
or a re-download.

### 5.6 The pilot — 30 minutes, do not skip

Before committing to a multi-day run, measure the real step time on the
real card.

```bash
python tests/e2e_gpu_smoke.py --steps 30
```

Then start the real training and watch the first 200 steps:

```bash
nohup python train.py > train.log 2>&1 &
tail -f train.log
```

Read from the log / W&B:

| Metric | What it must show |
|---|---|
| `gpu/memory_peak_mb` | under ~21,000 MB at micro-batch 48 |
| `train/tokens_per_sec` | stable, not falling over steps |
| `train/data_wait_ms` | low relative to `train/step_time_ms` |
| `train/loss` | finite, falling |

**Real bill = measured tokens/s.** Compute it as
`8,257,536,000 / tokens_per_sec` seconds, then multiply by the pod's
hourly rate. The estimate in §7 is a guess until this number exists.

If `gpu/memory_peak_mb` is near 24,000, lower `config['batch_size']` to
32 and raise `gradient_accumulation` to 3. Tokens per optimizer step stay
the same. `max_steps` does not change.

### 5.7 Full run

```bash
nohup python train.py > train.log 2>&1 &
```

`nohup` keeps the run alive when SSH drops. The pod is dedicated and
persistent, but a pod that is stopped and restarted loses the process,
not the disk, if `weights/` and `data_cache/` are on a network volume.

## 6. Disk and RAM budget

| Item | Size |
|---|---|
| `shards/` (downloaded) | 29.15 GiB |
| `data_cache/tokens.bin` (built) | 31.3 GB |
| `weights/` at peak (3 step ckpts + best + final) | ~25 GB |
| PyTorch + pip environment | ~15 GB |
| **Total** | **~100 GB** |

Fits 200 GB with room to spare. Keep `shards/` until `tokens.bin` is
built and `doctor.py --checksums` is green; then it can be deleted if
space gets tight (re-downloadable from Drive).

RAM: the training loader memmaps `tokens.bin`, so the corpus does not sit
in system RAM. 32 GB is comfortable; 16 GB is the floor.

## 7. Wall clock and cost — labelled guess

Training FLOPs are about `6 x 513.8e6 x 8.26e9 = 2.55e19`.

RTX 4090 BF16 dense peak is roughly 165 TFLOPS. That is a ceiling, not a
forecast. Net of gradient-checkpointing recompute and non-GEMM work:

| Assumed MFU | Wall clock | Cost at $0.38/h |
|---|---|---|
| 25% | ~7.2 days | ~$65 |
| 30% | ~6.0 days | ~$54 |
| 40% | ~4.5 days | ~$41 |

Treat these as a range to budget against, not a plan to schedule by.
Replace the whole table with one line once §5.6 reports a measured
`train/tokens_per_sec`.

## 8. Checkpoints and survival

| Item | Size | Where |
|---|---|---|
| Full checkpoint (model + AdamW + EMA + RNG) | ~8 GB | `weights/llama3-515M_step_N.pt` |
| Best model (weights only) | ~1 GB | `weights/llama3-515M_best.pt` |
| Final checkpoint | ~8 GB | `weights/llama3-515M_final.pt` |

`keep_last_n_checkpoints: 3` prunes older step checkpoints.

A checkpoint is only useful if it survives the pod. Every N hours:

```bash
rclone copy weights gdrive:llm-corpus/weights_llama3 --transfers 2
```

### Resume

`config['preload']` selects a checkpoint; the checkpoint load restores
model, optimizer, scheduler, RNG states (torch, numpy, python, cuda) and
EMA, then continues from the stored optimizer step.

To resume after a crash, leave `preload` set and rerun `python train.py`.
It picks the highest `*_step_*.pt` in `weights/`.

## 9. Monitoring

W&B is the primary surface. `train.py` logs:

- `train/loss`, `train/lr`, `train/grad_norm`, `train/step_time_ms`
- `train/tokens_per_sec`, `train/tokens_seen`, `train/effective_batch_size`
- `train/data_wait_ms`
- `gpu/memory_used_mb`, `gpu/memory_peak_mb`, `gpu/memory_reserved_mb`, `gpu/utilization_pct`
- `val/loss`, `val/perplexity` every `val_interval` (2,000 steps)
- `gen/samples` as a W&B table every `generation_interval` (20,000 steps)

Local mirror is `train.log`:

```bash
tail -f train.log
```

### What a bad run looks like

| Signal | Meaning | Action |
|---|---|---|
| `gpu/memory_peak_mb` climbing toward 24,000 | fragmentation or too-large micro-batch | restart at batch 32 / accum 3 |
| `train/loss` = `nan` or `inf` | LR too high, or a corrupt shard | stop; re-run `doctor.py --checksums` |
| `val/perplexity` flat for >10k steps | dead run | stop, do not burn the card |
| `train/data_wait_ms` large vs `step_time_ms` | dataloader starved | raise `num_workers` |
| loss curve identical to a previous run | training on synthetic data | check the log for the synthetic warning |
| loss far lower than a comparable 515M run | training on a stale/toy cache | re-run `doctor.py`, check `data.cache` |

## 10. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-07 | RTX 4090 24 GB, dedicated RunPod pod, 200 GB, persistent | user's hardware choice |
| 2026-10-07 | micro-batch 48 x accum 2 | derived strict peak at batch 96 is ~26 GB; 24 GB card |
| 2026-10-07 | `step` counts optimizer steps | metric formulas and `max_steps` already assumed it |
| 2026-10-07 | synthetic-data fallback opt-in | a wrong data path must not burn days of GPU |
| 2026-10-07 | workspace split aligns on `manifest.eos_token_id` | corpus separator 128009 != tokenizer eos 128001 |
| 2026-10-07 | vendored split left chunk-only | leakage ≤ 1 document out of 391M val tokens; not worth touching |
| 2026-10-07 | doctor counts separators from bytes | `manifest.n_eos` records 0 on all 157 shards while separators exist |
| 2026-10-07 | doctor checks `tokens.bin` token count | `reuse_data_cache: True` would train on a stale cache silently |
| 2026-10-07 | W&B project `langgpt-llama3-pretrain`, tags `rtx4090`/`runpod` | reflect the real hardware |
| 2026-10-07 | pod not deployed | user asked for repo prep and plan first |

## 11. Known issues not fixed here

`shared_data/tests/test_shims.py` (`test_shim_end_to_end`, Mamba-3-Lite
case) fails on a clean tree. It expects 5 documents after clean+dedup and
gets 10. `DedupFilter` is exact-match SHA-256 by design (see its module
docstring), so it cannot drop the near-duplicates the fixture contains.
The failure is in the clean/dedup stage and is not on the LLaMA-3-Lite
training path. Pre-existing, left alone.

The shared pipeline also warns `mixture.total_tokens (9,500,000,000) !=
target_total_tokens (8,000,000,000)`. Harmless here because the corpus is
already packed, but the two config files disagree.
