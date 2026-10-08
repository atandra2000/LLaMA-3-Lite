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

### 3.5 `compile_mode='reduce-overhead'` crashed on step 0

`reduce-overhead` captures CUDA graphs. Gradient checkpointing recomputes
the forward pass during backward, which overwrites the graph's output
buffer:

```
RuntimeError: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run ... model.py:Transformer.forward
    x = self.input_embedding(x)
```

Fixed 2026-10-07: `compile_mode` is now `'default'`. Keep the fusion wins,
drop the graphs. This is a correctness bug, not just the memory
reservation that §2 warned about.

### 3.6 micro-batch 48 OOMed at 23.4 GB

The derived peak of ~17.2 GB is not what the card actually used. Real
peak at micro-batch 48 was **23.39 GB**, 192 MiB short of a 23.52 GiB
card. `torch.compile` activations sit on top of the paper estimate, and
autocast keeps the residual stream in FP32 as §2 predicted.

Fixed 2026-10-07: `batch_size` 48 → 32, `gradient_accumulation` 2 → 3.
Tokens per optimizer step stay 196,608, effective batch stays 96,
`max_steps` does not change. Real peak is **21,204 MB** — fits with
~3 GB of margin.

Also launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the
environment. `train.py` sets it from config, but that runs after
`import torch`; exporting it before the process starts is the reliable
way.

### 3.7 `torch.cuda.utilization()` killed the run at step 50

`log_interval` is 50, so the first metric flush calls
`torch.cuda.utilization()`, which needs `pynvml`. That package is not in
the image or in the runbook's pip line, so the run died at the first log
with `ModuleNotFoundError: pynvml does not seem to be installed`. No
checkpoint had been written yet.

Fixed 2026-10-07 two ways: `pip install nvidia-ml-py`, and the call is
now wrapped in `try/except` in `train.py` so telemetry can never kill a
multi-day run again.

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
| Template | `runpod-torch-v280` — `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Persistence | Network volume or attached disk for `weights/` and `data_cache/` |
| Access | SSH key only. No public ports. |
| Data center | **`EU-RO-1`** — the only DC that has RTX 4090 stock *and* STANDARD network volumes |

Verified against the live catalog 2026-10-07. The RTX 4090 fleet reports
CUDA **12.8 / 13.0 / 13.2 / 13.3** only — not 12.4. `runpod-torch-v280` is
the official template whose `allowedCudaVersions` covers 12.8, and it
already ships `torch 2.8.0+cu128`. A live 4090 in the golden-path run
reported `2.8.0+cu128 True`, so do not `pip install torch` on top of it.

Stock is **LOW** on all six 4090 data centers (EU-CZ-1, EU-RO-1, EUR-IS-1,
EUR-IS-2, US-CA-2, US-IL-1). Provisioning may need a retry or a short wait.

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
# torch 2.8.0+cu128 is already in the image. Install the rest only.
pip install transformers datasets numpy wandb nvidia-ml-py
```

W&B is wired by env var, not by a login prompt. Pass the key at pod create
time so nothing needs a human at the keyboard:

```bash
runpodctl pod create ... --env '{"WANDB_API_KEY":"<key>"}'
```

`wandb` reads `WANDB_API_KEY` on its own, so skip `wandb login` entirely.
Locally the key already lives in `~/.zshrc` and `~/.wandb/settings`; entity
resolves to `atandrabharati-self`. `config.py` sets the project to
`langgpt-llama3-pretrain` and leaves `wandb_entity` as `None`, which is
correct — `None` means "use the default entity".

### 5.4 Build the training cache

`train.py` does not read the shards. It memmaps one flat file,
`data_cache/tokens.bin`. Build it from the shards:

```bash
export LLM_DATA_ROOT=/workspace/llm_corpus
python data/prepare_data.py --concat-only
```

Use `--concat-only`, not the four `--skip-*` flags. The skip form imports the
universal pipeline at `LLM/shared_data/`, which the pod does not have — the
repo vendors only the loader. `prepare_data.py` says so itself in its own
error text and names `--concat-only` as the fix for exactly this corpus.
It streams shard by shard, so RAM stays flat.

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
forecast.

**Measured 2026-10-07 on the real 4090** (run `5gq8nhw1`, W&B project
`langgpt-llama3-pretrain`), replacing the estimate table that was here:

| Metric | Measured |
|---|---|
| `train/step_time_ms` | 12,150 – 12,200 ms |
| `train/tokens_per_sec` | **16,140** |
| `gpu/memory_peak_mb` | **21,204** of 24,564 |
| `gpu/utilization_pct` | 93 – 100 |
| `train/data_wait_ms` | 56 (vs 12,200 ms step — dataloader is not the limit) |
| Full run (42,000 steps) | **5.92 days** |
| Cost at $0.761/h | **$108** |

So the old 30% MFU guess (~6.0 days, ~$107) was almost exactly right. Budget
against **$108** for a full run.
Replace the whole table with one line once §5.6 reports a measured
`train/tokens_per_sec`.

## 8. Checkpoints and survival

| Item | Size | Where |
|---|---|---|
| Full checkpoint (model + AdamW + EMA + RNG) | ~8 GB | `weights/llama3-515M_step_N.pt` |
| Best model (weights only) | ~1 GB | `weights/llama3-515M_best.pt` |
| Final checkpoint | ~8 GB | `weights/llama3-515M_final.pt` |

`keep_last_n_checkpoints: 3` prunes older step checkpoints.

`checkpoint_interval` is **500**, not 5000. At ~12 s/step that is roughly
1.7 hours of work at risk, not 17. `train.py` only writes a `*_final.pt`
after all 42,000 steps, and it has no signal handler — so anything that
stops the process early keeps whatever the last step checkpoint holds.
A 5000-step interval meant an interrupted run saved nothing at all.

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

### Budget-limited stop, ship, resume

When the account balance is the limit rather than `max_steps`, stop before
the money runs out and ship the checkpoint off the pod. A pod that dies at
zero balance is killed mid-write.

```bash
# 1. stop training. the newest *_step_*.pt is what we keep.
pkill -f 'python train.py' ; sleep 20   # 20 s covers a mid-save async thread

# 2. ship ONLY the newest checkpoint. weights/ holds ~25 GB across 3 pruned
#    step ckpts + best; uploading all of it is billed pod time for nothing.
LATEST=$(ls -t weights/*_step_*.pt | head -1)
rclone copy "$LATEST" gdrive:llm-corpus/weights_llama3 --transfers 2
rclone copy weights/llama3-515M_best.pt gdrive:llm-corpus/weights_llama3 --transfers 2

# 3. record where to resume
ls -la weights/ | rclone rcat gdrive:llm-corpus/weights_llama3/RESUME.txt
```

On resume, put the newest `*_step_*.pt` back in `weights/`, leave `preload`
set, and rerun `python train.py`. It continues from the stored optimizer
step with model, AdamW, scheduler, RNG and EMA restored.

### A100 resume — planned 2026-10-07, not yet run

Decision: stop the 515M run at step 2000 and finish it on 1x A100 80GB.

Measured on the 4090: 16,140 tok/s at $0.761/h, which is $13.1 per 1B
tokens. An A100 80GB is $1.59/h on Runpod. The remaining 7.86B tokens cost
about $102 and take 134 h on the 4090. On an A100 with the settings below
they cost about $87 and take 55 h.

**Keep 196,608 tokens per optimizer step.** Batch 96 x 2048 x accum 1 is
the same count as 32 x 2048 x 3. The checkpointed scheduler and the loss
curve then continue without a kink. Micro-batching can change freely. Do
not change `max_steps` or `learning_rate`; the cosine schedule is bound to
`max_steps` and a mid-run change moves the target.

| key | 4090 | A100 | why |
| --- | --- | --- | --- |
| `batch_size` | 32 | 96 | fills 80 GB |
| `gradient_accumulation` | 3 | 1 | holds tokens/step at 196,608 |
| `gradient_checkpointing` | True | False | drops the recompute forward pass, 25-33% |
| `compile_mode` | `default` | `reduce-overhead` | CUDA graphs conflicted with gradient checkpointing; with GC off they are legal |
| `ce_chunk_size` | 256 | 1024 | fewer kernel launches at large batch |
| `max_steps` | 42000 | 42000 | do not touch |
| `learning_rate` | 3e-4 | 3e-4 | do not touch |

`train.py:load_checkpoint` restores model, AdamW, scheduler, RNG and EMA,
so the card and the micro-batching are the only things that change.

**The token cache dies with the pod.** `data_cache/tokens.bin` is 31.3 GB
on pod-local overlay storage, not on a network volume. It is preserved at
`gdrive:llm-corpus/tok_llama3_cache/tokens.bin`, verified at 31,299,688,948
bytes on 2026-10-07. Download it instead of rebuilding: 28 MiB/s up from the
pod measured 29.3 MiB/s, so the round trip is about 20 min, against 1-2 h to
re-pull the corpus and run `python data/prepare_data.py --concat-only` again.

Steps on the new pod:

```bash
# 1. provision 1x A100 80GB, 200 GB disk, Secure Cloud
# 2. ship the local tree with tar, do not clone origin/main
# 3. venv that inherits the image torch
python3 -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install transformers datasets numpy wandb nvidia-ml-py
# 4. the token cache, straight off Drive. do not rebuild it.
mkdir -p data_cache
rclone copy gdrive:llm-corpus/tok_llama3_cache data_cache \
  --transfers 2 --log-file /workspace/cache_download.log --log-level INFO
python3 -c "import os; assert os.path.getsize('data_cache/tokens.bin')==31299688948"
# 5. edit config.py with the table above
# 6. put llama3-515M_step_2000.pt in weights/, leave preload set
python train.py
```

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
| 2026-10-07 | template `runpod-torch-v280` (torch 2.8.0+cu128) | 4090 hosts expose CUDA 12.8+ only, not 12.4; image already has torch |
| 2026-10-07 | budget at Secure $0.74/h | live catalog price; the $0.38/h estimate was never real |

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
