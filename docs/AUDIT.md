# LLaMA-3-Lite — Documentation & Codebase Audit

> **Scope.** Full audit of the docs↔code alignment and a from-scratch
> explanation of the codebase, verified against the working tree on
> 2026-08-05. Every claim below was checked by running the code or
> inspecting the exact source cited. Nothing is asserted from memory.
>
> **Verification runs.** `python3 tests/test_doc_refs.py` → OK (16 docs,
> all citations resolve, no line anchors); `python3 -m pytest tests/ -q`
> → 70 passed, 1 skipped (GPU); `python benchmark_data.py` → **crashes**
> (finding C1); `python3 data/prepare_data.py --stage pretrain
> --skip-*` → runs but emits `data/shards/` + `manifest.json`, **never**
> `data_cache/tokens.bin` (finding C2).

---

## 1. State of the docs — what is already excellent

The doc corpus was massively expanded and cleaned in the last 5 hours of
commits (764a7ba, 411a9b4, 13a880e) plus an uncommitted working-tree pass
(+1,130/−713 lines across 15 files). The current state is unusually good:

- **Corpus:** 13 files under `docs/`, 114,216 words (measured `wc -w`),
  matching `docs/README.md`'s own table exactly (119,475 repo-wide).
- **Structure:** concepts (4) / references (3) / guides (4) / training.md /
  README nav map. The nav map's file→doc table replaces the retired
  `CODE_MAP.md`; no dead generator remains.
- **Machine gate:** `tests/test_doc_refs.py` resolves every
  `<module>.py:<symbol>` citation by import + hasattr, bans line-number
  anchors, validates intra-repo links, and requires python snippets to be
  marked `# illustrative` / `# verified`. It runs in CI
  (`.github/workflows/ci.yml`). All 16 docs pass.
- **Symbol coverage:** every public symbol in `model.py` (18), `train.py`
  (7), `config.py` (1), `benchmark_data.py` (3), all three kernels,
  `data/prepare_data.py`, and `data/shared_data/loader.py` (7) is cited
  at least once. Zero gaps (verified by AST scan).
- **Honesty discipline:** the docs flag their own estimates
  (`[INFERENCE]`, "measured / derived / estimated" tables in
  `docs/concepts/training-and-memory.md`), openly reconcile the 78%
  headline against alternative accounting (92→20 = 78.3%; 130→20 = 85%;
  strict ≈ 91%), and call out top-level stale claims rather than
  laundering them.
- **Numeric claims verified:** 513,840,128 params (513.8M, non-embed
  251.7M); full logits 50.3 GB BF16 / 100.6 GB FP32 at B·S·V =
  196,608×128,000; chunk 256 → 65.5 MB BF16 / 131 MB FP32; saved block
  inputs 16 × 402.65 MB = 6.44 GB; model state 2.06 (weights) + 2.06
  (grads) + 4.11 (Adam m/v) + 2.06 (EMA) = 8.23 GB — all re-derived from
  config and confirmed.

---

## 2. Findings table — docs↔code misalignment

| # | Severity | Where | Doc claim | Code reality (verified) |
|---|----------|-------|-----------|--------------------------|
| **C1** | **bug** | `benchmark_data.py:benchmark` | Docs (`docs/guides/quickstart.md`, `docs/concepts/training-and-memory.md`) present `python benchmark_data.py …` as a working command | **Crashes immediately**: `PackedDataset.__init__() got an unexpected keyword argument 'eos_id'`. The loader's `data/shared_data/loader.py:PackedDataset` takes only `(tokens, seq_len)` — no `eos_id`. The `eos_id=0` kwarg is a leftover from the pre-consolidation loader. Every documented benchmark invocation fails on line 1 of the loop. |
| **C2** | **bug** | `docs/training.md`, `docs/references/data-reference.md`, `docs/guides/quickstart.md` | "the packing stage produces `data_cache/tokens.bin`"; "`data/prepare_data.py:main` … produces `tokens.bin`"; "stored in `data_cache/tokens.bin`" | **Nothing produces `data_cache/tokens.bin`.** Verified: the workspace pipeline (`LLM/shared_data/scripts/pack_shards.py`) writes `DATA_ROOT/shards/shard_*.bin` + `manifest.json` (DATA_ROOT = `$LLM_DATA_ROOT` or `$PWD/data`); grep for `tokens.bin`/`data_cache` in the workspace = 0 hits. The loader `data/shared_data/loader.py:build_training_data` mmaps `data_cache/tokens.bin` exclusively. So Mode 2 (real data) is **unwired end-to-end**: `train.py:train_model` always falls to synthetic. `docs/concepts/data-and-kernels.md` pitfall 1 already documents this gap correctly — but three other docs (and `docs/training.md:899`) contradict it. |
| **C3** | stale | `docs/concepts/data-and-kernels.md:795` | "AGENTS.md's 'Sanctioned Triton paths' list still reads '(none yet)'" | AGENTS.md was rewritten (uncommitted): it now lists all three kernels (`kernels/rmsnorm_triton.py`, `kernels/swiglu_triton.py`, `kernels/cross_entropy_triton.py`) with their config keys. The "stale" claim is itself stale. Same text at `docs/references/data-reference.md:1041–1042` ("still says 'no custom Triton kernels exist'"). |
| **C4** | stale | `docs/training.md:735–737` ("Why '7 techniques' in AGENTS.md but 8 rows here") | "AGENTS.md's '7-technique memory stack' … includes `channels_last` … and 'Fused AdamW'" | AGENTS.md now says "8-technique memory stack" and its rows are GQA + fused SwiGLU — `channels_last` / Fused AdamW appear nowhere in the file (grep = 0). The correction paragraph is now wrong on both counts; the 8-row table below it matches AGENTS.md exactly. |
| **C5** | stale | `docs/concepts/data-and-kernels.md:162–163` | "The workspace `README.md` §5 still shows an older five-source recipe (fineweb-edu 0.50 / fineweb 0.20 / the-stack-python 0.15 / openmath 0.10 / arxiv 0.05)" | That exact recipe lives in sibling-project docs (`LLM/Mamba-3-Lite/README.md`, `LLM/DeepSeek-v3-Lite/AGENTS.md`), **not** the workspace README (§5 is "TranslationLM"). The root `CoreProjects/README.md` §4 LLaMA-3-Lite entry carries a *different* stale six-source recipe (FineWeb-Edu 0.5 / FineWeb-Code 0.1 / Stack-Python 0.2 / multi-lang 0.05 / Wikipedia 0.05 / SO-QA 0.05) plus `channels_last`, "fused AdamW", `GradScaler`, "architecture.md (1,234 lines)", "~16 GB cache" — all false vs. the tree. The citation is wrong in both file and content; the underlying "top-level docs rot" claim is still true, just mislocated. |
| **C6** | stale | `docs/references/model-reference.md:1032`, `docs/guides/glossary.md:153` | `use_z_loss` = "Informational in this repo" / "Informational toggle (behavior follows `z_loss_weight`)" | **Correct and verified** — but the sibling claim in `docs/concepts/training-and-memory.md:1221` and `:1592` ("compare … with a run that disables z-loss (`use_z_loss = False`)") implies `use_z_loss=False` disables the term. It does not: `z_loss_weight` is the only functional switch; `train.py:train_model` reads `use_z_loss` only in the startup banner print. Cross-doc inconsistency. |
| **C7** | stale | `docs/references/model-reference.md:1035, 1104–1111` | EMA shadow "costs one full BF16 copy (~1.03 GB)"; memory table rows "Weights (BF16) 1.03 GB", "EMA shadow (BF16) 1.03 GB" | `AveragedModel` deep-copies the model; params are FP32 (no `.bfloat16()` anywhere in `train.py` — grep-verified), so the EMA shadow is **2.06 GB** and live weights are 2.06 GB. `docs/concepts/training-and-memory.md:867,958` state this correctly ("2.06 GB … deep-copies the FP32 model"). model-reference's own "Peak total ≈ 23 GB" (22.9) actually uses the 8.23 GB model-state row, so only the per-row labels are wrong — but they disagree with the concept doc and with the code. |
| **C8** | stale | `docs/references/model-reference.md:1121` | "the 92 GB figure in AGENTS.md is the measured value for that configuration [measured]" | `.benchmarks/` is empty, no training run has ever completed (README status banner + `docs/concepts/training-and-memory.md:1033` both say so). Nothing was measured. The same doc's "≈23 GB … [measured per project docs]" is likewise unmeasurable today. |
| **C9** | stale | `docs/guides/quickstart.md:16` | wandb "imported unconditionally by `train.py` — install even for offline runs" | True, but the tests run without wandb because `tests/conftest.py` installs a stub. The doc's own quickstart "check commands" claim that everything works without wandb on a bare checkout — the stub is conftest-only; a *manual* `python train.py` without wandb installed fails at `import wandb`. Nuance worth one sentence. |
| **C10** | nit | `docs/guides/troubleshooting.md:239` | Prevention snippet: `from data.shared_data.loader import _SyntheticTokenizerStub` | Works (`data/` has no `__init__.py` but is a namespace package under Python 3.3+; verified import). Fine as-is — no change needed. |
| **C11** | nit | `docs/concepts/data-and-kernels.md:807–808` | "SKILLS.md still teaches adding sources to `config.py:get_config`" | SKILLS.md Skill 3 now says the opposite ("Edit the **canonical mixture**, not `config.py`"). Same stale-claim-about-a-stale-file pattern as C3/C4 — SKILLS.md was fixed in the same working-tree pass. |
| **C12** | risk | `docs/references/model-reference.md:1122`, `docs/concepts/training-and-memory.md:975` | "consistent with the ~20 GB advertised [measured per project docs]" | No measurement exists anywhere (see C8). The docs' own honest accounting gives 20 GB with ~20% variance / 24–26 GB strict. Recommend deleting every `[measured]` tag on the headline until a real run exists. |

---

## 3. From-scratch codebase explanation

### 3.1 The one-paragraph view

LLaMA-3-Lite is a from-scratch, raw-PyTorch, LLaMA-3-style decoder-only
transformer (513.8M params, no HF Trainer/Lightning) whose entire design
revolves around one number: **fitting batch 96 × seq 2048 (196,608
tokens/step) on a single A100 80 GB** with ~2× headroom. It does that with
eight cooperating techniques — gradient checkpointing, chunked LM-head CE +
z-loss, disk-backed mmap token cache, BF16 autocast, FA2 via SDPA, GQA,
fused SwiGLU (+ three opt-in Triton kernels), TF32 — plus three stability
features (QK-norm, z-loss, EMA). The full 42,000-step / 8.26B-token run has
not started; the repo is smoke-tested end to end on synthetic data.

### 3.2 File map

| File | Role | Lines |
|------|------|------:|
| `config.py` | Single source of truth for every hyperparameter + runtime toggle (`get_config`) | 105 |
| `model.py` | The transformer: RoPE, RMSNorm, GQA, SwiGLU, Decoder/Transformer, both chunked losses, `build_transformer` | 384 |
| `train.py` | Training loop, optimizer/scheduler/EMA, validate, generate, checkpoint/RNG restore, Triton gate | 614 |
| `dataset.py` | 32-line re-export shim → `data/shared_data/loader.py` | 32 |
| `data/shared_data/loader.py` | `PackedDataset` (mmap windows), `ShuffledRangeSampler`, `collate_fn`, `build_tokenizer`, `build_training_data`, `build_synthetic_data`, `_SyntheticTokenizerStub` | 209 |
| `data/prepare_data.py` | CLI shim delegating corpus construction to workspace `LLM/shared_data` | 72 |
| `kernels/{rmsnorm,swiglu,cross_entropy}_triton.py` | Opt-in Triton kernels + CPU-runnable PyTorch references | ~360 total |
| `benchmark_data.py` | Data-pipeline microbenchmark — **currently broken (C1)** | 157 |
| `tests/` | 70 passing tests (4 files + conftest) + standalone `e2e_gpu_smoke.py` + doc gate | ~1,900 |

### 3.3 `model.py` — the forward pass

1. **`Transformer.forward`** (`model.py:Transformer`): embed `[B,S]` ids →
   `[B,S,1024]`, run 16 `DecoderBlock`s, final `RMSNorm`, then either
   return hidden (training: `return_hidden=True`) or project to logits
   (generation/validation-free paths). With
   `gradient_checkpointing` on, each layer is wrapped in
   `torch.utils.checkpoint(..., use_reentrant=False)`; the final norm is
   applied explicitly in both branches (a historical bug — a skipped final
   norm — is fixed and covered by tests).
2. **`DecoderBlock`** (`model.py:DecoderBlock`): pre-norm residual
   `x = x + attn(norm₁(x))`, `x = x + ffn(norm₂(x))`. No dropout anywhere.
3. **`GroupedQueryAttention`** (`model.py:GroupedQueryAttention`): q
   `[B,S,8,128]`, k/v `[B,S,4,128]`; per-head `RMSNorm(head_dim)` (QK-norm,
   Qwen2/Gemma2 style) after projection, before RoPE; RoPE applied in
   head-transposed layout; K/V expanded `4→8` via
   `.expand(...).reshape(...)` (the 2×-KV-cache win); `F.scaled_dot_product_attention(..., is_causal=True)` — FA2 backend on Ampere, O(S) memory, no materialized score matrix. There is no `mask` parameter; causality lives in `is_causal`.
4. **`RoPE`** (`model.py:RoPE`): precomputed `cos_cached`/`sin_cached`
   `[1,1,2048,64]` from `inv_freq = theta^(-2i/128)`, θ=500,000 (LLaMA-3
   long-context choice, AGENTS.md hard rule 5); applies the rotation by
   splitting the last dim into even/odd halves and flattening back.
5. **`SwiGLUFFN`** (`model.py:SwiGLUFFN`): one fused `gate_up_proj`
   `[B,S,8192]` GEMM instead of two; `silu(gate) * up` → `down_proj`.
6. **Loss** (`model.py:chunked_head_cross_entropy_with_z`): the memory
   headline. Never materializes `[196608, 128000]` logits (50.3 GB BF16).
   Slices hidden into `ce_chunk_size` (256) rows, computes
   `F.linear(hidden_c, head_weight)` inside `checkpoint` (each chunk's
   131 MB FP32 logits is recomputed in backward), applies FP32
   logsumexp/z-loss + CE with `ignore_index=-100`, accumulates
   (ce_sum, count, z_sum). z-loss = `(logsumexp logits)²` averaged over
   non-ignored tokens, weight 1e-4 — the PaLM/Gemma2 late-run-collapse
   regularizer. `chunked_cross_entropy_with_z` is the dense-input variant
   used by tests.
7. **`build_transformer`** (`model.py:build_transformer`): config → model;
   prints param counts (513.8M total / 251.7M non-embed) and active Triton
   paths.

### 3.4 `train.py` — the loop

1. **GPU setup** (`train.py:setup_gpu_optimizations`): TF32 flags gated on
   `config['tf32']`, but `torch.set_float32_matmul_precision('high')` is
   called unconditionally — so matmul TF32 stays on even with `tf32=False`
   (documented in training-and-memory.md pitfall; a real footgun).
2. **Data** (`train.py:train_model`): tries
   `data/shared_data/loader.py:build_training_data`; on `FileNotFoundError`
   (always, today — see C2) falls back to synthetic random uint32 with a
   loud WARN. `ignore_index = -100` hard-coded ("keeps EOS separators
   learnable").
3. **Triton gate**: `*_impl='triton'` keys are force-restored to
   `'pytorch'` unless `ENABLE_TRITON_KERNELS=1` (AGENTS.md rule 7 —
   default runs can never silently run a fused path).
4. **Model**: `build_transformer(real_vocab_size = max(config['vocab_size'],
   len(tokenizer)))`, `.to(device)` — FP32 params, BF16 compute via
   `torch.autocast`. `torch.compile(model, mode='reduce-overhead')`
   (CUDA graphs) with a pre-warmup forward+backward to absorb autotune.
5. **Optimizer/schedule**: AdamW with decay on `p.dim() >= 2` only (0.1 on
   2D+, 0.0 on 1D); `SequentialLR(LinearLR 3e-5→3e-4 over 2000, cosine
   3e-4→3e-5 over 40000)`. EMA via `torch.optim.swa_utils.AveragedModel`
   + `get_ema_multi_avg_fn(0.999)` — shadow under `ema.module`, FP32.
6. **Step**: prefetch next batch first (`non_blocking` H2D + pinned memory
   — the only async compatible with CUDA-graph stream ownership), forward
   `return_hidden`, chunked head CE, `loss / grad_accum`, bare
   `loss.backward()` (no GradScaler — BF16 keeps FP32 exponent range),
   clip, step, `ema.update_parameters`, zero-grad, `scheduler.step()`.
   `train.py:_next_batch` wraps `StopIteration` → epoch bump +
   `sampler.set_epoch` so the 42k-step plan (8.26B tokens) survives a
   7.6B-token train split (~1.09 passes).
7. **Cadence**: log every 50 (loss, lr, grad norm, tok/s, data wait, GPU
   memory); validate every 2000 on **EMA weights**, chunked loss, PPL
   `exp(min(loss, 20))`, best-model save; generate every 20000 (5 prompts,
   128 tokens, top-k 50 / top-p 0.9 / temp 0.8, stop on
   `tokenizer.eos_token_id`); checkpoint every 5000 (full state + 4 RNG
   streams + EMA), async via `threading.Thread` (default `async_checkpoint`
   True) with final `join()`, prune to `keep_last_n_checkpoints`.
8. **Resume** (`train.py:load_checkpoint`): restores model/optim/sched/
   step/best-val/EMA/RNG — but **not** the sampler epoch/offset
   (documented gap: `epoch_state` restarts, so a resumed run's window
   order differs; the permutation itself is bit-identical thanks to RNG
   restore).

### 3.5 Data path

- **Loader** (`data/shared_data/loader.py`): `PackedDataset` slices
  `seq_len+1` windows off an `np.memmap(uint32)` (page-fault resident RAM
  ≈ 1 MB for an 8B-token corpus), no EOS awareness — windows are
  position-only. `ShuffledRangeSampler` = deterministic permutation
  `default_rng(seed + offset)`, `set_epoch` bumps offset. `collate_fn`
  stacks. `build_tokenizer` = `AutoTokenizer.from_pretrained`, pad→eos
  fallback; failures degrade to `_SyntheticTokenizerStub` (bytes⇄ids,
  `len` = vocab).
- **Pipeline** (workspace `LLM/shared_data/`): mixture.yaml (7 sources,
  8.0B tokens: fineweb-edu .40 / dclm-baseline .15 / stack-v2-python .15 /
  stack-v2-jupyter .05 / openmath .10 / arxiv .10 / cosmopedia .05) →
  download → clean+dedup (SHA-256) → tokenize (llama3, EOS 128,009, pad
  128,002) → `pack_shards` (50M-token shards, EOS-separated, manifest).
  **Gap (C2):** the loader reads `data_cache/tokens.bin`; the pipeline
  writes `data/shards/*.bin` + `manifest.json`. Concatenating shards in
  manifest order yields exactly the flat stream the loader expects — the
  wiring is a small missing stage, but missing.
- **Config residue:** `config.py:get_config`'s `data_sources` dict (6
  entries summing 0.95) is consumed by nothing — the loader reads only 12
  keys (verified by regex over loader source). Tests pin the surface.

### 3.6 Kernels (all opt-in, all CPU-testable)

Each of the three `kernels/*_triton.py` files: pure-PyTorch reference
(`rmsnorm_pytorch`, `swiglu_pytorch`, `cross_entropy_with_z_pytorch`),
`HAS_TRITON` via `try: import triton`, `@triton.jit` kernel, a
`torch.autograd.Function` wrapper whose backward is a **re-compute stub
through the reference**, and a public entry point that raises
`ImportError` when Triton is absent. Model dispatch
(`model.py:RMSNorm.forward`, `model.py:SwiGLUFFN.forward`,
`model.py:chunked_cross_entropy_with_z`,
`model.py:chunked_head_cross_entropy_with_z`) makes a direct call with
no `try/except`; any failure (missing triton, tripped shape guard,
compile error, OOM) propagates as a hard error — matching AGENTS.md rule
7's "must surface a clear error, not a silent fallback." Guard rails: `_MAX_BLOCK_SIZE = 8192` (RMSNorm/SwiGLU rows),
`_MAX_VOCAB_BLOCK = 131072` (CE vocab axis). No microbenchmark exists
(`scripts/microbench_a100.py` referenced by AGENTS.md rule 2 is absent),
so the 1.5× contract is unmeasured.

### 3.7 Tests

70 passed / 1 skipped on CPU (Mac M1, ~20 s): `test_config.py` (7 —
REQUIRED_KEYS bijection, production values, invariants), `test_model.py`
(35 — RMSNorm/RoPE/GQA/SwiGLU math, chunked≡dense CE equivalence,
grad-ckpt≡eager equivalence, QK-norm identity, param counts),
`test_smoke.py` (4 — end-to-end steps, loss descent), `test_train.py` (13
— sampling, checkpoint round-trip incl. exact RNG restore, async save,
GPU-setup idempotence; 1 GPU-skipped). `e2e_gpu_smoke.py` runs 8
assert-guarded stages on real hardware. `conftest.py` injects a wandb stub
and seeds deterministically. CI: import check → full CPU suite → doc-ref
checker.

---

## 4. Modification plan (priority order)

### P0 — fix broken real-data path (blocker for the flagship use case)

1. **Wire `tokens.bin` production.** Smallest correct fix: add a stage to
   `data/prepare_data.py` (or a post-pack step in the shim) that
   concatenates `data/shards/shard_*.bin` in manifest order into
   `data_cache/tokens.bin`, honoring `data_cache_dir`/`data_cache_filename`
   from `config.py:get_config`. ~30 lines. Then update the three docs that
   currently *claim* this already happens (training.md:899,
   data-reference.md:9,231,836, quickstart.md:45,96) to describe the
   explicit concat step.
2. **Fix `benchmark_data.py`.** Delete the `eos_id=0` kwarg
   (`PackedDataset(data, seq_len=seq_len)`) and keep the BOS..EOS buffer
   (it is already position-only data — the eos ids are just tokens).
   Re-run `python benchmark_data.py --steps 3 …` to prove it works, and
   add the invocation to the docs' verified examples.
3. **Add a regression test** (P0 follows from the docs contract: docs
   ship with code; the two broken commands are documented commands).

### P1 — excise stale claims about top-level docs (self-heal the corpus)

The docs contain three "AGENTS.md/SKILLS.md/README.md is stale" passages
that are now themselves stale because those files were fixed in the same
uncommitted pass:

- `docs/training.md` "Why 7 techniques … 8 rows" → rewrite: AGENTS.md now
  lists the same 8 rows; drop the `channels_last`/Fused-AdamW framing
  entirely.
- `docs/concepts/data-and-kernels.md:795` and
  `docs/references/data-reference.md:1041` → delete the "(none yet)"
  claims; AGENTS.md's sanctioned list is current.
- `docs/concepts/data-and-kernels.md:162,807` → fix the citation
  (root `CoreProjects/README.md` §4 LLaMA-3-Lite entry, not "workspace
  README §5"; recipe is the six-source one, not five) and drop the
  SKILLS.md claim (Skill 3 already points at mixture.yaml).

### P2 — numeric honesty pass

- `docs/references/model-reference.md` memory table: rename rows
  "Weights (FP32) 2.06 GB", "EMA shadow (FP32) 2.06 GB" (keep BF16 rows
  only as an explicit "design profile" note); delete `[measured]` tags on
  92 GB / 23 GB / 20 GB (nothing measured yet — `.benchmarks/` empty).
- Cross-fix `use_z_loss`: pick one story. Either (a) code: make
  `use_z_loss=False` actually zero the z term in
  `train.py:train_model`/`validate` (2-line change, makes the flag
  load-bearing), or (b) docs: delete the two "disable with
  `use_z_loss=False`" implications in `docs/concepts/training-and-memory.md`.
  Recommend (a) — a documented config key that silently does nothing is
  a trap.

### P3 — gaps worth filling (new doc content)

1. **`benchmark_data.py` reference section** — the docs cite it but no
   doc walks it line-by-line (only a paragraph in training-and-memory and
   a quickstart one-liner).
2. **Workspace pipeline reference** — `LLM/shared_data/` stages
   (download→clean→tokenize→pack), manifest schema, shard layout, and the
   data-root resolution rule (`LLM_DATA_ROOT` > `$PWD/data`) are
   documented only obliquely; a `docs/references/workspace-data.md` would
   make the C2 wiring obvious.
3. **EMA internals** — `AveragedModel` + `get_ema_multi_avg_fn` mechanics
   are spread across three docs; one canonical section (what
   `ema.module` is, why val uses EMA, the 2.06 GB cost) would remove
   repeated C7-style drift.

### P4 — hygiene

- Root `CoreProjects/README.md` §4 LLaMA-3-Lite entry: fix
  `channels_last`/fused-AdamW/`GradScaler`/architecture.md-1234/16GB-cache
  (sibling project, but it is the portfolio's front door and the docs
  point at it).
- The `.gitignore`d `.vscode/settings.json` is tracked; drop it if
  desired.
- Consider a `scripts/microbench_a100.py` so AGENTS.md rule 2 becomes
  enforceable (currently "enforced by rule, not measurement" — stated
  honestly in data-and-kernels.md:449).

---

## 5. Acceptance criteria for "audit complete"

- [x] Corpus inventoried (16 docs, 119,475 words) and gate-verified
  (16/16 docs, all citations resolve, no line anchors).
- [x] Every public symbol covered (AST scan: 0 gaps across 10 modules).
- [x] Numeric claims re-derived from config (513.8M, 50.3 GB, 65.5 MB,
  6.44 GB, 8.23 GB model state).
- [x] Internal links audited (0 dead links incl. #fragments).
- [x] Suite run: 70 passed / 1 GPU-skipped.
- [x] Broken documented commands reproduced (`benchmark_data.py` crash;
  `prepare_data.py` never emitting `tokens.bin`).
- [x] Stale-claims-about-stale-files identified (C3/C4/C5/C11) with the
  exact current text verified.
- [x] (Follow-up) P0–P2 applied with tests; docs re-verified; vault sync. Applied 2026-08-05: C1 fixed (`benchmark_data.py` runs, regression-tested in `tests/test_data_pipeline.py`), C2 wired (`data/prepare_data.py:concat_shards_to_cache` produces `tokens.bin`, verified end-to-end through `build_training_data`), C3/C4/C5/C11/C6/C7/C8/C9/C12 resolved in the docs, `use_z_loss` made load-bearing, `scripts/microbench_a100.py` added. Suite: 76 passed / 1 GPU-skipped; doc gate green; vault mirrored.
