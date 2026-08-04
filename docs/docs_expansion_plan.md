# LLaMA-3-Lite — Documentation Expansion Plan

> **Status:** EXECUTED (2026-08-04).
> **Goal:** Take the current ~28.6K-word doc set and expand it into a
> comprehensive, from-scratch, concept-building documentation system that
> explains every topic in the codebase with full theory, and is *strictly
> aligned* with the code it documents.
> **Owner:** Atandra Bharati. **Scope:** all `.md` under the repo + the
> mandatory Obsidian vault mirror.

---

## Table of Contents

1. Executive Summary
2. Current-State Audit (verified against the codebase)
3. Guiding Principles
4. Target Documentation Architecture
5. The Expansion, Doc by Doc
6. Alignment & Verification Machinery
7. Phased Execution Plan
8. Acceptance Criteria & Metrics

---

## 1. Executive Summary

The repo has a solid documentation core — `docs/rope.md` is an excellent
example of the target style — but the set as a whole has four systemic
problems:

1. **Codebase misalignment.** `docs/data_prep.md`, `docs/tokenizer.md` and
   `data/DATA_PIPELINE.md` describe functions and files that do not exist in
   this repo (they describe the workspace-level `LLM/shared_data` pipeline,
   attributed to a `dataset.py` that is now a 32-line re-export shim).
   `docs/README.md` claims `architecture.md` is 1,234 lines (it is 509).
   Line-number anchors throughout are stale after refactors.
2. **Thin theory coverage.** Half the codebase has no theory doc at all:
   z-loss, QK-norm, EMA, AdamW + cosine schedule, BF16/TF32 numerics, Flash
   Attention 2, gradient checkpointing, `torch.compile`/CUDA graphs, the
   Triton kernels, the memory-stack derivation, the config surface, the test
   suite.
3. **Duplication.** `architecture.md` and `docs/model_architecture.md` are
   two near-identical model.py walkthroughs.
4. **No navigation.** No reading paths, no glossary, no FAQ, no
   audience targeting, no cross-document linking discipline.

The plan: a 3-track doc tree (theory / reference / guides) built on a
from-scratch concept order per topic, each doc symbol-anchored to the code
(`file.py:Class.method`, never line numbers), with an automated
doc↔code reference checker so "strictly aligned" is enforceable, not
aspirational. Target scale: ~28.6K → ~130K words.

---

## 2. Current-State Audit (verified)

> **Historical record (pre-expansion baseline).** This section inventories
> the documentation as it stood on 2026-08-03, *before* the expansion
> executed on 2026-08-04. Most of the files it lists no longer exist in the
> working tree: `architecture.md` and `docs/{data_prep,memory_stack,
> model_architecture,rope,tokenizer,training}.md` were replaced by the
> 3-track tree under `docs/{theory,reference,guides}/` (see §4 and
> `docs/README.md`). It is kept intact as the record of what the audit
> found, not as a description of the current tree.

All counts verified 2026-08-03 against the working tree.

### 2.1 Inventory (word counts)

| Doc | Words | Depth grade | Notes |
|-----|-------|-------------|-------|
| `README.md` | 2,852 | B | Good public overview; config tables stale (`target_tokens` fixed in defect pass) |
| `architecture.md` | 2,337 | B | Model walkthrough; **duplicates `docs/model_architecture.md`** |
| `docs/README.md` | 241 | C | Claims architecture.md = 1,234 lines (actual: 509) |
| `docs/model_architecture.md` | 3,152 | B | Model walkthrough; stale line anchors |
| `docs/training.md` | 1,430 | C+ | Good mechanics; theory thin; line anchors stale |
| `docs/memory_stack.md` | 683 | C+ | **Asserts** the 92→20 GB numbers; never derives them |
| `docs/rope.md` | 5,436 | **A** | The gold standard: 60-sec summary → intuition → proof → impl → pitfalls |
| `docs/tokenizer.md` | 4,255 | C | **References functions absent from the repo** |
| `docs/data_prep.md` | 5,857 | C | **References functions absent from the repo** |
| `data/DATA_PIPELINE.md` | 555 | D | Claims vendored copy = 24 files / ~160 KB; actual = 2 files |
| `SKILLS.md` / `AGENTS.md` | 1,833 | B | Operational; fixed in defect pass |
| **Total** | **~28.6K** | | |

### 2.2 Verified alignment defects (the critical issue)

| # | Doc | Claim | Reality |
|---|-----|-------|---------|
| 1 | `docs/data_prep.md`, `docs/tokenizer.md` | Describe `dataset.py` functions `_stream_to_disk`, `_doc_hash`, `_build_source_streams`, `interleave_datasets`, `_align_split_to_docs_and_chunks` with line refs (`dataset.py:74`, `dataset.py:250-266`, …) | These symbols exist **nowhere in this repo** (verified `grep` = 0 hits in `data/shared_data/loader.py` and `dataset.py`). They belong to the workspace `LLM/shared_data` pipeline. `dataset.py` is now a 32-line re-export shim. |
| 2 | `data/DATA_PIPELINE.md` | "`data/shared_data/` is a verbatim copy… ~160 KB · 24 source files"; shim "resolves it first… falls back to the workspace copy" | Vendored package contains exactly 2 files (`__init__.py`, `loader.py`); `prepare_data.py` resolves the **workspace** copy first. |
| 3 | `docs/README.md` | `architecture.md` is "1,234 lines" | 509 lines. |
| 4 | all walkthroughs | Line anchors (`model.py L15–32`, `train.py L114–154`) | Stale after ponytail refactors + the chunked-head defect fix (e.g. `build_transformer` now ~L360+, not L270). |
| 5 | `docs/training.md` §7 | Describes loop internals | Partially stale after the head-chunked loss change (validate/warmup/loop now use `return_hidden=True` + `chunked_head_cross_entropy_with_z`). |

### 2.3 Topics with **no** documentation at all

- **Z-loss** (PaLM/Gemma2): theory, gradient, why `z_loss_weight=1e-4`, masking semantics (now implemented: masked z-loss).
- **QK-norm**: why per-head RMSNorm before RoPE, placement rationale, interaction with z-loss.
- **EMA**: why EMA for val/generation, decay selection (0.999 @ 42k steps), shadow mechanics (`AveragedModel` + `get_ema_multi_avg_fn`), warmup interplay, checkpoint restore.
- **AdamW theory**: decoupled weight decay (why 2D+ params only), β₁=0.9/β₂=0.95 rationale, FP32 moments in BF16 training.
- **Cosine schedule + warmup theory**: why warmup, `start_factor = min_lr/peak_lr`, why cosine → min_lr floor, the `SequentialLR` construction.
- **Mixed precision**: FP32/BF16/FP16/TF32 numeric ranges, why no `GradScaler`, `autocast` scoping rules, `torch.set_float32_matmul_precision('high')`.
- **Flash Attention 2 / SDPA**: O(S) memory vs O(S²), kernel fusion, `is_causal`, why `mask` param was removed, backend selection.
- **Gradient checkpointing**: activation memory math `O(L·B·S·d)`, recompute cost, `use_reentrant=False` semantics, interplay with `torch.compile` + CUDA graphs.
- **`torch.compile` / CUDA graphs / `reduce-overhead`**: graph capture, static-shape requirement, stream ownership (why only `non_blocking` H2D), warmup necessity.
- **The Triton kernels** (`kernels/`): `model_architecture.md` §13 is 3 one-line bullets. No grid/block design, online softmax, `atomic_add`, autograd.Function pattern, backward re-compute, `_MAX_VOCAB_BLOCK` constraint.
- **The memory stack derivation**: the 92→20 GB table is asserted; no per-tensor accounting.
- **Config surface**: no doc explains every key + interactions + a worked budget.
- **Test suite**: 59 test functions across 4 files + `conftest.py` + the e2e GPU script — completely undocumented (fixtures, markers `gpu`/`numeric`/`smoke`, CI).
- **Scaling/token math**: why 42,000 steps × batch 96 × seq 2048 = 8.26B tokens; Chinchilla context; expected loss curves; validation methodology.
- **Glossary, learning paths, troubleshooting/FAQ.**

### 2.4 What is genuinely good (keep the style)

- `docs/rope.md` structure: 60-second summary → why → intuition → math with
  proofs → implementation → shape trace → edge cases → pitfalls → references.
- Mermaid usage (flowcharts, sequence diagrams, tensor-shape traces).
- `docs/README.md` as an index (needs fixing, but the pattern is right).
- Code-keyed walkthroughs ("Implementation → Detailed Code Walkthrough").
- AGENTS.md hard rules that *require* rationale to live in docs rather than
  comments — the docs are the designated home for theory; the expansion
  should honor that division of labor.

---

## 3. Guiding Principles

1. **From scratch, in concept order.** Every topic starts with "why does
   this exist?" → intuition → math/proof → the specific implementation →
   edge cases. No doc assumes knowledge of another doc's topic; each is
   self-contained but cross-linked.
2. **Symbol-anchored, never line-anchored.** Citations use
   `model.py:GroupedQueryAttention.forward` or `train.py:validate`. Line
   numbers are banned (they rot).
3. **Strict code alignment is enforced by machine.** A reference checker
   (see §6) fails CI if a documented symbol doesn't exist. "Strictly
   aligned" is a test, not a promise.
4. **Snippets are executable or marked.** Code blocks in docs must be
   runnable as written, or explicitly marked `# illustrative` (pseudo).
5. **One fact, one home.** No duplicated walkthroughs. Consolidate
   `architecture.md` + `docs/model_architecture.md` into one model
   reference; theory lives in `docs/theory/`, implementation walkthroughs in
   `docs/reference/`.
6. **Every number is derived.** Memory, throughput, param budgets, token
   counts: show the arithmetic; mark measured vs estimated; never assert a
   headline without its derivation (the 78% claim must be reproducible from
   the memory doc alone).
7. **Audience-aware.** Each doc declares its audience
   (beginner/intermediate/expert) and is reachable from a learning path.
8. **Mandatory vault mirror.** Every new/modified `.md` is synced via
   `bash scripts/sync_to_vault.sh` per workspace rule.

---

## 4. Target Documentation Architecture

```
LLaMA-3-Lite/
├── README.md                      (public overview; links into docs/)
├── AGENTS.md / SKILLS.md          (operational; unchanged except cross-links)
├── docs/
│   ├── README.md                  (index + learning paths; fixed claims)
│   ├── docs_expansion_plan.md     (this file)
│   ├── CODE_MAP.md                (symbol ↔ doc ↔ test table; verified)
│   ├── theory/                    (from-scratch concept building)
│   │   ├── transformers-from-scratch.md
│   │   ├── attention.md
│   │   ├── positional-encoding.md
│   │   ├── normalization.md
│   │   ├── feedforward.md
│   │   ├── loss-functions.md
│   │   ├── optimization.md
│   │   ├── mixed-precision.md
│   │   ├── gradient-checkpointing.md
│   │   ├── memory-engineering.md
│   │   ├── kernel-programming.md
│   │   ├── data-engineering.md
│   │   ├── reproducibility.md
│   │   └── scaling-and-metrics.md
│   ├── reference/                 (code-keyed walkthroughs)
│   │   ├── model.md               (rework: absorbs architecture.md + model_architecture.md)
│   │   ├── training.md            (rework)
│   │   ├── data.md                (rework: aligned to the REAL vendored loader + prepare_data shim)
│   │   ├── tokenizer.md           (rework: aligned to loader.build_tokenizer)
│   │   ├── rope.md                (keep: implementation deep-dive; theory links to positional-encoding.md)
│   │   ├── memory-stack.md        (rework: full derivation)
│   │   ├── kernels.md             (new: the 3 Triton kernels)
│   │   ├── config.md              (new)
│   │   └── tests.md               (new)
│   └── guides/
│       ├── learning-paths.md
│       ├── quickstart.md
│       ├── troubleshooting.md
│       └── glossary.md
└── data/DATA_PIPELINE.md          (rewrite: honest 2-file vendored reality)
```

Rationale for the tree: theory is code-agnostic-ish and reusable; reference
is code-keyed and changes with the code; guides are navigational. The split
makes the "concept building" track cleanly separable from the "what does
this file do" track.

---

## 5. The Expansion, Doc by Doc

### 5.1 New theory docs (`docs/theory/`) — the core of the expansion

Each doc follows the template: *Overview → Why it exists → Intuition →
Formal treatment (math/proofs) → Concrete numbers at this project's scale →
How the code realizes it (symbol anchors) → Edge cases & pitfalls → Further
reading (links)*. Target 4–9K words each.

#### T1. `transformers-from-scratch.md` (beginner → intermediate)
- The language-modeling task: next-token prediction; why decoder-only.
- Sequence-to-sequence origin (encoder-decoder) and why LLaMA drops the encoder.
- The residual stream view of a transformer block: `x = x + block(x)`.
- Pre-norm vs post-norm (why LLaMA pre-norms; RMSNorm placement).
- Tokenization→embedding→stack of blocks→LM head→loss: the full data flow with shapes at this scale (`[96, 2048, 1024]`).
- The 513.8M-param anatomy: embeddings vs non-embedding (251.7M), what the LM head costs (128K×1024).
- **Anchors:** `model.py:Transformer.forward`, `build_transformer`, `model.py:DecoderBlock`.

#### T2. `attention.md` (beginner → intermediate)
- Scaled dot-product attention from first principles: why `softmax(QKᵀ/√d_k)`, why √d_k (variance argument, proof).
- Causal masking: the `is_causal=True` flag, why future info must not leak; the causality test that exists (`tests/test_model.py::test_causality`).
- Multi-head attention: why heads, the projection anatomy (`q/k/v/out_proj`), head_dim=128, 8 heads.
- GQA: KV sharing math (2× smaller KV cache/params), `n_rep=2`, the eager expansion in code vs FA2.
- Flash Attention 2 / SDPA: O(S) memory, tiling, the backend dispatch, why the code's `mask` param was removed.
- Complexity: FLOPs of attention at this scale.
- **Anchors:** `model.py:GroupedQueryAttention.forward`, `F.scaled_dot_product_attention`, `tests/test_model.py`.

#### T3. `positional-encoding.md` (beginner → intermediate)
- Why positions at all; permutation-invariance of attention without them.
- The three families: absolute (sinusoidal/learned), relative (bias/T5), rotary (RoPE).
- RoPE math in the general case; the relative-position payoff (inner product depends on i−j only); proof sketch.
- θ=500K: what the frequency schedule controls; long-context extrapolation; the AGENTS.md hard rule.
- NTK/YaRN extension landscape (links to `docs/reference/rope.md` for implementation).
- **Anchors:** `model.py:RoPE`; cross-link `docs/reference/rope.md`.

#### T4. `normalization.md` (intermediate)
- Why normalization in deep nets; covariate-shift intuition; LayerNorm math.
- RMSNorm: dropping the mean — math, why it works, the `eps` term, scale/gain parameter.
- Pre-norm residual placement; QK-norm (see T4b below or fold in).
- **Anchors:** `model.py:RMSNorm.forward`, `model.py:DecoderBlock`.

#### T5. `feedforward.md` (intermediate)
- The FFN block: two matmuls + nonlinearity; why 4× d_model.
- SwiGLU: gate/up/down, why gated variants (PaLM/Gemma), the fused `gate_up_proj` (2·d_ff wide), FLOP comparison vs plain ReLU FFN.
- **Anchors:** `model.py:SwiGLUFFN.forward`, `tests/test_model.py::test_fused_equals_unfused_reference`.

#### T6. `loss-functions.md` (intermediate)
- CE for LM: the `[N, V]` logits → target shift-by-1; `ignore_index` semantics; why the training path now uses `ignore_index=-100` (no padding; EOS must stay learnable).
- Chunked CE: why `[N,V]` doesn't fit (50 GB BF16 / 100 GB FP32); the chunking trick; **proof that chunked CE ≡ dense CE** (per-chunk reduction over disjoint index sets); the checkpoint-per-chunk design and its memory bound (`chunk_size=256` → 131 MB FP32 slice); `ce_chunk_size` knob.
- Z-loss: log-partition growth problem late in training (PaLM/Gemma2); `L_z = mean((log Σ exp z)²)`; the gradient `dL_z/dz = 2·log_z·softmax(z)`; why the code masks ignored tokens; `z_loss_weight=1e-4`.
- The Triton CE variant: online softmax, `atomic_add` accumulators, why mean-of-chunk-means for the triton path is exact only for equal chunks.
- **Anchors:** `model.py:chunked_head_cross_entropy_with_z`, `chunked_cross_entropy_with_z`, `kernels/cross_entropy_triton.py`, `tests/test_model.py::TestChunkedCrossEntropyWithZ`, `TestChunkedHeadCrossEntropyWithZ`.

#### T7. `optimization.md` (intermediate)
- AdamW: the update rule with math; decoupled weight decay vs L2; why the code decays 2D+ params only (`param.dim() >= 2`); why β₂=0.95 (faster-moving second moment for LLMs); FP32 master moments.
- Gradient clipping: `max_grad_norm=1.0`; global norm semantics.
- Warmup + cosine: why warmup (early-Adam variance), the `start_factor = min_lr/peak_lr` trick, `LinearLR → CosineAnnealingLR → SequentialLR` construction; the 3e-4 → 3e-5 curve.
- LR/batch relationship at 196,608 tokens/step.
- **Anchors:** `train.py` optimizer/scheduler construction, `tests/test_train.py::make_tiny_scheduler`.

#### T8. `mixed-precision.md` (intermediate)
- IEEE-754 recap: FP32/FP16/BF16/TF32 exponent+mantissa; BF16's 8-bit exponent (why no underflow → no GradScaler); TF32's 10-bit mantissa for matmuls.
- `torch.autocast` scoping rules: which ops are downcast (matmul/linear/conv), which stay FP32 (loss/norms), the `enabled=device.type=='cuda'` guard.
- The FP32 loss chain (upcast per chunk), `torch.set_float32_matmul_precision('high')`, `allow_tf32` toggles, and the weight-memory math (513.8M × 2 B = 1.03 GB).
- **Anchors:** `train.py:setup_gpu_optimizations`, `train.py` autocast blocks, `model.py:chunked_head_cross_entropy_with_z` `.float()`.

#### T9. `gradient-checkpointing.md` (intermediate)
- Activation memory math: `O(L·B·S·d)` per layer type; derive the ~70 GB figure at B=96/S=2048.
- The tradeoff: recompute ≈ 1 extra forward; the `checkpoint(layer, x, use_reentrant=False)` call; why `use_reentrant=False` (no double-backward weirdness, compile-friendly).
- Interaction with the head-chunked loss (checkpoint per chunk) and with CUDA graphs (static shapes).
- **Anchors:** `model.py:Transformer.forward`, `model.py:chunked_head_cross_entropy_with_z`, `docs/reference/memory-stack.md`.

#### T10. `memory-engineering.md` (intermediate → expert) — *flagship doc*
- The full derivation of 92 GB → 20 GB, per component, with arithmetic:
  - Model state: weights BF16 (1.03 GB) + AdamW moments FP32 (2×513.8M×4 = 4.11 GB) + grads BF16 (1.03 GB).
  - Activations with grad-ckpt (derive ~3.2 GB from per-layer tensor sizes).
  - Logits: full vs chunked head (50.3 GB → 0.4 GB hidden + 0.13 GB chunk).
  - KV/attention via FA2 (O(S) instead of O(S²): derive).
  - Data: memmap → ~1 MB resident (page-fault argument).
  - CUDA caching allocator, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- A worked end-to-end peak estimate at B=96 and at B=48/16 (sizing guide numbers).
- What is measured vs estimated (mark honestly; `.benchmarks/` currently empty).
- **Anchors:** `config.py` memory-related keys, `train.py:setup_gpu_optimizations`, `data/shared_data/loader.py:PackedDataset`, `docs/reference/memory-stack.md`.

#### T11. `kernel-programming.md` (expert)
- Triton model of computation: grid/program_id, `tl.arange` blocks, masks, `tl.constexpr`, num_warps/stages.
- Pattern 1 — `rmsnorm_triton.py`: row-wise reduce, `next_power_of_2`, the `_MAX_BLOCK_SIZE=8192` guard.
- Pattern 2 — `swiglu_triton.py`: elementwise fuse, why it saves launches.
- Pattern 3 — `cross_entropy_triton.py`: online softmax (the m/l running-max trick), `atomic_add` for CE_SUM/CE_CNT/Z_SUM, `_MAX_VOCAB_BLOCK=131072` (vocab axis fits one program), the autograd.Function wrapper + backward re-compute.
- The autograd.Function contract: `ctx.save_for_backward`, custom backward, memory implications.
- The opt-in gating: `ENABLE_TRITON_KERNELS=1` + per-kernel `*_impl` keys, AGENTS.md rule 7.
- **Anchors:** `kernels/*.py`, `tests/test_model.py` triton-fallback tests, `model.py` RMSNorm/SwiGLU/CE dispatch.

#### T12. `data-engineering.md` (intermediate) — *rewrite of data_prep.md, aligned*
- The **actual** project data path (verified): `data/shared_data/loader.py` (vendored, 2 files) + `data/prepare_data.py` shim → workspace `LLM/shared_data` universal pipeline.
- The 8.0B-token mixture (table from `LLM/shared_data/config`), Chinchilla context (~400–515M params).
- Document packing: why EOS separators (AGENTS.md rule 6), the shift-by-1 window (`seq_len+1` chunks), cross-document attention cost.
- Dedup (SHA-256 over first 256 tokens), min/max doc length rules.
- Streaming/shuffling theory; `ShuffledRangeSampler` determinism + `set_epoch` + the new epoch-wrap in `train.py:_next_batch`.
- The memmap layout: uint32, 32 GB at 8B tokens, OS page-fault residency.
- **Anchors:** `data/shared_data/loader.py:*`, `data/prepare_data.py`, `train.py:_next_batch`, `docs/reference/data.md`.

#### T13. `reproducibility.md` (intermediate)
- RNG state theory: torch/numpy/python/CUDA generators; why full-state restore gives bit-identical resumes.
- The checkpoint round-trip (`save_checkpoint`/`load_checkpoint`): what's stored, the cross-device RNG move fix (`rng_torch.cpu().to(torch.uint8)`), the EMA shadow.
- `ShuffledRangeSampler` seed+offset determinism; why `set_epoch` keeps permutation reproducibility.
- **Anchors:** `train.py:save_checkpoint`, `load_checkpoint`, `data/shared_data/loader.py:ShuffledRangeSampler`, `tests/test_train.py::TestCheckpointRoundTrip`.

#### T14. `scaling-and-metrics.md` (intermediate)
- Token math: 42,000 × 96 × 2048 = 8.26B; why 8B corpus + epoch wrap (the defect-fix rationale).
- Chinchilla-optimal context; param/token ratios.
- Expected loss/perplexity trajectory (log-log power law shape); how validation works (`val_interval=2000`, `val_max_batches=100`, EMA validation); why perplexity = `exp(loss)`.
- W&B metrics reference: every logged key and what it means.
- **Anchors:** `config.py`, `train.py:validate`, wandb log dicts.

### 5.2 Reference reworks (`docs/reference/`)

#### R1. `model.md` — consolidate `architecture.md` + `docs/model_architecture.md`
- One file: block-by-block walkthrough of `model.py` with symbol anchors, the tensor-shape trace at `[96, 2048]`, and the parameter budget (513.8M total / 251.7M non-embedding).
- Replace all `L15–32`-style anchors with `model.py:Class.method`.
- Keep the Triton integration as a pointer to `kernels.md`.
- Retire both old files (delete + update `docs/README.md`, README, AGENTS references).

#### R2. `training.md` — rework
- Fix §7 to the current loop: `return_hidden=True` + `chunked_head_cross_entropy_with_z`, `_next_batch` epoch wrap, `_head_weight` wrapper resolution, `async_checkpoint` wiring + join, `grad_norm` persistence.
- Add theory pointers to `optimization.md`, `mixed-precision.md`, `gradient-checkpointing.md`, `reproducibility.md` (no duplication).
- Keep the sampling/generation walkthrough (top-k/top-p, temperature).

#### R3. `data.md` — rework `data_prep.md`, aligned to the real code
- Walk the vendored loader line by line: `PackedDataset`, `ShuffledRangeSampler`, `collate_fn`, `build_synthetic_data`, `build_tokenizer`, `build_training_data`, `_SyntheticTokenizerStub`.
- The `prepare_data.py` shim: what it delegates to, the workspace-path resolution, the `ModuleNotFoundError` guidance.
- Data-preparation *theory* moves to `data-engineering.md`; this file is the code tour.

#### R4. `tokenizer.md` — rework
- Fix every `dataset.py:N` anchor to `data/shared_data/loader.py:build_tokenizer` / `_SyntheticTokenizerStub`.
- Keep the BPE theory, special-token table (EOS 128009, PAD 128002), byte-fallback, `len(tokenizer)` → `real_vocab_size` logic (`train.py`).
- Correct the "streaming tokenization in dataset.py" section — the real pipeline tokenizes in the workspace pipeline; describe that path honestly.

#### R5. `rope.md` — keep, lightly rework
- Already the gold standard. Retitle/retarget as the *implementation* deep dive; theory landscape moves to `positional-encoding.md`; add cross-links both ways. Verify anchors against current `model.py:RoPE`.

#### R6. `memory-stack.md` — rework into the summary of `memory-engineering.md`
- Keep the 8-row stack table (fix the numbering inconsistency: table says 8 techniques, AGENTS says 7); each row links to its theory doc; every number links to a derivation section in `memory-engineering.md`.

#### R7. `kernels.md` — new reference counterpart to `kernel-programming.md`
- Per-kernel: signature, grid config, launch params, the autograd.Function wrapper, fallback semantics, microbenchmark instructions (`scripts/microbench_a100.py`), the ≥1.5× speedup rule (AGENTS rule 2).

#### R8. `config.md` — new
- Every key in `config.get_config()` grouped by concern, with type, default, why, interactions, and a worked memory budget using the values. Flag keys that are informational vs load-bearing (`rope_theta`, `ce_chunk_size`, `tie_embeddings` absence, `qknorm`, `use_z_loss`, `use_ema`).
- Cross-link to `tests/test_config.py` (the REQUIRED_KEYS contract).

#### R9. `tests.md` — new
- Test strategy: unit (per-module) / equivalence (chunked ≡ dense) / smoke (end-to-end) / e2e GPU script.
- Fixture reference: `tiny_config`, `tiny_model`, `device`/`dtype` (FP32 on CPU for exactness), `weights_dir`, `make_token_stream`.
- Markers: `gpu`, `numeric`, `smoke`; `pytest.ini` config; the conftest wandb stub.
- How to run: CPU suite, GPU suite (`--run-gpu`), the e2e script, CI workflow.
- Anchors: every test class ↔ the contract it defends.

### 5.3 Guides

#### G1. `learning-paths.md`
- Beginner path: quickstart → transformers-from-scratch → attention → normalization → feedforward → loss-functions → model.md.
- Intermediate path: + positional-encoding, optimization, mixed-precision, gradient-checkpointing, data-engineering, training.md.
- Expert path: + memory-engineering, kernel-programming, kernels.md, tests.md, troubleshooting.

#### G2. `quickstart.md`
- What `python train.py` actually does now (synthetic fallback warning), how to build real data (`python data/prepare_data.py` + workspace prerequisite), resume, W&B offline mode, and the GPU smoke script.

#### G3. `troubleshooting.md`
- OOM at batch 96 (memory-engineering links, `ce_chunk_size` knob, grad-ckpt), CUDA-graph capture stalls, triton import failures (Mac/CPU), `len(tokenizer)`/cache-missing errors, checkpoint corruption, `StopIteration`/epoch wrap, wandb offline.

#### G4. `glossary.md`
- Notation (`N`, `V`, `d`, `S`, `B`, `n_kv`), every acronym, tensor-shape conventions, config-key glossary.

### 5.4 Root & meta files

- `docs/README.md` — fix the 1,234-line claim; become the index + learning-path entry.
- `docs/CODE_MAP.md` — the verified symbol↔doc↔test table (§6).
- `data/DATA_PIPELINE.md` — rewrite to the true 2-file vendored reality; document the `rsync` refresh command (already present) but correct the "verbatim copy" claim.
- `README.md` — update the docs links + quickstart (mention synthetic fallback), point to learning paths.
- `AGENTS.md` — update the `architecture.md` reference (line count), add a hard rule: "doc changes ship with code changes; `docs/CODE_MAP.md` and the reference checker must stay green."

---

## 6. Alignment & Verification Machinery

"Strictly aligned with the codebase" is enforced, not hoped for.

### 6.1 Symbol-anchor convention
- Docs cite `file.py:Class.method` or `file.py:function` — resolvable by a script.
- Line-number anchors are banned (search-replace pass removes existing ones).

### 6.2 Reference checker (new)
- `tests/test_doc_refs.py` (runs in CI): parses every `` `*.py:*` `` citation in
  all `.md` files, imports the module, and asserts the symbol exists
  (`hasattr` chain). Any stale citation fails the suite.
- Also asserts every documented code block either executes or is marked
  `# illustrative` (a light static check; full doctest execution is a
  phase-7 optional).
- Enforcement: `pytest tests/test_doc_refs.py` in `.github/workflows/ci.yml`
  alongside the existing import + smoke jobs.

### 6.3 `docs/CODE_MAP.md`
- Table: `Module → Symbol → Doc(s) → Test(s)`. Regenerated/verified by
  `scripts/check_doc_refs.py` (or by the pytest itself emitting the diff).
- This is the single source of truth for "where is X documented".

### 6.4 Snippet policy
- Every code block is one of: (a) runnable as written (tested by the
  checker where cheap), (b) marked `# illustrative` pseudo-code, or
  (c) an output/table. No unmarked pseudo-code.

### 6.5 Sync-on-change rule
- Any code change that alters a documented symbol updates the doc + CODE_MAP
  in the same commit (AGENTS.md hard rule addition).

---

## 7. Phased Execution Plan

Each phase lands independently and leaves the suite green.

| Phase | Deliverables | Exit criteria |
|-------|--------------|---------------|
| **0. Alignment retrofit** | Fix `docs/README.md` claims, `data/DATA_PIPELINE.md`, remove line anchors repo-wide, add `tests/test_doc_refs.py` + first `CODE_MAP.md`, correct `data_prep.md`/`tokenizer.md` attribution or quarantine them | checker green; no line-number anchors remain; suite green |
| **1. Theory foundation** | T1 transformers-from-scratch, T2 attention, T3 positional-encoding, T4 normalization, T5 feedforward | all drafted (4–9K words each), symbol-anchored, checker green |
| **2. Numerics & optimization** | T6 loss-functions, T7 optimization, T8 mixed-precision | full derivations incl. chunked-CE equivalence proof + z-loss gradient; checker green |
| **3. Memory & systems** | T9 gradient-checkpointing, T10 memory-engineering (flagship), T11 kernel-programming | 92→20 GB fully derived; kernel grid/block diagrams; checker green |
| **4. Data & training ops** | T12 data-engineering, T13 reproducibility, T14 scaling-and-metrics, R3 data.md, R4 tokenizer.md | data docs aligned to vendored loader (verified against loader.py); checker green |
| **5. Reference reworks** | R1 model.md (consolidate + delete duplicates), R2 training.md, R5 rope.md, R6 memory-stack.md, R7 kernels.md, R8 config.md, R9 tests.md | single model reference; all walkthroughs current vs working tree; checker green |
| **6. Guides & navigation** | G1–G4, docs/README.md index, README.md link updates, AGENTS.md rule addition | learning paths complete; every doc reachable from index |
| **7. Verification & polish** | Full snippet execution pass, cross-link audit, glossary completeness, `make doctor`-style doc lint, vault sync | checker green + suite green + vault mirror verified (`--dry-run`) |

Rough sequencing rationale: Phase 0 first because everything else depends on
the anchor/checker discipline; theory before reference so reference docs can
link instead of duplicate; guides last so they index a stable tree.

---

## 8. Acceptance Criteria & Metrics

- **Comprehensiveness:** every public symbol in `model.py`, `train.py`,
  `config.py`, `data/shared_data/loader.py`, `kernels/*.py`, and every test
  class in `tests/` appears in exactly one reference doc and ≥1 theory doc
  (enforced by CODE_MAP coverage = 100%).
- **Concept building:** each theory doc follows the template (why → intuition
  → math/proof → implementation → pitfalls) and contains ≥1 worked derivation
  or proof with real project numbers.
- **Alignment:** `tests/test_doc_refs.py` green in CI; zero line-number
  anchors; zero unmarked pseudo-code blocks.
- **Scale:** total docs ≥ 28 files, ≥ 120K words (from ~28.6K), with the
  theory track ≥ 60K words.
- **Navigation:** every doc reachable from `docs/README.md` via the learning
  paths; glossary covers every acronym used in any doc.
- **Vault:** `bash scripts/sync_to_vault.sh --dry-run` shows the new docs
  mirrored; no stale `.md` left unsynced.
- **No regressions:** full test suite green at every phase boundary;
  AGENTS.md headline metrics (78% / 92→20 GB) reproducible from
  `memory-engineering.md` arithmetic alone.
