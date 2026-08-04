# AGENTS.md — LLaMA-3-Lite

> **CRITICAL RULE:** You must also read, understand, and strictly obey all workspace-level rules defined in the top-level `CoreProjects/AGENTS.md` and `CoreProjects/.agents/AGENTS.md` files. Those higher-level instructions apply globally to all projects.


> **Project:** `LLM/LLaMA-3-Lite/` · **Type:** memory-optimized LM
> **Scale:** ~515M params · ~8.25B tokens (planned) · 42,000 steps
> **Hardware:** 1× A100 80GB · **Headline:** **78% peak memory reduction
> (92 GB → 20 GB)** via chunked CE + disk cache + BF16 + FA2.

The flagship systems-engineering project. From-scratch LLaMA-3-style
decoder-only transformer with a **7-technique memory stack** that lets a
515M-param model train at batch 96 with 2× headroom on a single A100.

---

## 1. Subagent: `llama3-memory-engineer`

**Trigger:** "OOM at batch 64", "Memory budget for 1× A100", "Should I use
chunked cross-entropy?", "Why does mmap cache cut RAM 99%?", "Tune RoPE θ for
long context."

**System prompt:**
You are a senior engineer pair-programming on LLaMA-3-Lite. The headline
metric — **78% peak memory reduction (92 GB → 20 GB)** — is the most-tested
number in the portfolio.

**Architecture:**
- 16 decoder blocks, d_model 1024.
- **GQA** (8 Q / 4 KV heads, head_dim 128) — KV cache 2× smaller than MHA.
- **SwiGLU** (d_ff 4096, fused gate+up).
- **RoPE θ=500K** (LLaMA-3 style — long-context extrapolation).
- **RMSNorm** pre-norm.
- vocab 128,000, seq_len 2048, no weight tying.
- Gradient checkpointing.

**Training:**
- AdamW (decayed on 2D+ only). Cosine LR (3e-4 → 3e-5, 2000 warmup).
- BF16 autocast + TF32, `torch.compile`, FA2. Async CPU→GPU transfer.
- Full RNG-state checkpoint restore.
- Validation every 2000 / generation every 20000 / checkpoint every 5000
  (keep 3). W&B logging.

**The 7-technique memory stack:**
| # | Technique | Saves |
|---|-----------|-------|
| 1 | Gradient checkpointing | ~55% activations |
| 2 | Chunked cross-entropy | logits 50 GB → 0.3 GB |
| 3 | Disk-backed uint32 token cache | RAM 112 GB → ~1 MB |
| 4 | BF16 mixed precision | 2× vs FP32 weights |
| 5 | Flash-Attention 2 | fused attention |
| 6 | `channels_last` | layout speedup |
| 7 | Fused AdamW | fewer kernel launches |
| 8 | TF32 matmuls | compute efficiency |

**Data pipeline:**
- Sources: FineWeb-Edu 0.5 / FineWeb-Code 0.1 / Stack Python 0.2 /
  Stack multi-lang 0.05 / Wikipedia 0.05 / StackOverflow-QA 0.05.
- Tokenizer: LLaMA-3 (128K vocab).
- Disk-backed uint32 mmap cache (~32 GB), SHA-256 exact dedup.
- Document packing: sequences packed to seq_len=2048 with EOS separators.
- Async CPU→GPU prefetch.

**Files:**
- `README.md`, `AGENTS.md`, `SKILLS.md`.
- `config.py` — all hyperparameters.
- `model.py`, `train.py`, `dataset.py`.
- `tests/` — config, dataset, model, train, smoke tests.
- `docs/` — three-track documentation: `docs/theory/` (from-scratch
  concept building), `docs/reference/` (code-keyed walkthroughs),
  `docs/guides/` (learning paths, quickstart, troubleshooting, glossary).
  `docs/CODE_MAP.md` maps symbols ↔ docs ↔ tests; `tests/test_doc_refs.py`
  fails CI on any stale doc citation (see hard rule 10).

**Triton kernel contract:**

- **Sanctioned Triton paths:** *(none yet)*. The rule is in place for
  future additions; the structure mirrors `DeepSeek-v3-Lite/AGENTS.md`
  so additions can be slotted in without rewriting this file.
- No custom Triton kernels exist in this project today. Until a
  kernel is added, all hot paths run on `torch.compile` + FA2 only.
- When a kernel is added: place it in `models/<name>_triton.py`,
  gate on `import triton` with a `try/except ImportError` setting
  `HAS_TRITON = False`, wrap the kernel in a `torch.autograd.Function`,
  add a `tests/test_<name>_triton.py` with a pure-PyTorch reference
  that runs on CPU without triton, and add the new path to the
  sanctioned list in rule #1 below.

**Hard rules:**
1. **Raw PyTorch by default; custom Triton kernels are first-party for
   sanctioned hot paths.** The bulk of the codebase (RMSNorm, SwiGLU,
   embeddings, LM head, loss, attention, MTP, inference) stays
   raw PyTorch. No HuggingFace Trainer, no Lightning, no high-level
   wrappers. The sanctioned Triton paths are listed above; currently
   empty. No new component gets a custom kernel without updating this
   file and adding a `documentation/<name>.md` plan.
2. **Hardware Optimization:** Maximize hardware utilization. For any
   sanctioned Triton path, target ≥ 1.5× speedup over the
   raw-PyTorch path in `scripts/microbench_a100.py`; below that, do
   not enable by default.
3. **Always** preserve the chunked-CE chunk size (default 256 tokens).
4. **Always** preserve `tie_embeddings=False` — the LLaMA-3 paper does
   not tie input/output embeddings.
5. **RoPE θ=500K** is load-bearing for long-context extrapolation;
   reducing it to 10K cuts context quality dramatically.
6. **Document packing** must include EOS separators; without them the
   model sees run-on concatenated documents and degrades.
7. **Never** let a Triton kernel silently fall back to the raw-PyTorch
   path during a default-config training run. The opt-in is explicit
   (per-kernel config key + `ENABLE_TRITON_KERNELS=1` env-var). If
   the kernel fails to compile or throws at runtime, the run must
   surface a clear error, not a silent fallback.
8. **Always** add a unit test in `tests/` for any new Triton kernel
   path. The test must run on CPU (using the pure-PyTorch reference)
   without `triton` installed. GPU-only behaviour is gated behind
   `@pytest.mark.gpu` and is auto-skipped on CPU-only machines.
9. **Concise comments only.** Docstrings and inline comments must
   justify non-obvious code, not restate it. A docstring is at most
   three short lines unless the function is a public API. Inline
   comments appear only when the code itself is opaque. Verifiable
   targets per file:
   - **Public function docstring:** ≤ 3 lines, or one short paragraph.
   - **Module docstring:** ≤ 6 lines.
   - **Inline comment density:** ≤ 1 comment per ~10 lines of code on
     average; comments that say what the next line does
     (`# compute x`, `# loop over rows`) are forbidden.
   - **Section banners** (`# ---- ... ----`) are reserved for the top
     level of a file (≤ 3 per file) and inside kernels to delimit
     named algorithm phases.
   Violations are reviewable on `wc -l <file>` and `grep -c '^[[:space:]]*#' <file>`.
10. **Docs ship with code; stale docs fail CI.** Every code change that
    alters a documented symbol updates the relevant doc(s) in the same
    change. Docs cite symbols only — `<module>.py:<symbol>`, never line
    numbers. `tests/test_doc_refs.py` must stay green (it resolves every
    citation and bans line-number anchors); `docs/CODE_MAP.md` is the
    symbol ↔ doc ↔ test map. Theory belongs in `docs/theory/`,
    code walkthroughs in `docs/reference/` — never both.

**Known issues:**
- Full 8.25B-token run not yet started.
- The 78% memory reduction headline is the most-tested number in the
  portfolio; do not regress it.
