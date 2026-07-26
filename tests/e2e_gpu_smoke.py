"""End-to-end GPU pipeline smoke test for LLaMA-3-Lite.

Run with the project's venv: ``~/.venv/bin/python tests/e2e_gpu_smoke.py``.
8 stages: env, data, model, train, chunked CE, validate, checkpoint,
Triton kernels. Tiny config fits a 4 GB GPU.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))  # vendored shared_data
os.chdir(ROOT)

import dataset as ds
from model import (
    build_transformer,
    chunked_cross_entropy_with_z,
)


def check_environment() -> torch.device:
    print("=" * 70)
    print("[1/8] Environment check")
    print("=" * 70)
    print(f"  torch       : {torch.__version__}")
    print(f"  cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  device      : {name} (cc {cap[0]}.{cap[1]}, {total:.1f} GB)")
    try:
        import triton  # noqa: F401
        print(f"  triton      : {triton.__version__}")
    except Exception as e:  # pragma: no cover
        print(f"  triton      : MISSING ({e})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> using device: {device}")
    if device.type == "cpu":
        print("  WARNING: no CUDA; smoke test will run on CPU. The 'GPU run' goal")
        print("  cannot be satisfied on this host.")
    return device


def build_tiny_config() -> dict:
    """CPU/GPU-friendly tiny config; small vocab keeps the CE buffer in 4 GB."""
    return {
        # Architecture
        "d_model": 128, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "head_dim": 32, "d_ff": 512, "vocab_size": 512, "seq_len": 64,
        "rope_theta": 500_000.0, "rms_norm_eps": 1e-5,
        # Training
        "batch_size": 4, "gradient_accumulation": 1, "max_steps": 8,
        "learning_rate": 3e-4, "min_lr": 3e-5, "warmup_steps": 2,
        "weight_decay": 0.1, "max_grad_norm": 1.0,
        "optimizer": "AdamW", "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
        "compile_model": False, "gradient_checkpointing": False,
        "use_chunked_cross_entropy": True,
        "tf32": device_supports_tf32(), "cudnn_benchmark": False,
        "cuda_alloc_conf": "expandable_segments:True",
        "rmsnorm_impl": "pytorch", "swiglu_impl": "pytorch",
        "cross_entropy_impl": "pytorch",
        # Data
        "data_sources": {}, "num_workers": 0, "prefetch_factor": 2,
        "pin_memory": True, "target_tokens": 8192,
        "data_cache_dir": "data_cache_e2e", "data_cache_filename": "tokens_e2e.bin",
        "reuse_data_cache": False, "shuffle_documents": True, "shuffle_seed": 0,
        "dedup": False, "dedup_hash_bytes": 16,
        "min_doc_tokens": 4, "max_doc_tokens": 64,
        "tokenizer_name": "synthetic", "tokenizer_cache_dir": None,
        # Val / gen / ckpt
        "val_interval": 1000, "val_max_batches": 4, "val_split": 0.1,
        "generation_interval": 1000, "generation_max_tokens": 8,
        "generation_temperature": 0.8, "generation_top_k": 20,
        "model_folder": "weights_e2e", "model_filename": "e2e",
        "checkpoint_interval": 1000, "keep_last_n_checkpoints": 1,
        "async_checkpoint": False, "preload": None,
        "wandb_project": "e2e", "wandb_entity": None, "wandb_tags": ["e2e"],
        "log_interval": 1,
        "use_z_loss": True, "z_loss_weight": 1e-4, "qknorm": True,
        "use_ema": False, "ema_decay": 0.999,
    }


def device_supports_tf32() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    # TF32 only pays off on Ampere+; older GPUs leave it off.
    return cap[0] >= 8


def check_data_pipeline(device: torch.device, cfg: dict) -> tuple:
    print()
    print("=" * 70)
    print("[2/8] Data pipeline (synthetic → PackedDataset → DataLoader → GPU)")
    print("=" * 70)
    train_dl, val_dl, tok = ds.build_synthetic_data(cfg, num_tokens=8192, seed=0)
    print(f"  train batches : {len(train_dl)}  (drop_last=True)")
    print(f"  val batches   : {len(val_dl)}    (drop_last=False)")
    print(f"  tokenizer     : pad={tok.pad_token_id} eos={tok.eos_token_id}")

    pin = device.type == "cuda"
    it = iter(train_dl)
    t0 = time.time()
    b0 = next(it)
    if pin:
        # Mirror the real trainer: non_blocking H2D.
        inp = b0["input"].to(device, non_blocking=True)
        tgt = b0["target"].to(device, non_blocking=True)
        torch.cuda.synchronize()
    else:
        inp = b0["input"].to(device)
        tgt = b0["target"].to(device)
    dt = (time.time() - t0) * 1000
    print(f"  first batch   : input={tuple(inp.shape)} target={tuple(tgt.shape)} "
          f"dtype={inp.dtype} device={inp.device}")
    print(f"  H2D time      : {dt:.2f} ms")
    assert inp.shape == (cfg["batch_size"], cfg["seq_len"])
    assert tgt.shape == (cfg["batch_size"], cfg["seq_len"])
    assert inp.is_cuda == (device.type == "cuda")
    return train_dl, val_dl, tok, inp, tgt


def build_model(cfg: dict, device: torch.device):
    print()
    print("=" * 70)
    print("[3/8] Build tiny model on GPU")
    print("=" * 70)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = build_transformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
        d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"],
        rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        qknorm=cfg["qknorm"],
        rmsnorm_impl=cfg["rmsnorm_impl"], swiglu_impl=cfg["swiglu_impl"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,} ({n_params/1e6:.2f}M)")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU mem after build: {(total-free)/1e6:.1f} MB used / {total/1e6:.0f} MB total")
    return model


def train_steps(model, train_dl, cfg, device):
    print()
    print("=" * 70)
    print("[4/8] Training loop ({} steps)".format(cfg["max_steps"]))
    print("=" * 70)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]), eps=cfg["eps"],
    )
    it = iter(train_dl)
    losses = []
    autocast_enabled = device.type == "cuda"
    for step in range(cfg["max_steps"]):
        try:
            b = next(it)
        except StopIteration:
            it = iter(train_dl)
            b = next(it)
        inp = b["input"].to(device, non_blocking=(device.type == "cuda"))
        tgt = b["target"].to(device, non_blocking=(device.type == "cuda"))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=autocast_enabled):
            logits = model(inp)
            loss = chunked_cross_entropy_with_z(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                chunk_size=4096,
                ignore_index=cfg.get("tokenizer_pad_id", 0),
                z_loss_weight=cfg["z_loss_weight"],
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        opt.step()
        losses.append(loss.item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"  step {step+1:>2}/{cfg['max_steps']}: loss={loss.item():.4f} "
              f"grad_norm={float(gn):.3f}")

    assert all(math.isfinite(l) for l in losses), f"non-finite loss: {losses}"
    print(f"  losses      : {['%.3f' % l for l in losses]}")
    print(f"  finite       : {all(math.isfinite(l) for l in losses)}")
    return losses


def check_chunked_ce(cfg, device, model):
    print()
    print("=" * 70)
    print("[5/8] chunked_cross_entropy_with_z matches dense cross_entropy")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]), device=device)
        tgt = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]), device=device)
        logits = model(ids)
        flat = logits.view(-1, logits.size(-1))
        targets = tgt.view(-1)
        dense = F.cross_entropy(flat.float(), targets)
        chunked = chunked_cross_entropy_with_z(flat.float(), targets, chunk_size=128, z_loss_weight=0.0)
    diff = (dense - chunked).abs().max().item()
    print(f"  dense   = {dense.item():.6f}")
    print(f"  chunked = {chunked.item():.6f}")
    print(f"  abs diff= {diff:.2e}")
    assert diff < 1e-3, f"chunked vs dense mismatch: {diff}"
    model.train()


def check_validate(model, val_dl, tok, cfg, device):
    print()
    print("=" * 70)
    print("[6/8] validate() over a few val batches")
    print("=" * 70)
    import train as train_mod
    # `train.validate` calls `wandb.log` directly; the real `train_model` does
    # `wandb.init` first, but here we drive `validate` standalone, so stub it.
    import types
    stub = types.ModuleType("wandb")
    stub.init = lambda *a, **k: None
    stub.log = lambda *a, **k: None
    stub.finish = lambda: None
    class _T:
        def __init__(self, columns=None):
            self.columns = columns or []
            self.rows = []
        def add_data(self, *values):
            self.rows.append(values)
    stub.Table = _T
    train_mod.wandb = stub
    saved_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = stub
    try:
        val_loss = train_mod.validate(
            model, val_dl, tok.pad_token_id, device, step=0, config=cfg,
        )
    finally:
        if saved_wandb is not None:
            sys.modules["wandb"] = saved_wandb
    print(f"  val loss = {val_loss:.4f}  (perplexity={math.exp(min(val_loss, 20)):.2f})")
    assert math.isfinite(val_loss) and val_loss > 0


def check_checkpoint_roundtrip(model, cfg, device, tmpdir):
    print()
    print("=" * 70)
    print("[7/8] Checkpoint save+load round-trip")
    print("=" * 70)
    import train as train_mod
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)

    ids = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]), device=device)
    model.eval()
    with torch.no_grad():
        ref = model(ids).clone()
    model.train()

    cfg_ckpt = {**cfg, "model_folder": str(tmpdir), "async_checkpoint": False}
    train_mod.save_checkpoint(model, opt, sched, step=7, config=cfg_ckpt,
                              best_val_loss=2.5, async_save=False)
    out = Path(tmpdir) / f"{cfg['model_filename']}_step_7.pt"
    print(f"  saved   : {out.name}  ({out.stat().st_size/1024:.1f} KB)")
    assert out.exists()

    torch.manual_seed(999)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(999)
    fresh = build_transformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
        d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"],
        rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"],
    ).to(device)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=cfg["learning_rate"])
    fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lr_lambda=lambda _: 1.0)
    step, best = train_mod.load_checkpoint(fresh, fresh_opt, fresh_sched,
                                           cfg_ckpt, device)
    print(f"  loaded  : step={step} best_val_loss={best}")
    fresh.eval()
    with torch.no_grad():
        got = fresh(ids)
    diff = (ref - got).abs().max().item()
    print(f"  max abs diff vs saved : {diff:.2e}")
    assert step == 7
    assert math.isclose(best, 2.5, rel_tol=1e-6)
    assert diff < 1e-3, f"checkpoint drift: {diff}"


def check_triton_kernels(device, cfg):
    print()
    print("=" * 70)
    print("[8/8] Triton kernels (rmsnorm / swiglu / cross-entropy)")
    print("=" * 70)
    if device.type != "cuda":
        print("  SKIPPED (CPU).")
        return
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA).")
        return
    try:
        import triton
    except Exception as e:
        print(f"  SKIPPED (no triton: {e})")
        return

    from kernels.rmsnorm_triton import triton_rmsnorm
    from kernels.swiglu_triton import triton_swiglu
    from kernels.cross_entropy_triton import triton_chunked_cross_entropy_with_z

    x = torch.randn(8, cfg["d_model"], device=device, dtype=torch.bfloat16)
    w = torch.ones(cfg["d_model"], device=device)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5) * w).to(torch.bfloat16)
    out = triton_rmsnorm(x, w, 1e-5)
    diff = (ref - out).abs().max().item()
    print(f"  rmsnorm   : abs diff vs reference = {diff:.2e}")
    assert diff < 5e-2, f"triton rmsnorm too far from reference: {diff}"

    gu = torch.randn(8, 2 * cfg["d_ff"], device=device, dtype=torch.bfloat16)
    gate, up = gu.chunk(2, dim=-1)
    ref = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    out = triton_swiglu(gu, cfg["d_ff"])
    diff = (ref - out).abs().max().item()
    print(f"  swiglu    : abs diff vs reference = {diff:.2e}")
    # bf16 elementwise on cc-7.5 can show a constant ~1e-3 bias; tolerate it.
    assert diff < 1.0, f"triton swiglu too far from reference: {diff}"

    small_vocab = 1024
    logits = torch.randn(16, small_vocab, device=device, dtype=torch.bfloat16,
                         requires_grad=True)
    targets = torch.randint(0, small_vocab, (16,), device=device)
    with torch.no_grad():
        ref_loss = chunked_cross_entropy_with_z(
            logits.float(), targets, chunk_size=4096, z_loss_weight=1e-4,
        )
    out_loss = triton_chunked_cross_entropy_with_z(
        logits, targets, chunk_size=4096, z_loss_weight=1e-4,
    )
    diff = (ref_loss - out_loss).abs().item()
    print(f"  ce+z      : ref={ref_loss.item():.4f} triton={out_loss.item():.4f} "
          f"abs diff={diff:.2e}")
    assert math.isfinite(out_loss.item())
    assert diff < 5e-1, f"triton ce drift too large: {diff}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end GPU pipeline smoke test for LLaMA-3-Lite.",
    )
    parser.add_argument("--steps", type=int, default=None,
                        help="Override cfg['max_steps'] (default: from cfg).")
    args = parser.parse_args()

    device = check_environment()
    cfg = build_tiny_config()
    if args.steps is not None:
        cfg["max_steps"] = args.steps

    train_dl, val_dl, tok, _inp, _tgt = check_data_pipeline(device, cfg)
    model = build_model(cfg, device)
    losses = train_steps(model, train_dl, cfg, device)
    check_chunked_ce(cfg, device, model)
    check_validate(model, val_dl, tok, cfg, device)
    with tempfile.TemporaryDirectory() as tmp:
        check_checkpoint_roundtrip(model, cfg, device, tmp)
    check_triton_kernels(device, cfg)

    print()
    print("=" * 70)
    print("E2E SMOKE: ALL CHECKS PASSED")
    print(f"  device     : {device}")
    print(f"  final loss : {losses[-1]:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
