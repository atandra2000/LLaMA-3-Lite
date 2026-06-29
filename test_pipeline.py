#!/usr/bin/env python
"""Standalone CPU smoke test for the LLaMA-3-Lite pipeline (no HF download)."""
from __future__ import annotations

import argparse
import os
import sys
import time
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
import torch.nn.functional as F

import dataset as ds
from config import get_config
from model import build_transformer, chunked_cross_entropy
import train as train_mod


class CheckResult:
    """Aggregate pass/fail counters shared across checks."""
    passed = 0
    failed = 0


class Check:
    """Context manager that runs a named check and counts pass/fail."""
    verbose = False

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if self.verbose:
            print(f"  ... {self.name}", flush=True)
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = (time.time() - self.t0) * 1000
        if exc is None:
            print(f"  [PASS] {self.name}  ({dt:.1f} ms)", flush=True)
            CheckResult.passed += 1
        else:
            print(f"  [FAIL] {self.name}  ({dt:.1f} ms)", flush=True)
            if self.verbose and tb is not None:
                traceback.print_exception(exc_type, exc, tb)
            CheckResult.failed += 1
        return True


def check(name: str) -> Check:
    return Check(name)


def tiny_config():
    base = get_config()
    base.update({
        "d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "head_dim": 16, "d_ff": 128, "vocab_size": 256, "seq_len": 32,
        "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "dropout": 0.0,
        "tie_embeddings": False, "bias": False,
        "batch_size": 4, "gradient_accumulation": 1, "max_steps": 10,
        "learning_rate": 3e-4, "min_lr": 3e-5, "warmup_steps": 2,
        "weight_decay": 0.1, "max_grad_norm": 1.0, "beta1": 0.9, "beta2": 0.95,
        "eps": 1e-8, "dtype": "float32", "use_flash_attention": False,
        "compile_model": False, "gradient_checkpointing": False,
        "use_chunked_cross_entropy": True, "tf32": False,
        "cudnn_benchmark": False, "num_workers": 0, "prefetch_factor": 2,
        "pin_memory": False, "document_packing": True, "target_tokens": 4096,
        "data_cache_dir": "data_cache_smoke", "data_cache_filename": "t.bin",
        "reuse_data_cache": False, "shuffle_documents": True, "shuffle_seed": 42,
        "dedup": True, "dedup_hash_bytes": 16, "min_doc_tokens": 4,
        "max_doc_tokens": 64, "tokenize_batch_size": 10,
        "val_interval": 1000, "val_max_batches": 2, "val_split": 0.1,
        "generation_interval": 1000, "generation_max_tokens": 8,
        "generation_temperature": 0.8, "generation_top_k": 20,
        "model_folder": "weights_smoke", "model_filename": "tiny",
        "checkpoint_interval": 1000, "keep_last_n_checkpoints": 2,
        "async_checkpoint": False, "preload": None, "log_interval": 1,
        "top_k": 20, "temperature": 0.8,
    })
    return base


def synthetic_dataloaders(cfg, device):
    """Build dataloaders from random tokens (no tokenizer needed)."""
    seq_len = cfg["seq_len"]
    eos, bos = 0, 1
    rng = np.random.default_rng(0)
    doc_len = max(8, seq_len // 2)
    tokens: list[int] = []
    while len(tokens) < (seq_len + 1) * 64:
        tokens.append(bos)
        tokens.extend(rng.integers(2, cfg["vocab_size"],
                                   size=doc_len - 2).tolist())
        tokens.append(eos)
    data = np.asarray(tokens, dtype=np.uint32)
    chunk = seq_len + 1
    split = (int(len(data) * (1.0 - cfg["val_split"])) // chunk) * chunk
    train_ds = ds.PackedDataset(data[:split], seq_len, eos)
    val_ds = ds.PackedDataset(data[split:], seq_len, eos)
    sampler = ds.ShuffledRangeSampler(train_ds.n_chunks, seed=42, offset=0)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=sampler,
        collate_fn=ds.collate_fn, drop_last=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=ds.collate_fn)
    return train_dl, val_dl


def main():
    parser = argparse.ArgumentParser(description="LLaMA-3-Lite CPU smoke test")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print tracebacks on failure")
    args = parser.parse_args()
    Check.verbose = args.verbose

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLLaMA-3-Lite smoke test  (device={device})\n" + "=" * 40)

    cfg = tiny_config()
    torch.manual_seed(0); np.random.seed(0)

    model = None
    with check("build_transformer"):
        model = build_transformer(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
            n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
            d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"],
            rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"],
            gradient_checkpointing=cfg["gradient_checkpointing"],
        ).to(device)
        assert model is not None
    if model is None:
        print("\nCannot continue: model build failed.")
        sys.exit(1)

    with check("forward_output_shape"):
        ids = torch.randint(0, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]),
                            device=device, dtype=torch.long)
        logits = model(ids)
        assert logits.shape == (cfg["batch_size"], cfg["seq_len"], cfg["vocab_size"])

    with check("chunked_cross_entropy_matches_full"):
        tgt = torch.randint(0, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]),
                            device=device, dtype=torch.long)
        full = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1),
                               reduction="mean")
        chk = chunked_cross_entropy(logits.view(-1, logits.size(-1)),
                                    tgt.view(-1), chunk_size=7)
        assert torch.allclose(full, chk, atol=1e-5), (full.item(), chk.item())

    with check("backward_and_optimizer_step"):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()
        loss = chunked_cross_entropy(logits.view(-1, logits.size(-1)),
                                     tgt.view(-1), chunk_size=7)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        opt.step()

    with check("loss_decreases_on_fixed_batch"):
        torch.manual_seed(1)
        ids2 = torch.randint(0, cfg["vocab_size"], (4, cfg["seq_len"]),
                             device=device, dtype=torch.long)
        tgt2 = torch.randint(0, cfg["vocab_size"], (4, cfg["seq_len"]),
                             device=device, dtype=torch.long)
        opt2 = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()
        first = None; last = None
        for step in range(20):
            l = chunked_cross_entropy(model(ids2).view(-1, cfg["vocab_size"]),
                                      tgt2.view(-1), chunk_size=7)
            opt2.zero_grad(); l.backward(); opt2.step()
            if step == 0: first = l.item()
            last = l.item()
        assert last < first, (first, last)

    with check("cosine_lr_schedule"):
        sched = train_mod.CosineWithWarmup(opt2, warmup_steps=2, max_steps=10,
                                           min_lr=1e-5, peak_lr=3e-4)
        lrs = []
        for _ in range(10):
            sched.step(); lrs.append(sched.get_lr())
        assert lrs[0] < lrs[1], lrs
        assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(1, 9)), lrs
        assert lrs[-1] >= 1e-5 - 1e-12, lrs[-1]

    with check("top_k_top_p_sampling"):
        torch.manual_seed(0)
        sl = torch.randn(2, cfg["vocab_size"], device=device)
        tok = train_mod.top_k_top_p_sampling(sl, top_k=10, top_p=0.9,
                                              temperature=1.0)
        assert tok.shape == (2, 1)
        assert (0 <= tok).all() and (tok < cfg["vocab_size"]).all()

    with check("checkpoint_round_trip"):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_cfg = {**cfg, "model_folder": tmp, "async_checkpoint": False}
            opt3 = torch.optim.AdamW(model.parameters(), lr=3e-4)
            sched3 = train_mod.CosineWithWarmup(opt3, 2, 10, 1e-5, 3e-4)
            loss = chunked_cross_entropy(model(ids2).view(-1, cfg["vocab_size"]),
                                          tgt2.view(-1), chunk_size=7)
            opt3.zero_grad(); loss.backward(); opt3.step(); sched3.step()
            train_mod.save_checkpoint(model, opt3, sched3, step=5, config=ckpt_cfg,
                                       best_val_loss=1.0, async_save=False)
            fresh = build_transformer(
                vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
                d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"],
                rope_theta=cfg["rope_theta"],
                rms_norm_eps=cfg["rms_norm_eps"]).to(device)
            fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
            fresh_sched = train_mod.CosineWithWarmup(fresh_opt, 2, 10, 1e-5, 3e-4)
            step, best = train_mod.load_checkpoint(fresh, fresh_opt, fresh_sched,
                                                   ckpt_cfg, device)
            model.eval(); fresh.eval()
            with torch.no_grad():
                a = model(ids2); b = fresh(ids2)
            assert step == 5, step
            assert best == 1.0, best
            assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()

    with check("synthetic_dataloaders"):
        train_dl, val_dl = synthetic_dataloaders(cfg, device)
        batch = next(iter(train_dl))
        assert batch["input"].shape == (cfg["batch_size"], cfg["seq_len"])
        assert batch["target"].shape == batch["input"].shape
        assert batch["input"].dtype == torch.long

    with check("mini_training_loop"):
        train_dl, val_dl = synthetic_dataloaders(cfg, device)
        opt4 = torch.optim.AdamW(model.parameters(), lr=3e-4)
        sched4 = train_mod.CosineWithWarmup(opt4, 2, 10, 1e-5, 3e-4)
        model.train()
        losses = []
        it = iter(train_dl)
        for step in range(5):
            batch = next(it, None)
            if batch is None:
                it = iter(train_dl); batch = next(it)
            b_ids = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            l = chunked_cross_entropy(model(b_ids).view(-1, cfg["vocab_size"]),
                                       b_tgt.view(-1), chunk_size=7)
            opt4.zero_grad(); l.backward(); opt4.step(); sched4.step()
            losses.append(l.item())
        assert len(losses) == 5
        assert all(np.isfinite(x) for x in losses), losses

    print("\n" + "=" * 40)
    total = CheckResult.passed + CheckResult.failed
    print(f"Summary: {CheckResult.passed}/{total} checks passed, "
          f"{CheckResult.failed} failed.\n")
    sys.exit(0 if CheckResult.failed == 0 else 1)


if __name__ == "__main__":
    main()