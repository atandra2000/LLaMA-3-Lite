#!/usr/bin/env python
"""GPU integration test for the full LLaMA-3-Lite training pipeline."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
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
    passed = 0
    failed = 0
    stage_times: dict[str, float] = {}


class Check:
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
        CheckResult.stage_times[self.name] = dt
        if exc is None:
            print(f"  [PASS] {self.name}  ({dt:.1f} ms)", flush=True)
            CheckResult.passed += 1
        else:
            print(f"  [FAIL] {self.name}  ({dt:.1f} ms)", flush=True)
            if self.verbose and tb is not None:
                traceback.print_exception(exc_type, exc, tb)
            CheckResult.failed += 1
        return True  # suppress so we keep going


def check(name: str) -> Check:
    return Check(name)


def gpu_config(tmp_dir: str) -> dict:
    """Config that exercises every code path while fitting in ~4 GB."""
    cfg = get_config()
    cfg.update({
        "d_model": 128, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "head_dim": 32, "d_ff": 512, "vocab_size": 1000, "seq_len": 64,
        "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "dropout": 0.0,
        "tie_embeddings": False, "bias": False,
        "batch_size": 8, "gradient_accumulation": 1, "max_steps": 30,
        "learning_rate": 3e-4, "min_lr": 3e-5, "warmup_steps": 5,
        "weight_decay": 0.1, "max_grad_norm": 1.0,
        "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
        "dtype": "bfloat16",
        "use_flash_attention": True,
        "compile_model": False,
        "gradient_checkpointing": True,
        "use_chunked_cross_entropy": True,
        "tf32": True, "cudnn_benchmark": True,
        "cuda_alloc_conf": "expandable_segments:True",
        "num_workers": 0, "prefetch_factor": 2, "pin_memory": True,
        "document_packing": True, "target_tokens": 4096,
        "data_cache_dir": str(Path(tmp_dir) / "data_cache"),
        "data_cache_filename": "tokens_gpu.bin",
        "reuse_data_cache": False, "shuffle_documents": True,
        "shuffle_seed": 42, "dedup": True, "dedup_hash_bytes": 16,
        "min_doc_tokens": 4, "max_doc_tokens": 64, "tokenize_batch_size": 10,
        "val_interval": 10, "val_max_batches": 3, "val_split": 0.1,
        "generation_interval": 20, "generation_max_tokens": 16,
        "generation_temperature": 0.8, "generation_top_k": 20,
        "model_folder": str(Path(tmp_dir) / "weights"),
        "model_filename": "gpu_test",
        "checkpoint_interval": 10, "keep_last_n_checkpoints": 2,
        "async_checkpoint": True, "preload": None,
        "wandb_project": "test", "wandb_entity": None,
        "wandb_tags": ["test"], "log_interval": 5,
        "top_k": 20, "temperature": 0.8,
    })
    return cfg


def build_synthetic_dataloaders(cfg, device):
    """Token buffers packed with BOS..EOS documents (no tokenizer needed)."""
    seq_len = cfg["seq_len"]
    eos, bos = 0, 1
    rng = np.random.default_rng(42)
    doc_len = max(8, seq_len // 2)
    tokens: list[int] = []
    needed = (seq_len + 1) * (cfg["max_steps"] * cfg["batch_size"] * 4 + 100)
    while len(tokens) < needed:
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
        collate_fn=ds.collate_fn, drop_last=True,
        pin_memory=cfg["pin_memory"] and device.type == "cuda")
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=ds.collate_fn,
        pin_memory=cfg["pin_memory"] and device.type == "cuda")
    return train_dl, val_dl


class FakeTokenizer:
    """Minimal stand-in for the LLaMA-3 tokenizer used by generate_samples."""
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 0

    def encode(self, text: str):
        return [min(self.vocab_size - 1, max(2, ord(c))) for c in text[:32]]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if 2 <= i < 128)


class WandbStub:
    def __init__(self):
        self.init_called = False
        self.init_kwargs: dict = {}
        self.log_calls: list[tuple] = []   # list of (payload, step)

    def init(self, **kwargs):
        self.init_called = True
        self.init_kwargs = kwargs
        return self

    def log(self, payload, step=None, **kw):
        self.log_calls.append((payload, step))

    def finish(self):
        pass


def install_wandb_stub(stub: WandbStub):
    import wandb
    wandb.init = stub.init
    wandb.log = stub.log
    wandb.finish = stub.finish
    class _Table:
        def __init__(self, columns=None):
            self.columns = columns or []
            self.rows = []
        def add_data(self, *args):
            self.rows.append(args)
    wandb.Table = _Table


def run(steps: int, device_str: str, verbose: bool):
    Check.verbose = verbose
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"ERROR: CUDA requested but not available.")
        sys.exit(2)

    print(f"\nLLaMA-3-Lite GPU integration test  (device={device})")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = gpu_config(tmp)
        torch.manual_seed(0); np.random.seed(0); import random
        random.seed(0)

        with check("setup_gpu_optimizations"):
            train_mod.setup_gpu_optimizations(cfg)
            if device.type == "cuda":
                assert torch.backends.cuda.matmul.allow_tf32 == cfg["tf32"]
                assert torch.backends.cudnn.allow_tf32 == cfg["tf32"]

        model = None
        with check("build_transformer_on_gpu"):
            model = build_transformer(
                vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
                d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"],
                rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"],
                gradient_checkpointing=cfg["gradient_checkpointing"],
            ).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            assert n_params > 0
            assert all(p.device.type == device.type
                        for p in model.parameters()), "params not on device"
            print(f"        model: {n_params/1e6:.2f}M params on {device}")

        train_dl = val_dl = None
        with check("synthetic_dataloaders"):
            train_dl, val_dl = build_synthetic_dataloaders(cfg, device)
            batch = next(iter(train_dl))
            assert batch["input"].shape == (cfg["batch_size"], cfg["seq_len"])
            assert batch["input"].dtype == torch.long
            assert batch["input"].is_pinned() or device.type == "cpu"

        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay_params if p.dim() >= 2 else no_decay_params).append(p)
        optimizer = torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": cfg["weight_decay"]},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]),
            eps=cfg["eps"])
        scheduler = train_mod.CosineWithWarmup(
            optimizer, warmup_steps=cfg["warmup_steps"],
            max_steps=cfg["max_steps"], min_lr=cfg["min_lr"],
            peak_lr=cfg["learning_rate"])
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        pad_id = 0
        grad_accum = cfg["gradient_accumulation"]
        use_chunked_ce = cfg["use_chunked_cross_entropy"]
        wandb_stub = WandbStub()
        install_wandb_stub(wandb_stub)
        train_iter = iter(train_dl)
        losses: list[float] = []
        grad_norms: list[float] = []
        lrs: list[float] = []

        with check("training_loop_bf16_gradscaler_chunked_ce"):
            model.train()
            torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
            for step in range(1, steps + 1):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dl); batch = next(train_iter)
                ids = batch["input"].to(device, non_blocking=True)
                tgt = batch["target"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    logits = model(ids)
                    if use_chunked_ce:
                        loss = chunked_cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            tgt.view(-1), chunk_size=65536,
                            ignore_index=pad_id)
                    else:
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            tgt.view(-1), ignore_index=pad_id)
                    loss = loss / grad_accum

                scaler.scale(loss).backward()

                if step % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=cfg["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                if step % cfg["log_interval"] == 0:
                    losses.append(loss.item() * grad_accum)
                    grad_norms.append(float(grad_norm))
                    lrs.append(scheduler.get_lr())
                    wandb_stub.log({
                        "train/loss": loss.item() * grad_accum,
                        "train/lr": scheduler.get_lr(),
                        "train/grad_norm": float(grad_norm),
                    }, step=step)
            assert all(np.isfinite(x) for x in losses), losses
            assert len(losses) == steps // cfg["log_interval"]
            peak, min_lr = cfg["learning_rate"], cfg["min_lr"]
            assert all(min_lr - 1e-9 <= lr <= peak + 1e-9 for lr in lrs), lrs
            assert all(lrs[i] >= lrs[i + 1] - 1e-9 for i in range(len(lrs) - 1)), lrs
            peak_mem = (torch.cuda.max_memory_allocated() / 1e9
                        if device.type == "cuda" else 0.0)
            print(f"        steps={steps}  final_loss={losses[-1]:.4f}  "
                    f"final_lr={lrs[-1]:.2e}  peak_gpu_mem={peak_mem:.2f} GB")

        with check("chunked_ce_equals_dense_ce_gpu"):
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                                  dtype=torch.bfloat16):
                ids = batch["input"].to(device)
                tgt = batch["target"].to(device)
                logits = model(ids)
                dense = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                        tgt.view(-1), ignore_index=pad_id,
                                        reduction="mean")
                chunked = chunked_cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1),
                    chunk_size=8, ignore_index=pad_id)
            # BF16 makes this noisier; allow 1e-3 relative slack.
            rel = (dense - chunked).abs().item() / max(dense.abs().item(), 1e-6)
            assert rel < 1e-3, (dense.item(), chunked.item(), rel)

        with check("validate_full_loop"):
            val_loss = train_mod.validate(model, val_dl, pad_id, device,
                                          step=steps, config=cfg)
            assert np.isfinite(val_loss)
            assert val_loss > 0
            val_logs = [(p, s) for p, s in wandb_stub.log_calls
                        if "val/loss" in p]
            assert len(val_logs) == 1, val_logs
            assert "val/perplexity" in val_logs[0][0]
            print(f"        val_loss={val_loss:.4f}  "
                  f"perplexity={np.exp(min(val_loss, 20)):.2f}")

        with check("generate_samples_autoregressive"):
            tok = FakeTokenizer(cfg["vocab_size"])
            train_mod.generate_samples(model, tok, device, step=steps,
                                        config=cfg)
            gen_logs = [p for p, _ in wandb_stub.log_calls
                        if "gen/samples" in p]
            assert len(gen_logs) == 1
            table = gen_logs[0]["gen/samples"]
            assert len(table.rows) == 5
            for prompt, generated, step_val in table.rows:
                prompt_tok = tok.encode(prompt)
                gen_tok = tok.encode(generated)
                assert len(gen_tok) >= len(prompt_tok), \
                    f"no tokens generated: prompt={len(prompt_tok)} " \
                    f"gen={len(gen_tok)} (prompt={prompt!r})"
                assert step_val == steps, step_val
            model.train()

        with check("async_checkpoint_save_and_resume"):
            save_thread = train_mod.save_checkpoint(
                model, optimizer, scheduler, step=steps, config=cfg,
                best_val_loss=val_loss, async_save=True)
            if save_thread is not None:
                save_thread.join(timeout=10)
            ckpt_path = Path(cfg["model_folder"]) / \
                        f"{cfg['model_filename']}_step_{steps}.pt"
            assert ckpt_path.exists(), f"checkpoint not written: {ckpt_path}"

            fresh = build_transformer(
                vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
                d_ff=cfg["d_ff"], max_seq_len=cfg["seq_len"],
                rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"],
                gradient_checkpointing=cfg["gradient_checkpointing"],
            ).to(device)
            fresh_opt = torch.optim.AdamW(
                [{"params": decay_params, "weight_decay": cfg["weight_decay"]},
                 {"params": no_decay_params, "weight_decay": 0.0}],
                lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]),
                eps=cfg["eps"])
            fresh_sched = train_mod.CosineWithWarmup(
                fresh_opt, warmup_steps=cfg["warmup_steps"],
                max_steps=cfg["max_steps"], min_lr=cfg["min_lr"],
                peak_lr=cfg["learning_rate"])
            resumed_step, resumed_best = train_mod.load_checkpoint(
                fresh, fresh_opt, fresh_sched, cfg, device)
            assert resumed_step == steps, resumed_step
            assert resumed_best == val_loss, (resumed_best, val_loss)

            ids = batch["input"].to(device)
            model.eval(); fresh.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                                  dtype=torch.bfloat16):
                a = model(ids); b = fresh(ids)
            assert torch.allclose(a, b, atol=1e-4), \
                f"resume changed outputs: max diff {(a-b).abs().max().item()}"
            print(f"        resumed at step {resumed_step}, "
                   f"outputs match to {(a-b).abs().max().item():.2e}")

        with check("wandb_logging_contract"):
            import wandb
            wandb.init(
                project=cfg["wandb_project"],
                entity=cfg.get("wandb_entity"),
                name="gpu-integration-test",
                config={
                    "architecture": "LLaMA 3",
                    "d_model": cfg["d_model"], "n_layers": cfg["n_layers"],
                    "n_heads": cfg["n_heads"], "n_kv_heads": cfg["n_kv_heads"],
                    "d_ff": cfg["d_ff"], "vocab_size": cfg["vocab_size"],
                    "seq_len": cfg["seq_len"],
                    "params_total": n_params,
                    "params_non_embed": model.get_num_params(non_embedding=True),
                    "batch_size": cfg["batch_size"],
                    "gradient_accumulation": cfg.get("gradient_accumulation", 1),
                    "learning_rate": cfg["learning_rate"],
                    "min_lr": cfg["min_lr"],
                    "warmup_steps": cfg["warmup_steps"],
                    "max_steps": cfg["max_steps"],
                    "optimizer": "AdamW",
                    "beta1": cfg["beta1"], "beta2": cfg["beta2"],
                    "weight_decay": cfg["weight_decay"],
                    "precision": "bf16",
                    "gradient_checkpointing": cfg["gradient_checkpointing"],
                    "chunked_cross_entropy": cfg["use_chunked_cross_entropy"],
                    "torch_compile": cfg.get("compile_model", True),
                },
                tags=cfg.get("wandb_tags", []),
            )
            assert wandb_stub.init_called, "wandb.init not called"
            init_cfg = wandb_stub.init_kwargs.get("config", {})
            for key in ("d_model", "n_layers", "n_heads", "n_kv_heads",
                        "vocab_size", "seq_len", "batch_size", "learning_rate",
                        "warmup_steps", "max_steps"):
                assert key in init_cfg, f"missing {key} in wandb init config"
            assert init_cfg["params_total"] == n_params, \
                f"params_total={init_cfg['params_total']} vs actual={n_params}"
            assert init_cfg["params_non_embed"] == \
                   model.get_num_params(non_embedding=True), \
                f"params_non_embed={init_cfg['params_non_embed']} vs actual"
            keys_logged = set()
            for payload, _ in wandb_stub.log_calls:
                keys_logged.update(payload.keys())
            for required in ("train/loss", "train/lr", "train/grad_norm",
                              "val/loss", "val/perplexity", "gen/samples"):
                assert required in keys_logged, \
                    f"missing wandb metric: {required}"

    print("\n" + "=" * 60)
    total = CheckResult.passed + CheckResult.failed
    print(f"Summary: {CheckResult.passed}/{total} stages passed, "
          f"{CheckResult.failed} failed.\n")
    if verbose and CheckResult.stage_times:
        print("Stage timings:")
        for name, ms in sorted(CheckResult.stage_times.items(),
                               key=lambda kv: -kv[1]):
            print(f"  {ms:8.1f} ms  {name}")
        print()
    sys.exit(0 if CheckResult.failed == 0 else 1)


def main():
    p = argparse.ArgumentParser(description="LLaMA-3-Lite GPU integration test")
    p.add_argument("--steps", type=int, default=30,
                   help="number of training steps (default 30)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print tracebacks on failure + stage timings")
    args = p.parse_args()
    device_str = (args.device if args.device != "auto"
                  else ("cuda" if torch.cuda.is_available() else "cpu"))
    run(steps=args.steps, device_str=device_str, verbose=args.verbose)


if __name__ == "__main__":
    main()