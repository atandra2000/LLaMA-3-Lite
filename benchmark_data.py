#!/usr/bin/env python
"""Data pipeline benchmark for LLaMA-3-Lite (no HuggingFace download)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

import dataset as ds


def build_benchmark_buffer(num_tokens: int, vocab_size: int, seq_len: int,
                           seed: int = 42) -> np.ndarray:
    """Synthetic uint32 token buffer packed with BOS..EOS documents."""
    rng = np.random.default_rng(seed)
    doc_len = max(8, seq_len // 2)
    out: list[int] = []
    while len(out) < num_tokens:
        out.append(1)  # BOS
        out.extend(rng.integers(2, max(3, vocab_size),
                                size=doc_len - 2).tolist())
        out.append(0)  # EOS
    return np.asarray(out[:num_tokens], dtype=np.uint32)


def benchmark(steps: int, batch_size: int, seq_len: int, vocab_size: int,
              num_workers: int, prefetch_factor: int, pin_memory: bool,
              device: torch.device, with_model_forward: bool) -> dict:
    """Run the benchmark and return a metrics dict."""
    chunk = seq_len + 1
    n_chunks_target = steps * batch_size * 4  # enough to not run out
    n_tokens = n_chunks_target * chunk + 100
    data = build_benchmark_buffer(n_tokens, vocab_size, seq_len)
    train_ds = ds.PackedDataset(data, seq_len=seq_len, eos_id=0)
    sampler = ds.ShuffledRangeSampler(train_ds.n_chunks, seed=42, offset=0)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, prefetch_factor=(prefetch_factor
                                                  if num_workers > 0 else None),
        pin_memory=pin_memory and device.type == "cuda",
        collate_fn=ds.collate_fn, drop_last=True,
        persistent_workers=num_workers > 0,
    )

    model = None
    if with_model_forward:
        from model import build_transformer
        model = build_transformer(
            vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4,
            n_kv_heads=2, head_dim=32, d_ff=512, max_seq_len=seq_len,
            rope_theta=500000.0,
        ).to(device).eval()

    it = iter(loader)
    warm = next(it)
    _ = warm["input"].to(device, non_blocking=True)

    times = []
    tokens_seen = 0
    start = time.time()
    for step in range(steps):
        t0 = time.time()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        inp = batch["input"].to(device, non_blocking=True)
        if model is not None:
            with torch.no_grad():
                _ = model(inp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        tokens_seen += inp.numel()
    total = time.time() - start

    return {
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": batch_size * seq_len,
        "total_tokens": tokens_seen,
        "total_time_s": total,
        "tokens_per_sec": tokens_seen / total if total > 0 else 0.0,
        "mean_step_ms": (sum(times) / len(times) * 1000) if times else 0.0,
        "p50_step_ms": (sorted(times)[len(times) // 2] * 1000) if times else 0.0,
        "p99_step_ms": (sorted(times)[int(len(times) * 0.99)] * 1000
                        if len(times) >= 100 else 0.0),
        "with_model_forward": with_model_forward,
        "device": str(device),
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
    }


def main():
    p = argparse.ArgumentParser(description="LLaMA-3-Lite data pipeline benchmark")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--vocab_size", type=int, default=128000)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prefetch_factor", type=int, default=16)
    p.add_argument("--pin_memory", action="store_true",
                   help="pin host memory for faster CUDA H2D copy")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--with_model_forward", action="store_true",
                   help="also run a tiny model forward to measure end-to-end")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    metrics = benchmark(
        steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        vocab_size=args.vocab_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, pin_memory=args.pin_memory,
        device=device, with_model_forward=args.with_model_forward,
    )

    if args.json:
        import json
        print(json.dumps(metrics, indent=2))
        return

    print(f"\nLLaMA-3-Lite data pipeline benchmark")
    print(f"device            : {metrics['device']}")
    print(f"steps             : {metrics['steps']}")
    print(f"batch_size        : {metrics['batch_size']}")
    print(f"seq_len           : {metrics['seq_len']}")
    print(f"tokens/step       : {metrics['tokens_per_step']:,}")
    print(f"num_workers       : {metrics['num_workers']}")
    print(f"prefetch_factor   : {metrics['prefetch_factor']}")
    print(f"with_model_forward: {metrics['with_model_forward']}")
    print(f"-" * 40)
    print(f"total_tokens      : {metrics['total_tokens']:,}")
    print(f"total_time        : {metrics['total_time_s']:.3f} s")
    print(f"throughput        : {metrics['tokens_per_sec']:.0f} tokens/s "
          f"({metrics['tokens_per_sec']/1e6:.2f}M tok/s)")
    print(f"mean step         : {metrics['mean_step_ms']:.1f} ms")
    print(f"p50 step          : {metrics['p50_step_ms']:.1f} ms")
    print(f"p99 step          : {metrics['p99_step_ms']:.1f} ms")


if __name__ == "__main__":
    main()