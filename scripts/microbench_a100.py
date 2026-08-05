#!/usr/bin/env python
"""Microbenchmark for the three sanctioned Triton kernels (AGENTS.md rule 2).

Compares each Triton path against its pure-PyTorch reference at project
scale (batch 96, seq 2048, d_model 1024, d_ff 4096, vocab 128,000) and
reports the speedup. AGENTS.md rule 2: a sanctioned path must clear
>= 1.5x before it may be enabled by default; this script is the gate.

Usage:
    python scripts/microbench_a100.py            # full suite, project scale
    python scripts/microbench_a100.py --kernel rmsnorm
    python scripts/microbench_a100.py --json

On machines without triton or CUDA it prints the reason and exits 0
(the rule is unmeasurable there, not violated).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "kernels"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MIN_SPEEDUP = 1.5
DEFAULT_REPS = 20
WARMUP_REPS = 5


def _time_fn(fn, *args, reps: int, warmup: int) -> float:
    """Median wall time of ``fn(*args)`` over ``reps`` runs, in seconds."""
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            t0 = time.perf_counter()
        fn(*args)
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
        else:
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def _bench_pair(name: str, ref, triton, *args, reps: int) -> dict:
    ref_t = _time_fn(ref, *args, reps=reps, warmup=WARMUP_REPS)
    tri_t = _time_fn(triton, *args, reps=reps, warmup=WARMUP_REPS)
    return {
        "kernel": name,
        "pytorch_ms": ref_t * 1000,
        "triton_ms": tri_t * 1000,
        "speedup": ref_t / tri_t if tri_t > 0 else float("inf"),
    }


def bench_rmsnorm(reps: int, device: torch.device) -> dict:
    from kernels.rmsnorm_triton import rmsnorm_pytorch, triton_rmsnorm, HAS_TRITON

    B, S, D = 96, 2048, 1024
    x = torch.randn(B * S, D, device=device)
    w = torch.randn(D, device=device)
    if not HAS_TRITON:
        return {"kernel": "rmsnorm", "skipped": "triton not installed"}
    return _bench_pair("rmsnorm", rmsnorm_pytorch, triton_rmsnorm,
                       x, w, 1e-5, reps=reps)


def bench_swiglu(reps: int, device: torch.device) -> dict:
    from kernels.swiglu_triton import swiglu_pytorch, triton_swiglu, HAS_TRITON

    B, S, D = 96, 2048, 4096
    gate_up = torch.randn(B * S, 2 * D, device=device)
    gate, up = gate_up.split(D, dim=-1)
    if not HAS_TRITON:
        return {"kernel": "swiglu", "skipped": "triton not installed"}

    ref = lambda: swiglu_pytorch(gate, up)
    tri = lambda: triton_swiglu(gate_up, D)
    ref_t = _time_fn(ref, reps=reps, warmup=WARMUP_REPS)
    tri_t = _time_fn(tri, reps=reps, warmup=WARMUP_REPS)
    return {
        "kernel": "swiglu",
        "pytorch_ms": ref_t * 1000,
        "triton_ms": tri_t * 1000,
        "speedup": ref_t / tri_t if tri_t > 0 else float("inf"),
    }


def bench_cross_entropy(reps: int, device: torch.device) -> dict:
    from kernels.cross_entropy_triton import (
        cross_entropy_with_z_pytorch,
        triton_chunked_cross_entropy_with_z,
        HAS_TRITON,
    )

    chunk_rows, vocab = 256, 128_000
    logits = torch.randn(chunk_rows, vocab, device=device)
    targets = torch.randint(0, vocab, (chunk_rows,), device=device)
    if not HAS_TRITON:
        return {"kernel": "cross_entropy", "skipped": "triton not installed"}
    return _bench_pair("cross_entropy",
                       lambda l, t: cross_entropy_with_z_pytorch(l, t, -100, 1e-4),
                       lambda l, t: triton_chunked_cross_entropy_with_z(l, t, -100, 1e-4),
                       logits, targets, reps=reps)


def main() -> int:
    p = argparse.ArgumentParser(description="Triton kernel microbenchmark (AGENTS.md rule 2)")
    p.add_argument("--kernel", choices=("rmsnorm", "swiglu", "cross_entropy"),
                   default=None, help="benchmark one kernel (default: all)")
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device — AGENTS.md rule 2 is unmeasurable here; "
              "run on the A100 (Linux + CUDA) to enforce the 1.5x gate.")
        return 0

    device = torch.device("cuda")
    torch.manual_seed(0)

    benchs = [
        bench_rmsnorm if args.kernel in (None, "rmsnorm") else None,
        bench_swiglu if args.kernel in (None, "swiglu") else None,
        bench_cross_entropy if args.kernel in (None, "cross_entropy") else None,
    ]
    results = [b(args.reps, device) for b in benchs if b is not None]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"\nTriton kernel microbenchmark (device={device}, reps={args.reps})")
    print(f"{'kernel':<15}{'pytorch (ms)':>14}{'triton (ms)':>14}{'speedup':>10}  verdict")
    failed = False
    for r in results:
        if "skipped" in r:
            print(f"{r['kernel']:<15}  skipped: {r['skipped']}")
            continue
        ok = r["speedup"] >= MIN_SPEEDUP
        failed = failed or not ok
        print(f"{r['kernel']:<15}{r['pytorch_ms']:>14.2f}{r['triton_ms']:>14.2f}"
              f"{r['speedup']:>9.2f}x  {'OK >= 1.5x' if ok else 'BELOW 1.5x'}")
    if failed:
        print(f"\nFAIL: a kernel is below the {MIN_SPEEDUP}x bar — do not "
              f"enable it by default (AGENTS.md rule 2).")
        return 1
    print(f"\nAll measured kernels clear the {MIN_SPEEDUP}x bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
