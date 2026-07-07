"""Shared check harness for the standalone pipeline smoke scripts.

Used by ``test_pipeline.py`` (CPU) and ``test_gpu_pipeline.py`` (GPU).
``testpaths`` in ``pytest.ini`` is restricted to ``tests/``, so this module
is not collected by pytest; it is imported by the two standalone scripts.
"""
from __future__ import annotations

import time
import traceback


class CheckResult:
    """Aggregate pass/fail counters (and per-check timings) shared across checks."""
    passed = 0
    failed = 0
    stage_times: dict[str, float] = {}


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
        CheckResult.stage_times[self.name] = dt
        if exc is None:
            print(f"  [PASS] {self.name}  ({dt:.1f} ms)", flush=True)
            CheckResult.passed += 1
        else:
            print(f"  [FAIL] {self.name}  ({dt:.1f} ms)", flush=True)
            if self.verbose and tb is not None:
                traceback.print_exception(exc_type, exc, tb)
            CheckResult.failed += 1
        return True  # suppress so the script keeps going


def check(name: str) -> Check:
    return Check(name)