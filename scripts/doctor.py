#!/usr/bin/env python3
"""Preflight gate for a paid pretraining run.

Run this on the target machine BEFORE launching ``train.py``. Exits non-zero
if any check fails. Expected FAILs on a laptop (no GPU) prove the gate works.

    python scripts/doctor.py --checksums   # also verify every shard sha256
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)


def record(status: str, name: str, detail: str) -> None:
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}: {detail}")


def ok(name, detail):
    record("PASS", name, detail)


def fail(name, detail):
    record("FAIL", name, detail)


def warn(name, detail):
    record("WARN", name, detail)


# --- config ---------------------------------------------------------------

def check_config():
    from config import get_config
    cfg = get_config()
    missing = [k for k in ("batch_size", "gradient_accumulation", "seq_len", "max_steps",
                          "target_tokens", "vocab_size", "model_folder", "model_filename",
                          "wandb_project") if k not in cfg]
    if missing:
        fail("config", f"key(s) not wired yet: {missing}")
        return cfg
    ok("config", f"{len(cfg)} keys, required keys present")

    micro, ga, seq = cfg["batch_size"], cfg["gradient_accumulation"], cfg["seq_len"]
    tokens_per_step = micro * seq * ga
    planned = cfg["max_steps"] * tokens_per_step
    ok("config.tokens", f"micro={micro} accum={ga} seq={seq} -> {tokens_per_step:,} tok/step")
    ratio = planned / cfg["target_tokens"]
    if 0.95 <= ratio <= 1.05:
        ok("config.budget", f"{planned:,} tokens planned vs target {cfg['target_tokens']:,} ({ratio:.2f}x)")
    else:
        fail("config.budget",
             f"{planned:,} tokens planned vs target {cfg['target_tokens']:,} ({ratio:.2f}x) — "
             f"raise/lower max_steps")
    return cfg


# --- GPU ------------------------------------------------------------------

def check_gpu(cfg):
    try:
        import torch
    except ImportError as exc:
        fail("torch", f"not importable ({exc}); pip install torch")
        return None
    ok("torch", f"{torch.__version__}")

    if not torch.cuda.is_available():
        fail("gpu", "CUDA not available — this run needs a CUDA GPU")
        return None

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)
    ok("gpu", f"{name} | {vram_gb:.1f} GB | sm_{cc[0]}{cc[1]}")

    # Derived peak at the configured micro-batch. See docs/training.md "Peak
    # memory: 92 GB -> 20 GB"; the strict (FP32-residual) figure is the one
    # that has to fit, not the optimistic 20 GB.
    micro = cfg["batch_size"]
    peak_strict_gb = 8.5 + 17.5 * (micro / 96)
    if peak_strict_gb < vram_gb - 3:
        ok("gpu.vram", f"derived strict peak ~{peak_strict_gb:.1f} GB at micro-batch {micro} "
                       f"fits {vram_gb:.1f} GB with margin")
    else:
        fail("gpu.vram", f"derived strict peak ~{peak_strict_gb:.1f} GB at micro-batch {micro} "
                         f"vs {vram_gb:.1f} GB — lower config['batch_size']")

    if not (torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else cc[0] >= 8):
        fail("gpu.bf16", "BF16 not supported; train.py autocasts to bfloat16")
    else:
        ok("gpu.bf16", "supported")
    return torch


# --- data -----------------------------------------------------------------

def check_data(cfg, do_checksums: bool):
    root = os.environ.get("LLM_DATA_ROOT")
    if not root:
        fail("data.root", "LLM_DATA_ROOT is empty, not wired yet — export it to the shards' parent dir")
        return
    root_p = Path(root)
    if not root_p.exists():
        fail("data.root", f"path missing on disk: {root_p}")
        return
    ok("data.root", str(root_p))

    shards_dir = root_p / "shards"
    manifest_path = shards_dir / "manifest.json"
    if not manifest_path.exists():
        fail("data.manifest", f"path missing on disk: {manifest_path}")
        return

    manifest = json.loads(manifest_path.read_text())
    ok("data.manifest",
       f"tokenizer={manifest.get('tokenizer_name')} vocab={manifest.get('vocab_size')} "
       f"eos={manifest.get('eos_token_id')} tokens={manifest.get('total_tokens'):,} "
       f"shards={manifest.get('shard_count')}")

    if manifest.get("tokenizer_name") != "llama3":
        fail("data.tokenizer", f"manifest tokenizer is {manifest.get('tokenizer_name')!r}, expected 'llama3'")
    if int(manifest.get("vocab_size", 0)) != int(cfg["vocab_size"]):
        warn("data.vocab", f"manifest vocab {manifest.get('vocab_size')} != config {cfg['vocab_size']} "
                           f"(train.py widens the embedding to the tokenizer length — check it covers eos)")
    eos = int(manifest.get("eos_token_id", 0))
    if eos >= int(cfg["vocab_size"]):
        warn("data.eos", f"eos_token_id {eos} >= vocab_size {cfg['vocab_size']}; "
                         f"needs real_vocab_size = max(vocab_size, len(tokenizer)) to cover it")

    entries = manifest.get("shards", [])
    listed = {e["path"] for e in entries}
    on_disk = {p.name for p in shards_dir.glob("shard_*.bin")}
    missing = listed - on_disk
    extra = on_disk - listed
    if missing:
        fail("data.shards", f"{len(missing)} manifest shard(s) missing on disk, e.g. {sorted(missing)[:3]}")
    else:
        ok("data.shards", f"all {len(listed)} manifest shards present")
    if extra:
        warn("data.shards.extra", f"{len(extra)} shard file(s) not in manifest: {sorted(extra)[:3]}")

    total = sum(int(e["n_tokens"]) for e in entries)
    if total != int(manifest.get("total_tokens", -1)):
        fail("data.counts", f"manifest n_tokens sum {total:,} != total_tokens {manifest.get('total_tokens'):,}")
    else:
        ok("data.counts", f"{total:,} tokens across {len(entries)} shards")

    # The manifest's n_eos is untrustworthy (the 2026-10-06 llama3 pack records
    # 0 while the bytes do contain separators). Count real separators instead.
    sep = int(manifest.get("eos_token_id", 128_009))
    if not do_checksums and entries:
        import numpy as np
        probe = shards_dir / entries[0]["path"]
        head = np.fromfile(probe, dtype=np.uint32, count=500_000)
        n_sep = int((head == sep).sum())
        if n_sep == 0:
            fail("data.eos", f"no separator id {sep} in the first 500k tokens of {probe.name} — "
                             f"documents run together, train/val split cannot align")
        else:
            ok("data.eos", f"separator id {sep} present ({n_sep} per 500k tokens) in {probe.name}")

    if do_checksums:
        import hashlib
        import numpy as np
        bad = []
        eos_total = 0
        eos_claims = 0
        for e in entries:
            p = shards_dir / e["path"]
            h = hashlib.sha256()
            arr = np.fromfile(p, dtype=np.uint32)
            eos_total += int((arr == sep).sum())
            eos_claims += int(e.get("n_eos", 0))
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() != e["sha256"]:
                bad.append(e["path"])
        if bad:
            fail("data.sha256", f"{len(bad)} shard(s) corrupt: {bad[:5]}")
        else:
            ok("data.sha256", f"all {len(entries)} shards match manifest")
        if eos_total == 0:
            fail("data.eos", f"no separator id {sep} anywhere in the corpus")
        elif eos_total != eos_claims:
            warn("data.eos", f"manifest n_eos={eos_claims:,} but {eos_total:,} real separators of id {sep} — "
                             f"manifest metadata undercounts; data is fine, split alignment uses the real id")
        else:
            ok("data.eos", f"{eos_total:,} document separators match the manifest")


def check_cache(cfg):
    """The training cache must match the corpus it claims to be.

    ``reuse_data_cache: True`` means a stale or toy ``data_cache/tokens.bin``
    is silently trained on instead of the real corpus. Size is the cheapest
    honest fingerprint.
    """
    cache = PROJECT_ROOT / cfg.get("data_cache_dir", "data_cache") / cfg.get("data_cache_filename", "tokens.bin")
    if not cache.exists():
        warn("data.cache", f"not built yet at {cache} — run data/prepare_data.py "
                          f"--skip-download --skip-clean --skip-tokenize --skip-pack first")
        return
    size = cache.stat().st_size
    tokens = size // 4

    root = os.environ.get("LLM_DATA_ROOT")
    expected = None
    if root:
        mpath = Path(root) / "shards" / "manifest.json"
        if mpath.exists():
            expected = int(json.loads(mpath.read_text()).get("total_tokens", 0))

    if expected is None:
        warn("data.cache", f"{tokens:,} tokens ({size / 1e9:.1f} GB) at {cache}; "
                           f"no manifest to compare against")
    elif tokens != expected:
        fail("data.cache",
             f"{cache} holds {tokens:,} tokens but the corpus manifest claims "
             f"{expected:,} — stale or partial cache. "
             f"reuse_data_cache would silently train on this. Delete the file and rebuild.")
    else:
        ok("data.cache", f"{tokens:,} tokens ({size / 1e9:.1f} GB) match the manifest")


# --- resources ------------------------------------------------------------

def check_disk(cfg):
    root = os.environ.get("LLM_DATA_ROOT")
    target = Path(root) if root else PROJECT_ROOT
    if not target.exists():
        target = PROJECT_ROOT
    free_gb = shutil.disk_usage(target).free / 1e9

    # shards (already downloaded) + keep_last_n full ckpts + best + final + env
    n_params_gb_per_ckpt = 8.0  # model + AdamW moments + EMA + RNG, ~515M params
    ckpt_gb = n_params_gb_per_ckpt * (cfg.get("keep_last_n_checkpoints", 3) + 2)
    need_gb = ckpt_gb + 20.0
    if free_gb >= need_gb:
        ok("disk", f"{free_gb:.1f} GB free at {target}; need ~{need_gb:.0f} GB "
                   f"(checkpoints {ckpt_gb:.0f} GB + headroom)")
    else:
        fail("disk", f"{free_gb:.1f} GB free at {target}; need ~{need_gb:.0f} GB")

    # The training loader memmaps data_cache/tokens.bin, so it does not need
    # the corpus resident in RAM. 16 GB is a floor for torch + dataloader
    # workers, not a corpus-size figure.
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        phys = os.sysconf("SC_PHYS_PAGES") * page / 1e9
        if phys >= 16:
            ok("ram", f"{phys:.1f} GB physical (loader memmaps the corpus; 16 GB floor)")
        else:
            fail("ram", f"{phys:.1f} GB physical; give the pod at least 16 GB system RAM")
    except (ValueError, OSError):
        warn("ram", "cannot read physical RAM")


# --- tooling --------------------------------------------------------------

def check_tooling():
    for tool in ("git", "python3", "rclone"):
        if shutil.which(tool):
            ok(f"tool.{tool}", shutil.which(tool))
        else:
            (fail if tool != "rclone" else warn)(f"tool.{tool}", "not on PATH")

    try:
        import wandb
        ok("wandb.import", wandb.__version__)
    except ImportError as exc:
        fail("wandb.import", f"not installable ({exc}); pip install wandb")
        return

    if os.environ.get("WANDB_API_KEY"):
        ok("wandb.auth", "WANDB_API_KEY set")
    else:
        try:
            import wandb.api as wandb_api
            logged_in = wandb_api.api.api_key is not None
        except Exception:
            logged_in = False
        if logged_in:
            ok("wandb.auth", "logged in via ~/.netrc or wandb login")
        else:
            fail("wandb.auth", "no credentials — run `wandb login` or export WANDB_API_KEY")

    mode = os.environ.get("WANDB_MODE", "run")
    if mode != "run":
        warn("wandb.mode", f"WANDB_MODE={mode!r} — metrics will not reach the website")
    else:
        ok("wandb.mode", "run")


def check_repo():
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=30)
        dirty = [l for l in out.stdout.splitlines() if l.strip()]
        if dirty:
            warn("repo.clean", f"{len(dirty)} uncommitted change(s)")
        else:
            ok("repo.clean", "working tree clean")
    except Exception as exc:
        warn("repo.clean", f"could not read git state ({exc})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks before a paid training run.")
    parser.add_argument("--checksums", action="store_true",
                        help="sha256 every shard against manifest.json (slow, ~29 GiB)")
    args = parser.parse_args()

    print("=" * 70)
    print("LLaMA-3-Lite preflight doctor")
    print("=" * 70)

    cfg = check_config()
    if cfg is None:
        print("\nconfig unreadable — aborting")
        return 1
    check_gpu(cfg)
    check_data(cfg, do_checksums=args.checksums)
    check_cache(cfg)
    check_disk(cfg)
    check_tooling()
    check_repo()

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print("=" * 70)
    print(f"{len(RESULTS)} checks: {len(RESULTS) - len(fails) - len(warns)} pass, "
          f"{len(warns)} warn, {len(fails)} fail")
    if fails:
        print("\nDO NOT start the run. Fix these first:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
        return 1
    print("\nPreflight clean. Safe to start the run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
