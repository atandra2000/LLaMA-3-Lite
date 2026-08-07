"""LLaMA-3-Lite data preparation: thin shim over the universal pipeline."""
import argparse
import json
import os
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _PROJECT_ROOT.parent
for _p in (_DATA_ROOT, _PROJECT_ROOT, _WORKSPACE_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


LLAMA3_TOKENIZER_NAME = "llama3"
LLAMA3_VOCAB_SIZE = 128_000
LLAMA3_EOS_TOKEN_ID = 128_009
LLAMA3_PAD_TOKEN_ID = 128_002

# Bound each streaming read so concatenation stays flat in host memory.
_CONCAT_CHUNK_TOKENS = 1 << 26


def _apply_llama3_defaults() -> None:
    from shared_data.config import UNIVERSAL_TOTAL_TOKENS
    print(f"[data/llama3] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
    print(f"[data/llama3] tokenizer: {LLAMA3_TOKENIZER_NAME} "
          f"(vocab={LLAMA3_VOCAB_SIZE:,}, EOS={LLAMA3_EOS_TOKEN_ID})")
    print(f"[data/llama3] shard size: 50,000,000 tokens (uint32)")
    print(f"[data/llama3] note: identical byte layout to GPT-OSS-Lite; "
          f"shards can be shared verbatim between the two projects.")


def concat_shards_to_cache(shards_dir: Path, manifest_path: Path,
                           cache_path: Path) -> int:
    """Concatenate packed shards (manifest order) into the flat ``tokens.bin`` cache.

    The workspace pipeline emits ``data/shards/shard_*.bin`` + ``manifest.json``;
    the vendored loader mmaps a single flat ``uint32`` stream instead. This
    stage bridges the two: concatenating shards in manifest order yields
    exactly the byte stream ``data/shared_data/loader.py:build_training_data``
    expects. Streams shard-by-shard so RAM stays flat; writes atomically via a
    sibling ``.tmp`` + ``os.replace``.
    """
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest at {manifest_path}. Run the pipeline without "
            f"--skip-pack first: the pack stage produces the shards + "
            f"manifest this stage concatenates."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = sorted(manifest["shards"], key=lambda s: int(s["index"]))
    if not shards:
        raise SystemExit(f"Manifest at {manifest_path} lists no shards.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    total = 0
    with open(tmp, "wb") as out:
        for shard in shards:
            p = shards_dir / shard["path"]
            with open(p, "rb") as f:
                while True:
                    buf = f.read(_CONCAT_CHUNK_TOKENS * 4)
                    if not buf:
                        break
                    out.write(buf)
                    total += len(buf) // 4
    os.replace(tmp, cache_path)
    print(f"[data] concatenated {len(shards)} shards -> {cache_path} "
          f"({total:,} tokens, {total * 4 / 1e9:.1f} GB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLaMA-3-Lite data prep (delegates to universal pipeline)",
    )
    parser.add_argument("--stage", choices=["pretrain"], default="pretrain")
    parser.add_argument("--mixture", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()

    try:
        _apply_llama3_defaults()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "LLaMA-3-Lite data prep delegates to the universal pipeline at "
            "`LLM/shared_data/` (shared_data.config / shared_data.prepare_data). "
            "That workspace package is not importable on this machine "
            f"({exc}). This project vendors only the loader (data/shared_data/)."
        ) from exc

    from shared_data.config import UNIVERSAL_MIXTURE_PATH, UNIVERSAL_DATA_CONFIG_PATH
    from shared_data.prepare_data import run_pipeline

    rc = run_pipeline(
        mixture_path=Path(args.mixture) if args.mixture else UNIVERSAL_MIXTURE_PATH,
        data_config_path=Path(args.data_config) if args.data_config else UNIVERSAL_DATA_CONFIG_PATH,
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        data_root=Path(args.data_root) if args.data_root else None,
    )
    if rc != 0:
        return rc

    # Convert the pipeline's shard layout to the loader's flat cache layout.
    from shared_data.common import DATA_ROOT

    # Resolve the shard directory from the manifest, relative to DATA_ROOT.
    probe = DATA_ROOT / "shards" / "manifest.json"
    shards_dir = DATA_ROOT / json.loads(
        probe.read_text(encoding="utf-8")).get("shards_dir", "shards")
    manifest_path = shards_dir / "manifest.json"

    from config import get_config
    cfg = get_config()
    cache_path = Path(cfg["data_cache_dir"]) / cfg["data_cache_filename"]
    if cfg.get("reuse_data_cache", True) and cache_path.exists():
        print(f"[data] cache already exists at {cache_path} "
              f"(reuse_data_cache=True); skipping concat.")
        return 0
    return concat_shards_to_cache(shards_dir, manifest_path, cache_path)


if __name__ == "__main__":
    sys.exit(main())