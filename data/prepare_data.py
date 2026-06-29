"""LLaMA-3-Lite data preparation: thin shim over the universal pipeline."""
import argparse
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent.parent  # .../CoreProjects/
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


LLAMA3_TOKENIZER_NAME = "llama3"
LLAMA3_VOCAB_SIZE = 128_000
LLAMA3_EOS_TOKEN_ID = 128_009
LLAMA3_PAD_TOKEN_ID = 128_002


def _apply_llama3_defaults() -> None:
    from shared_data.config import UNIVERSAL_TOTAL_TOKENS
    print(f"[data/llama3] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
    print(f"[data/llama3] tokenizer: {LLAMA3_TOKENIZER_NAME} "
          f"(vocab={LLAMA3_VOCAB_SIZE:,}, EOS={LLAMA3_EOS_TOKEN_ID})")
    print(f"[data/llama3] shard size: 50,000,000 tokens (uint32)")
    print(f"[data/llama3] note: identical byte layout to GPT-OSS-Lite; "
          f"shards can be shared verbatim between the two projects.")


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

    _apply_llama3_defaults()

    from shared_data.config import UNIVERSAL_MIXTURE_PATH, UNIVERSAL_DATA_CONFIG_PATH
    from shared_data.prepare_data import run_pipeline

    return run_pipeline(
        mixture_path=Path(args.mixture) if args.mixture else UNIVERSAL_MIXTURE_PATH,
        data_config_path=Path(args.data_config) if args.data_config else UNIVERSAL_DATA_CONFIG_PATH,
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        data_root=Path(args.data_root) if args.data_root else None,
    )


if __name__ == "__main__":
    sys.exit(main())