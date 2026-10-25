"""Build ``facts.json`` — every number the explainer puts on screen.

Sources of truth, in order of authority:

1. ``config.py:get_config()`` — architecture and hyperparameters.
2. The LLaMA-3 tokenizer — vocabulary size and the demo token IDs.
3. A seeded demo tensor pair — the attention walk-through. It is marked
   ``illustrative``: the values are not model measurements, only a
   self-consistent raw → scaled → masked-softmax chain.

The output is deterministic: identical inputs produce a byte-identical file
(no timestamps, sorted keys, fixed rounding). ``tools/check_facts.py`` is the
gate that proves it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
REPO = V2.parent.parent
FACTS = V2 / "facts.json"

TOKENIZER_ENV = "LLAMA3_TOKENIZER_JSON"
ATTENTION_SEED = 7
DEMO_SENTENCE = "The cat sat on the mat"
ROUND = 4

# The published headline rounds the exact count to the nearest 5 M; the exact
# integer in facts.json is authoritative and the label is cosmetic.
PARAMS_LABEL = "≈515M"


def load_config() -> dict:
    """Import the repository's ``config.py`` without touching ``sys.path``."""
    spec = importlib.util.spec_from_file_location("llama3lite_config",
                                                 REPO / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def find_tokenizer_json(cfg: dict) -> Path:
    """Locate ``tokenizer.json``: explicit override, config dir, then HF cache."""
    candidates = []
    env = os.environ.get(TOKENIZER_ENV)
    if env:
        candidates.append(Path(env).expanduser())
    if cfg.get("tokenizer_cache_dir"):
        candidates.append(Path(cfg["tokenizer_cache_dir"]).expanduser()
                          / "tokenizer.json")
    hub_dir = (Path.home() / ".cache" / "huggingface" / "hub"
               / ("models--" + cfg["tokenizer_name"].replace("/", "--")))
    candidates.extend(sorted(hub_dir.glob("snapshots/*/tokenizer.json")))
    for path in candidates:
        if path.is_file():
            return path.resolve()

    try:  # offline-first, but tolerate a cold cache on a fresh machine
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(cfg["tokenizer_name"], "tokenizer.json"))
    except Exception as exc:  # pragma: no cover - network-dependent
        raise SystemExit(
            f"tokenizer.json not found. Set {TOKENIZER_ENV}=/path/to/tokenizer.json "
            f"or pre-download {cfg['tokenizer_name']}.\n({exc})")


def load_tokenizer(cfg: dict):
    from tokenizers import Tokenizer

    path = find_tokenizer_json(cfg)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Tokenizer.from_file(str(path)), path, digest


def params_breakdown(cfg: dict, vocab: int) -> dict:
    """Parameter count straight from the module shapes in ``model.py``."""
    d = cfg["d_model"]
    q_dim = cfg["n_heads"] * cfg["head_dim"]
    kv_dim = cfg["n_kv_heads"] * cfg["head_dim"]
    ffn = cfg["d_ff"]
    head_dim = cfg["head_dim"]

    per_block = {
        "q_proj": d * q_dim,
        "k_proj": d * kv_dim,
        "v_proj": d * kv_dim,
        "out_proj": q_dim * d,
        "q_norm": head_dim,
        "k_norm": head_dim,
        "attention_norm": d,
        "ffn_norm": d,
        "gate_up_proj": d * 2 * ffn,
        "down_proj": ffn * d,
    }
    per_block_total = sum(per_block.values())
    top_level = {
        "input_embedding": vocab * d,
        "output_proj": vocab * d,
        "decoder_norm": d,
    }
    total = per_block_total * cfg["n_layers"] + sum(top_level.values())
    return {
        "per_block": per_block,
        "per_block_total": per_block_total,
        "layers_total": per_block_total * cfg["n_layers"],
        "top_level": top_level,
        "total": total,
    }


def token_demo(tokenizer, sentence: str = DEMO_SENTENCE) -> dict:
    """Tokenize the demo sentence; ``piece`` keeps the space marker, ``text`` does not."""
    enc = tokenizer.encode(sentence, add_special_tokens=False)
    items = []
    for piece, tid in zip(enc.tokens, enc.ids):
        items.append({
            "piece": piece,
            "text": piece.replace("\u0120", "").strip(),
            "id": int(tid),
            "lead_space": piece.startswith("\u0120"),
        })
    return {"sentence": sentence, "items": items,
            "ids": [it["id"] for it in items],
            "vocab": int(tokenizer.get_vocab_size())}


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _round(matrix: np.ndarray) -> list:
    return [[round(float(v), ROUND) for v in row] for row in matrix]


def attention_demo(cfg: dict, seed: int = ATTENTION_SEED) -> dict:
    """Raw scores → ÷√d_k → causal-masked softmax, from one seeded Q/K pair.

    The mask is applied *before* the softmax (upper triangle → -inf), so masked
    weights come out exactly 0.0 instead of being renormalised afterwards.
    """
    d_k = cfg["head_dim"]
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((4, d_k))
    k = rng.standard_normal((4, d_k))
    raw = q @ k.T
    scale = 1.0 / math.sqrt(d_k)
    scaled = raw * scale
    masked = scaled.copy()
    for i in range(4):
        masked[i, i + 1:] = -np.inf
    weights = _softmax(masked)
    return {
        "illustrative": True,
        "seed": seed,
        "d_k": d_k,
        "scale": round(scale, ROUND),
        "mask": "upper",
        "query_row": 3,
        "raw": _round(raw),
        "scaled": _round(scaled),
        "weights": _round(weights),
        "row_weights": [round(float(v), ROUND) for v in weights[2]],
    }


def rope_table(cfg: dict) -> dict:
    """Pair count and the fast / medium / slow frequencies used by the scene."""
    head_dim = cfg["head_dim"]
    theta = float(cfg["rope_theta"])
    pairs = head_dim // 2
    samples = []
    for pair in (0, 7, pairs - 1):
        inv_freq = theta ** (-2.0 * pair / head_dim)
        samples.append({
            "pair": pair,
            "inv_freq": round(inv_freq, 6),
            "period_tokens": round(2 * math.pi / inv_freq, ROUND),
        })
    return {"pairs": pairs, "theta": theta, "samples": samples}


def build() -> dict:
    cfg = load_config()
    tokenizer, tok_path, tok_digest = load_tokenizer(cfg)
    demo = token_demo(tokenizer)

    # train.py sizes the model to max(config, len(tokenizer)); 128,256 wins.
    vocab = max(int(cfg["vocab_size"]), int(tokenizer.get_vocab_size()))
    params = params_breakdown(cfg, vocab)

    return {
        "schema": 1,
        "vocab": vocab,
        "vocab_label": f"{vocab:,}",
        "params": params["total"],
        "params_label": PARAMS_LABEL,
        "spec": {
            "d_model": cfg["d_model"],
            "n_layers": cfg["n_layers"],
            "n_heads": cfg["n_heads"],
            "n_kv_heads": cfg["n_kv_heads"],
            "head_dim": cfg["head_dim"],
            "d_ff": cfg["d_ff"],
            "seq_len": cfg["seq_len"],
            "rope_theta": float(cfg["rope_theta"]),
            "rms_norm_eps": cfg["rms_norm_eps"],
            "vocab_config": cfg["vocab_size"],
            "qknorm": bool(cfg["qknorm"]),
        },
        "params_breakdown": params,
        "params_at_config_vocab": params_breakdown(cfg, cfg["vocab_size"])["total"],
        "tokens": demo,
        "attention": attention_demo(cfg),
        "rope": rope_table(cfg),
        "source": {
            "config": "config.py:get_config",
            "model": "model.py",
            "tokenizer": cfg["tokenizer_name"],
            "tokenizer_sha256": tok_digest,
        },
    }


def dumps(facts: dict) -> str:
    """Canonical serialisation — the contract behind the byte-identical gate."""
    return json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    facts = build()
    FACTS.write_text(dumps(facts), encoding="utf-8")
    rel = FACTS.relative_to(REPO)
    print(f"wrote {rel} · vocab {facts['vocab_label']} · "
          f"params {facts['params']:,} · {len(facts['tokens']['items'])} demo tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
