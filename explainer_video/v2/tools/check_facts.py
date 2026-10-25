"""Fact gate — prove ``facts.json`` is current, correct, and internally consistent.

Three independent checks, all of which must pass:

1. **Reproducible** — rebuilding must produce byte-identical output, so the
   file can be regenerated on any machine without a spurious diff.
2. **True to source** — every spec value is re-read from ``config.py`` and the
   tokenizer, and the parameter count is recomputed by a second, independent
   formula (a typo in the builder cannot hide itself here).
3. **Self-consistent** — the attention matrices must satisfy their own
   definition: scaled = raw / √d_k, masked weights exactly 0, rows sum to 1,
   weights = softmax(masked scaled).

Usage:
    python tools/check_facts.py            # verify (exit 1 on any failure)
    python tools/check_facts.py --fix      # rebuild facts.json, then verify
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:  # imported as ``tools.check_facts``
    from tools import build_facts
except ImportError:  # run as ``python tools/check_facts.py``
    import build_facts

TOL = 2e-3  # matches the 4-decimal rounding in facts.json


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.checks += 1
        print(f"  ok   {name}{(' — ' + detail) if detail else ''}")

    def fail(self, name: str, detail: str) -> None:
        self.checks += 1
        self.failures.append(f"{name}: {detail}")
        print(f"  FAIL {name} — {detail}")

    def expect(self, name: str, actual, expected, detail: str = "") -> None:
        if actual == expected:
            self.ok(name, detail or f"{actual!r}")
        else:
            self.fail(name, f"{actual!r} != expected {expected!r}")


def _expected_params(cfg: dict, vocab: int) -> int:
    """Parameter count written independently of ``build_facts.params_breakdown``."""
    d = cfg["d_model"]
    hd = cfg["head_dim"]
    nh = cfg["n_heads"]
    nkv = cfg["n_kv_heads"]
    ff = cfg["d_ff"]
    layers = cfg["n_layers"]

    embeddings = 2 * vocab * d                    # input + untied output head
    attention = nh * hd * d + 2 * nkv * hd * d + d * nh * hd
    norms = 2 * hd + 2 * d                        # q_norm, k_norm, attn_norm, ffn_norm
    ffn = d * 2 * ff + ff * d                     # fused gate+up, then down
    return embeddings + layers * (attention + norms + ffn) + d


def check_reproducible(gate: Gate, facts: dict) -> None:
    print("[1] reproducible")
    if not build_facts.FACTS.exists():
        gate.fail("facts.json present", "missing — run tools/build_facts.py")
        return
    on_disk = build_facts.FACTS.read_text(encoding="utf-8")
    rebuilt = build_facts.dumps(build_facts.build())
    if on_disk == rebuilt:
        gate.ok("rebuild is byte-identical", f"{len(rebuilt)} bytes")
    else:
        import difflib
        diff = list(difflib.unified_diff(
            on_disk.splitlines(), rebuilt.splitlines(),
            fromfile="facts.json (on disk)", tofile="facts.json (rebuilt)", lineterm=""))
        gate.fail("rebuild is byte-identical",
                  f"{len(diff)} diff lines; first: {diff[2] if len(diff) > 2 else '?'}")


def check_true_to_source(gate: Gate, facts: dict, cfg: dict, tokenizer) -> None:
    print("[2] true to source (config.py + tokenizer)")
    spec = facts["spec"]
    for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim",
                "d_ff", "seq_len", "rms_norm_eps", "qknorm"):
        gate.expect(f"spec.{key}", spec[key], cfg[key])
    gate.expect("spec.vocab_config", spec["vocab_config"], cfg["vocab_size"])
    gate.expect("spec.rope_theta", float(spec["rope_theta"]), float(cfg["rope_theta"]))

    tok_vocab = int(tokenizer.get_vocab_size())
    expected_vocab = max(int(cfg["vocab_size"]), tok_vocab)
    gate.expect("vocab == max(config, tokenizer)", facts["vocab"], expected_vocab,
                f"{facts['vocab']:,} (config {cfg['vocab_size']:,}, tokenizer {tok_vocab:,})")
    gate.expect("vocab_label", facts["vocab_label"], f"{expected_vocab:,}")
    if "128K" in facts["vocab_label"] or facts["vocab_label"] == "128,000":
        gate.fail("vocab_label", "stale 128K/128,000 label — the run builds 128,256")

    demo = facts["tokens"]
    enc = tokenizer.encode(demo["sentence"], add_special_tokens=False)
    gate.expect("demo token ids", demo["ids"], [int(t) for t in enc.ids])
    gate.expect("demo token count", len(demo["items"]), len(enc.ids))
    for item, piece in zip(demo["items"], enc.tokens):
        gate.expect(f"token {item['text']!r} piece", item["piece"], piece)
    gate.expect("demo vocab", demo["vocab"], tok_vocab)

    params = _expected_params(cfg, expected_vocab)
    gate.expect("params (independent formula)", facts["params"], params,
                f"{params:,}")
    gate.expect("params_at_config_vocab", facts["params_at_config_vocab"],
                _expected_params(cfg, cfg["vocab_size"]))
    gate.expect("params_breakdown sums to params",
                facts["params_breakdown"]["total"], facts["params"])
    gate.expect("params label is the exact count rounded to 5M",
                facts["params_label"],
                f"≈{int(round(facts['params'] / 1e6 / 5) * 5)}M")

    gate.expect("rope pairs", facts["rope"]["pairs"], cfg["head_dim"] // 2)
    gate.expect("rope theta", float(facts["rope"]["theta"]), float(cfg["rope_theta"]))


def check_attention(gate: Gate, facts: dict) -> None:
    print("[3] attention demo is self-consistent")
    att = facts["attention"]
    raw, scaled, weights = att["raw"], att["scaled"], att["weights"]
    d_k = att["d_k"]
    scale = 1.0 / math.sqrt(d_k)
    n = len(raw)

    gate.expect("matrix is square", all(len(r) == n for r in raw) and len(raw), n)
    gate.expect("scale == 1/sqrt(d_k)", att["scale"], round(scale, 4))
    row = att["query_row"]
    live = [j for j in range(n) if weights[row - 1][j] != 0.0]
    gate.expect("query_row's visible keys are exactly 1..query_row",
                live, list(range(row)), f"row {row} attends to keys {live}")

    bad_scale = [(i, j) for i in range(n) for j in range(n)
                 if abs(scaled[i][j] - raw[i][j] * scale) > TOL]
    gate.ok("scaled == raw / sqrt(d_k)") if not bad_scale else gate.fail(
        "scaled == raw / sqrt(d_k)", f"{len(bad_scale)} cells off, e.g. {bad_scale[:3]}")

    masked = [(i, j) for i in range(n) for j in range(i + 1, n) if weights[i][j] != 0.0]
    gate.ok("masked keys get exactly 0.00") if not masked else gate.fail(
        "masked keys get exactly 0.00", f"{len(masked)} masked cells non-zero: {masked[:3]}")

    row_sums = [sum(r) for r in weights]
    bad_rows = [i for i, s in enumerate(row_sums) if abs(s - 1.0) > TOL]
    gate.ok("every row sums to 1", str([round(s, 3) for s in row_sums])) if not bad_rows \
        else gate.fail("every row sums to 1", f"rows {bad_rows} sum {[row_sums[i] for i in bad_rows]}")

    worst, where = 0.0, None
    for i in range(n):
        logits = [scaled[i][j] if j <= i else None for j in range(n)]
        m = max(v for v in logits if v is not None)
        exps = [(math.exp(v - m) if v is not None else 0.0) for v in logits]
        total = sum(exps)
        for j in range(n):
            delta = abs(weights[i][j] - exps[j] / total)
            if delta > worst:
                worst, where = delta, (i, j)
    gate.ok("weights == softmax(masked scaled)", f"max |Δ| {worst:.1e}") \
        if worst <= TOL else gate.fail("weights == softmax(masked scaled)",
                                       f"max |Δ| {worst:.1e} at {where}")

    gate.expect("row_weights == weights[query_row-1]",
                att["row_weights"], weights[att["query_row"] - 1])


def main(argv: list[str]) -> int:
    cfg = build_facts.load_config()
    tokenizer, tok_path, _ = build_facts.load_tokenizer(cfg)

    if "--fix" in argv:
        build_facts.FACTS.write_text(
            build_facts.dumps(build_facts.build()), encoding="utf-8")
        print(f"rebuilt {build_facts.FACTS.relative_to(build_facts.REPO)}")

    try:
        shown = "~/" + str(tok_path.relative_to(Path.home()))
    except ValueError:
        shown = str(tok_path)
    print(f"facts.json · tokenizer {shown} · config.py")
    gate = Gate()
    if not build_facts.FACTS.exists():
        print("FAIL facts.json missing — run python tools/build_facts.py")
        return 1
    facts = json.loads(build_facts.FACTS.read_text(encoding="utf-8"))

    check_reproducible(gate, facts)
    check_true_to_source(gate, facts, cfg, tokenizer)
    check_attention(gate, facts)

    print()
    if gate.failures:
        print(f"FACTS GATE FAILED — {len(gate.failures)} of {gate.checks} checks")
        for failure in gate.failures:
            print(f"  · {failure}")
        return 1
    print(f"FACTS GATE PASSED — {gate.checks} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
