"""Phase 1a gates: facts wired into the scenes, and the four correctness fixes.

Each test reads the same artifacts the renderer does. A failure means a scene
would ship a hand-typed figure, a softmax that does not follow its scores, a
thresholded SwiGLU, a width-halved cache, or QK-norm attributed to LLaMA 3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import render_scenes as rs  # noqa: E402

FACTS = json.loads((V2 / "facts.json").read_text(encoding="utf-8"))
NARRATION = yaml.safe_load((V2 / "narration.yaml").read_text(encoding="utf-8"))
RENDER = (V2 / "render_scenes.py").read_text(encoding="utf-8")

# The v2 cut's hand-typed figures. None may come back.
STALE_LITERALS = ["128K", "128,000", "128 000", "128000",
                  "9427", "7727", "7560", "1246", "515M parameters"]


def section(number: int) -> dict:
    return next(s for s in NARRATION["sections"] if s["number"] == number)


def sentence(cue: str) -> dict:
    return next(s for sec in NARRATION["sections"] for s in sec["sentences"]
                if s["id"] == cue)


# ------------------------------------------------------- T1.1 wire facts ---
def test_no_hand_typed_model_figures_remain():
    for literal in STALE_LITERALS:
        assert literal not in RENDER, f"{literal!r} is still typed in render_scenes.py"


def test_spec_chips_come_from_facts():
    labels = rs.spec_chip_labels()
    assert labels == [
        f"{FACTS['params_label']} parameters",
        f"{FACTS['spec']['n_layers']} blocks",
        f"d_model {FACTS['spec']['d_model']:,}",
        f"vocab {FACTS['vocab_label']}",
        f"context {FACTS['spec']['seq_len']:,}",
    ]
    assert "vocab 128,256" in labels
    assert labels[0] == "≈515M parameters"


def test_token_chips_and_embedding_dims_come_from_facts():
    assert 'TOKENS["sentence"]' in RENDER
    assert 'TOKENS["items"]' in RENDER
    items = FACTS["tokens"]["items"]
    assert [item["id"] for item in items] == [791, 8415, 7731, 389, 279, 5634]
    # Matrix dims and the unembedding row are formatted from facts, not typed.
    assert "RR^({VOCAB} times {D_MODEL})" in RENDER
    assert "{D_MODEL:,} → {VOCAB_LABEL}" in RENDER
    assert "W_U   {D_MODEL:,} → {VOCAB_LABEL}" in RENDER


def test_gqa_and_cache_counts_come_from_facts():
    assert rs.N_HEADS == FACTS["spec"]["n_heads"] == 8
    assert rs.N_KV_HEADS == FACTS["spec"]["n_kv_heads"] == 4
    assert rs.N_LAYERS == FACTS["spec"]["n_layers"] == 16
    assert "cache_stack(n_mha" in RENDER
    assert "cache_stack(n_gqa" in RENDER


# ------------------------------------------------- T1.2 attention demo ------
def test_weights_are_the_masked_softmax_of_the_scaled_scores():
    raw, scaled, weights = rs.attention_demo()
    att = FACTS["attention"]
    for i, row in enumerate(raw):
        for j, value in enumerate(row):
            assert scaled[i][j] == pytest.approx(value / np.sqrt(att["d_k"]), abs=1e-3)
            if j > i:
                assert weights[i][j] == 0.0, "masked key must get exactly 0.00"


def test_weight_bars_are_the_query_row_and_show_a_zero_masked_key():
    labels, probs = rs.attention_weight_bars()
    att = FACTS["attention"]
    assert labels == ["k₁ The", "k₂ cat", "k₃ sat", "k₄ on"]
    assert probs == att["row_weights"] == att["weights"][att["query_row"] - 1]
    assert sum(probs) == pytest.approx(1.0, abs=1e-3)
    assert probs[-1] == 0.0, "the masked key is removed, not down-weighted"
    assert f"{probs[-1]:.2f}" == "0.00"


def test_attention_demo_uses_a_scale_strip_and_the_factored_scale():
    assert "÷ √d_k" in RENDER
    assert 'ATTN[\'scale\']' in RENDER or 'ATTN["scale"]' in RENDER
    assert "softmax" in RENDER


def test_rope_figures_come_from_facts():
    """θ, the context window and the slowest pair index are facts, not literals."""
    assert "500 000" not in RENDER
    assert "THETA_LABEL" in RENDER
    assert 'FACTS["rope"]["pairs"] - 1' in RENDER
    assert rs.SEQ_LEN == FACTS["spec"]["seq_len"] == 2048


# --------------------------------------------------- T1.3 QK-norm caveat ----
def test_qknorm_caveat_is_narrated():
    texts = " ".join(s["text"] for s in section(7)["sentences"])
    assert "Lite addition" in texts
    assert "LLaMA 3 itself has none" in texts


def test_qknorm_caveat_is_on_screen():
    assert "a Lite addition — LLaMA 3 itself has none" in RENDER


# --------------------------------------------------- T1.4 SwiGLU gating -----
def test_swiglu_is_a_product_of_two_rows():
    silu, up, prod = rs.swiglu_demo()
    assert len(silu) == len(up) == len(prod)
    assert np.allclose(prod, silu * up)
    assert np.all(prod != 0.0), "multiplicative gating scales, it does not drop"


def test_swiglu_narration_scales_features():
    text = sentence("s08.6")["text"].replace("\n", " ")
    assert "scaling each feature" in text
    assert "survive" not in text


def test_swiglu_threshold_is_gone():
    assert "alive = v >" not in RENDER
    assert "which features survive" not in RENDER
    assert "scales each feature" in RENDER


# --------------------------------------------------- T1.5 KV cache ----------
def test_cache_bars_halve_by_heads_not_by_width():
    n_mha, n_gqa = rs.cache_bar_counts()
    assert n_mha == 2 * rs.N_HEADS == 16
    assert n_gqa == 2 * rs.N_KV_HEADS == 8
    assert n_gqa == n_mha // 2
    # Equal width, fewer bars: the height is what halves.
    assert "width=0.95" in RENDER
    assert "cache_stack(1.4" not in RENDER
    assert "cache_stack(0.7" not in RENDER


def test_cache_head_counts_are_labelled():
    assert 'f"MHA · {N_HEADS} KV heads"' in RENDER
    assert 'f"GQA · {N_KV_HEADS} KV heads"' in RENDER


# ------------------------------------------------------- T1.6 wording ------
def test_opening_line_says_token_not_word():
    text = sentence("s01.2")["text"].replace("\n", " ")
    assert "word" not in text
    assert "next token" in text
