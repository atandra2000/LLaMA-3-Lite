"""Phase 0 gates: facts, narration single source, and the type floor.

These are the machine-checkable contracts from PLAN.md Phase 0. They read the
same artifacts the renderer does, so a failure here means a render would ship a
wrong number, a stale script, or an illegible label.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest
import yaml

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

from tools import build_facts, build_script_doc, check_facts, lint_script  # noqa: E402

FACTS = V2 / "facts.json"
NARRATION = V2 / "narration.yaml"
RENDER = V2 / "render_scenes.py"
FONT_DIR = V2 / "assets" / "fonts"


@pytest.fixture(scope="module")
def facts() -> dict:
    if not FACTS.exists():
        pytest.skip("facts.json missing — run tools/build_facts.py")
    import json
    return json.loads(FACTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def narration() -> dict:
    return yaml.safe_load(NARRATION.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ facts ---
def test_facts_rebuild_is_byte_identical(facts):
    """Deleting facts.json and rebuilding must not change a byte."""
    rebuilt = build_facts.dumps(build_facts.build())
    assert rebuilt == FACTS.read_text(encoding="utf-8")
    assert build_facts.dumps(build_facts.build()) == rebuilt


def test_fact_gate_passes():
    assert check_facts.main([]) == 0


def test_vocab_is_the_tokenizer_vocab(facts):
    """C3: the run builds 128,256, not the config's 128,000."""
    assert facts["vocab"] == 128_256
    assert facts["spec"]["vocab_config"] == 128_000
    assert facts["vocab_label"] == "128,256"
    assert "128K" not in facts["vocab_label"]


def test_demo_token_ids_are_real(facts):
    """C1: the six IDs the token chips show."""
    assert facts["tokens"]["ids"] == [791, 8415, 7731, 389, 279, 5634]
    assert [t["text"] for t in facts["tokens"]["items"]] == [
        "The", "cat", "sat", "on", "the", "mat"]


def test_param_counts(facts):
    assert facts["params"] == 514_364_416          # at vocab 128,256
    assert facts["params_at_config_vocab"] == 513_840_128   # at vocab 128,000
    assert facts["params_breakdown"]["total"] == facts["params"]


def test_attention_weights_follow_the_scores(facts):
    """C4: weights are the masked softmax of the scaled scores, and the
    masked key gets exactly 0.00."""
    att = facts["attention"]
    raw, scaled, weights = att["raw"], att["scaled"], att["weights"]
    scale = 1.0 / math.sqrt(att["d_k"])
    for i, row in enumerate(raw):
        for j, value in enumerate(row):
            assert scaled[i][j] == pytest.approx(value * scale, abs=1e-3)
            if j > i:      # causal mask: future keys are gone, not down-weighted
                assert weights[i][j] == 0.0
    for row in weights:
        assert sum(row) == pytest.approx(1.0, abs=1e-3)


# -------------------------------------------------------------- narration ---
def test_script_lint_is_clean(narration):
    errors, _, stats = lint_script.lint(narration)
    assert errors == []
    assert stats["max"] <= lint_script.MAX_WORDS


def test_section_five_is_tight(narration):
    section = narration["sections"][4]
    assert section["number"] == 5
    for sentence in section["sentences"]:
        assert lint_script.word_count(sentence["text"]) <= lint_script.STRICT_MAX_WORDS


def test_cue_ids_are_unique_and_ordered(narration):
    ids = [s["id"] for sec in narration["sections"] for s in sec["sentences"]]
    assert len(ids) == len(set(ids))
    assert ids[0] == "s01.1"
    for section in narration["sections"]:
        prefix = f"s{section['number']:02d}."
        assert [s["id"] for s in section["sentences"]] == [
            f"{prefix}{i}" for i in range(1, len(section["sentences"]) + 1)]


def test_script_doc_is_generated_and_current():
    assert build_script_doc.build() == (V2 / "VOICEOVER_SCRIPT.md").read_text(
        encoding="utf-8")


def test_audio_generator_reads_the_single_source(narration):
    """P3: narration lives in narration.yaml, not in the TTS script."""
    source = (V2 / "generate_audio.py").read_text(encoding="utf-8")
    assert "load_narration()" in source
    assert "Every large language model" not in source

    from generate_audio import SECTIONS
    expected = [(s["id"], " ".join(x["text"].strip() for x in s["sentences"]))
                for s in narration["sections"]]
    assert SECTIONS == expected


# ------------------------------------------------------------------ fonts ---
FONT_FILES = [
    "Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf",
    "BarlowSemiCondensed-Regular.ttf", "BarlowSemiCondensed-Medium.ttf",
    "BarlowSemiCondensed-Bold.ttf",
]


def test_fonts_are_bundled_with_licences():
    for name in FONT_FILES:
        assert (FONT_DIR / name).is_file(), f"missing {name}"
    assert (FONT_DIR / "OFL-Inter.txt").is_file()
    assert (FONT_DIR / "OFL-BarlowSemiCondensed.txt").is_file()
    assert (FONT_DIR / "CREDITS.md").is_file()


def test_fonts_register_and_render():
    """Cross-machine render: the bundled faces must resolve through Pango."""
    import manimpango
    from manim import Text

    for path in sorted(FONT_DIR.glob("*.ttf")):
        manimpango.register_font(str(path))
    for family in ("Inter", "Barlow Semi Condensed"):
        text = Text("LLaMA-3-Lite 0123", font=family, font_size=24)
        assert text.width > 0


# ------------------------------------------------------------- type floor ---
TEXT_CALLS = {"T", "DIN_T", "TY", "Text", "MathTypst", "pill", "token_pill",
              "bars", "head_badge"}


def _sizes(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in TEXT_CALLS:
            continue
        for kw in node.keywords:
            if kw.arg not in ("size", "font_size"):
                continue
            value = kw.value
            if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                value = value.operand
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                yield node.lineno, value.value


def test_no_text_below_the_legibility_floor():
    tree = ast.parse(RENDER.read_text(encoding="utf-8"))
    offenders = [(line, size) for line, size in _sizes(tree)
                 if size < 12 and size > 0]
    assert offenders == [], f"labels below 12 pt: {offenders[:10]}"


def test_type_scale_obeys_the_floor_and_is_ordered():
    source = RENDER.read_text(encoding="utf-8")
    assert "MIN_LABEL_SIZE = 12" in source
    import render_scenes as rs

    assert rs.MIN_LABEL_SIZE == 12
    values = list(rs.TYPE.values())
    assert min(values) >= rs.MIN_LABEL_SIZE
    assert values == sorted(values)
    with pytest.raises(ValueError):
        rs.pt(rs.MIN_LABEL_SIZE - 1)
