"""Phase 1b gates: the layout defects and the safe-rect guard.

Each test maps to a defect the v2 cut shipped — a clipped ε pill, the formula
card on the LayerNorm chip, the attention banner over its own header, a filled
loop wedge over the logits, the `× 16 blocks` pill on the band edge, the orphan
RoPE grid, a watermark on every scene, a caption below the frame. The guard's
own cases run on constructed mobjects, so they need no renderer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

V2 = Path(__file__).resolve().parents[1]
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))

import render_scenes as rs  # noqa: E402
from manim import VGroup  # noqa: E402

RENDER = (V2 / "render_scenes.py").read_text(encoding="utf-8")


# ------------------------------------------------------- T1.12 guard --------
def test_safe_rect_is_the_planned_rect():
    assert rs.SAFE == (-6.7, 6.7, -3.6, 3.6)


def test_guard_catches_text_outside_the_rect():
    below = rs.T("clipped ε pill", size=14).move_to([0, -3.9, 0])
    inside = rs.T("fine", size=14).move_to([0, 0, 0])
    (hit,) = rs.safe_violations([VGroup(below)])
    assert hit["text"] == "clipped ε pill"
    assert hit["over"]["bottom"] > 0.2
    assert rs.safe_violations([VGroup(inside)]) == []


def test_guard_covers_typst_formulas():
    """MathTypst is a different leaf class from Text; both must be checked."""
    formula = rs.TY("x + y", size=20).move_to([7.3, 0, 0])
    (hit,) = rs.safe_violations([formula])
    assert hit["over"]["right"] > 0.4


def test_overlap_log_reports_partial_not_contained():
    a = rs.pill("alpha").move_to([0, 0, 0])
    b = rs.pill("beta").move_to([0.6, 0, 0])            # partial
    assert len(rs.group_overlaps([a, b])) == 1
    outer = rs.pill("a much longer pill").move_to([0, 0, 0])
    inner = rs.pill("x").move_to(outer)                 # container, not a clash
    assert rs.group_overlaps([outer, inner]) == []
    assert rs.group_overlaps([rs.card(2, 1), rs.card(2, 1).move_to([1.2, 0, 0])]) == []


def test_guard_raises_instead_of_only_recording():
    body = RENDER.split("def sync_end", 1)[1]
    assert "raise RuntimeError(" in body
    assert 'os.environ.get("CHECK_LAYOUT")' in RENDER


def test_check_layout_flag_reaches_the_scene():
    assert "--check_layout" in RENDER
    assert 'os.environ["CHECK_LAYOUT"] = "1"' in RENDER
    assert 'self._layout_check(f"cue {sid}")' in RENDER


# ------------------------------------------------------- T1.7 §4 ------------
@pytest.mark.skip(reason="R5 re-cue: the RMSNorm unit-scale sketch left the "
                         "locked script; safe_violations now guards every caption")
def test_section4_geometry_caption_stays_in_frame():
    assert rs.GEO_SIZE <= 1.6
    assert "geo.move_to([-5.0, -2.5, 0])" in RENDER
    assert "geo_cap.next_to(geo, DOWN, buff=0.10)" in RENDER


def test_section4_drops_the_dead_circumscribe():
    """mean_chip has already become rms_chip by that beat."""
    assert "Circumscribe(mean_chip" not in RENDER
    assert "Circumscribe(rms_chip" in RENDER


def test_section4_root_sits_on_the_arrow():
    arrow = rs.dim_arrow([0, 1.0, 0], [0, 0.2, 0], op=0.4)
    root = rs.operator_on_arrow("√", arrow)
    assert np.allclose(root[1].get_center(), arrow.get_center(), atol=1e-6)
    assert 'operator_on_arrow("√", mean2rms' in RENDER


# ------------------------------------------------------- T1.8 §5 ------------
def test_attention_header_yields_to_the_banner():
    assert "hdr = self.section_header(" in RENDER
    assert "FadeOut(hdr)" in RENDER
    assert "banner_card.move_to([0, 2.45, 0])" in RENDER


def test_attention_captions_are_clear_of_the_banner():
    assert "wcap.next_to(wbars, DOWN" in RENDER
    # hcap under the heads, on two lines, not to the right of them.
    assert 'hcap = T("independent projections → concat\\n→ W_O"' in RENDER
    assert "hcap.next_to(heads, DOWN" in RENDER


# ------------------------------------------------------- T1.9 §9 ------------
def test_loop_is_stroke_only_and_routed_above():
    assert "CurvedArrow(mat_chip.get_top(), ctx[0].get_top()" in RENDER
    assert "angle=0.64" in RENDER
    assert "fill_opacity=0)" in RENDER
    assert "loop.tip.set_fill(ROSE, 0.95)" in RENDER


def test_probability_bars_survive_the_chip():
    assert "FadeOut(pb[0][1])" not in RENDER
    assert "FadeOut(pb[0][3])" not in RENDER
    assert "pb[0][2].copy()" in RENDER
    assert "pb[0][0].copy()" in RENDER


# ---------------------------------------------------- T1.10 sweep ----------
def test_blocks_pill_is_off_the_band_edge():
    """The × blocks pill must sit clear of the residual rail it counts."""
    assert "x16.move_to([3.90, 2.35, 0])" in RENDER


def test_ffn_card_is_above_the_safe_floor():
    assert "fx_card.move_to([3.45, -3.1, 0])" in RENDER


def test_qk_strip_moved_up():
    assert "strip_y = -2.15" in RENDER


def test_header_rule_is_not_stretched_to_nothing():
    assert "rule.stretch(0.01" not in RENDER


def test_opening_ring_marks_block_one():
    assert "ring.move_to(stack[0])" in RENDER
    assert "stack[11]" not in RENDER


def test_recap_brace_spans_the_repeated_pills():
    assert "Brace(VGroup(row1[2], row1[4]), UP" in RENDER


def test_watermark_only_on_the_end_card():
    chrome = RENDER.split("def _build_chrome", 1)[1].split("def watermark", 1)[0]
    assert "wm" not in chrome
    assert "self._chrome.add(tag, wm" not in RENDER
    assert "wm = self.watermark()" in RENDER


def test_progress_bar_has_chapter_ticks():
    assert "py = -3.85" in RENDER
    assert "one tick per chapter boundary" in RENDER


# ------------------------------------------------------- T1.11 §6 ----------
def test_orphan_rope_grid_is_gone():
    assert "heat = grid(3, 3" not in RENDER
    assert "GrowFromCenter(c) for c in heat" not in RENDER


def test_position_sweeps_zero_to_seven_continuously():
    assert "DEMO_LAST_M = 7" in RENDER
    assert "mtrack.animate(rate_func=linear).set_value(DEMO_LAST_M)" in RENDER
    assert "for m in (1, 2, 3)" not in RENDER


def test_period_labels_come_from_facts():
    assert 'FACTS["rope"]["samples"]' in RENDER
    assert "period_label(sample['period_tokens'])" in RENDER
    assert rs.period_label(6.2832) == "6.3"
    assert rs.period_label(26.394) == "26"
    assert rs.period_label(2559195.5174) == "2.6M"
