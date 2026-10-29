"""LLaMA-3-Lite architecture explainer — v3 flat-math design system.

Flat black canvas, saturated manim-classic accents (blue / gold / teal /
purple / red), white and grey type, flat fills, thin outlined panels.
No gradient, glow-field, or glass chrome — that look read as AI-generated.
Bundled Inter display type + Barlow Semi Condensed technical labels.

Every scene fills exactly its time budget:
    budget = narration + LEAD + TAIL
(Render and assemble steps unchanged — see README.md)
"""
import json
import os
from pathlib import Path

import manimpango
import numpy as np
from manim import (BLACK, DL, DOWN, DR, LEFT, RIGHT, TAU, UP, UL, UR,
                   Arc, ArcBetweenPoints, Axes, Arrow, Brace, Circle, Circumscribe,
                   Create, CurvedArrow, MoveAlongPath,
                   AnimationGroup, DashedVMobject, Dot, FadeIn, FadeOut, Flash,
                   GrowFromCenter, GrowFromEdge,
                   LaggedStart, Line, MathTypst, ORIGIN, Rectangle,
                   ReplacementTransform, Rotate,
                   RoundedRectangle, Scene, ShowPassingFlash, Square, Text,
                   Transform, ValueTracker, VGroup, VMobject, Wait, Write,
                   always_redraw,
                   config, interpolate_color, linear, there_and_back)


class Indicate(Transform):
    """Indicate without destroying dark/black background fills on composite groups."""
    def __init__(self, mobject, scale_factor=1.1, color=None, rate_func=None, **kwargs):
        self.color = color or GOLD
        self.scale_factor = scale_factor
        super().__init__(mobject, rate_func=rate_func or there_and_back, **kwargs)

    def create_target(self):
        target = self.mobject.copy()
        target.scale(self.scale_factor)
        for orig, sm in zip(self.mobject.get_family(), target.get_family()):
            if isinstance(sm, VGroup):
                continue
            if hasattr(orig, "get_fill_color") and orig.get_fill_color().to_hex() in ("#000000", "#0b0e14"):
                sm.set_stroke(color=self.color)
            else:
                sm.set_color(self.color)
        return target

# ---------------------------------------------------- 3b1b classic palette -
# manim's default colors: saturated marks on pure black, like 3Blue1Brown.
WHITE = "#FFFFFF"
MUTED = "#C8C8C8"
DIM = "#9A9A9A"
ACCENT = "#58C4DD"       # BLUE_C — chrome & highlights
BLUE = "#58C4DD"         # queries
GOLD = "#F0AC5F"         # keys / scale
MINT = "#5CD0B3"         # writes / outputs
VIOLET = "#9A72AC"       # RoPE
ROSE = "#FC6255"         # output / future-mask
RED = "#C55F73"

# ------------------------------------------------------------ type system ---
HERE = Path(__file__).parent
FONT_DIR = HERE / "assets" / "fonts"

# D1: families are bundled and registered at import, so a render on Linux
# matches macOS. Avenir Next / DIN Alternate are macOS-only; they survive only
# behind the escape hatch, for diffing against the v2 baseline.
USE_SYSTEM_FONTS = os.environ.get("USE_SYSTEM_FONTS") == "1"
if USE_SYSTEM_FONTS:  # pragma: no cover - baseline diffing only
    FONT, DIN = "Avenir Next", "DIN Alternate"
else:
    for _ttf in sorted(FONT_DIR.glob("*.ttf")):
        manimpango.register_font(str(_ttf))
    FONT, DIN = "Inter", "Barlow Semi Condensed"

# Nothing on screen may fall below the legibility floor. Barlow Semi Condensed
# sets ~8% narrower than DIN Alternate, so the tracked-label class steps up one
# point to hold its optical size against the old look.
MIN_LABEL_SIZE = 12
TYPE = {
    "micro": MIN_LABEL_SIZE,   # id labels, watermark, chip sublabels
    "tag": 14,                 # tracked technical labels (DIN_T default)
    "label": 16,
    "body": 18,
    "lead": 19,
    "h3": 24,
    "h2": 36,
    "formula": 40,             # MathTypst default
    "hero": 58,
}


def pt(size):
    """Enforce the type floor. Raising beats clamping: a silent clamp hides the overflow."""
    if size < MIN_LABEL_SIZE:
        raise ValueError(
            f"font size {size} is below the {MIN_LABEL_SIZE} pt floor — use TYPE[...]")
    return size


# ------------------------------------------------------------------- facts --
# Every model figure on screen comes from facts.json (built by
# tools/build_facts.py from config.py + the tokenizer, gated by check_facts).
FACTS = json.loads((HERE / "facts.json").read_text())
SPEC = FACTS["spec"]
VOCAB = FACTS["vocab"]                     # 128,256 — the run's real vocabulary
VOCAB_LABEL = FACTS["vocab_label"]
PARAMS = FACTS["params"]
PARAMS_LABEL = FACTS["params_label"]
D_MODEL = SPEC["d_model"]
D_FF = SPEC["d_ff"]
N_LAYERS = SPEC["n_layers"]
N_HEADS = SPEC["n_heads"]
N_KV_HEADS = SPEC["n_kv_heads"]
HEAD_DIM = SPEC["head_dim"]
SEQ_LEN = SPEC["seq_len"]
RMS_EPS = SPEC["rms_norm_eps"]
ROPE_THETA = SPEC["rope_theta"]
TOKENS = FACTS["tokens"]
ATTN = FACTS["attention"]
EPS_LABEL = f"{RMS_EPS:.0e}".replace("e-0", "e-")      # 1e-5
THETA_LABEL = f"{ROPE_THETA:,.0f}"                      # 500,000
# The config file names fewer rows than the tokenizer actually holds; the
# embedding table follows the tokenizer. Formatted from facts, never typed.
VOCAB_CONFIG_LABEL = f"{SPEC['vocab_config']:,}"
VOCAB_DELTA = VOCAB - SPEC["vocab_config"]

_SUBDIGITS = "₀₁₂₃₄₅₆₇₈₉"


def sub(n):
    """Subscript an integer for on-screen labels (q₃, k₄)."""
    negative = n < 0
    digits = "".join(_SUBDIGITS[int(d)] for d in str(abs(int(n))))
    return ("₋" if negative else "") + digits


def spec_chip_labels():
    """The five spec chips under the opening demo, each value from facts.json."""
    return [f"{PARAMS_LABEL} parameters", f"{N_LAYERS} blocks",
            f"d_model {D_MODEL:,}", f"vocab {VOCAB_LABEL}",
            f"context {SEQ_LEN:,}"]


def attention_demo():
    """(raw, scaled, weights) for the 4×4 demo — masked cells are exactly 0."""
    return ATTN["raw"], ATTN["scaled"], ATTN["weights"]


def attention_weight_bars():
    """Weight bars for the demo's query row: the masked softmax, not a sketch."""
    labels = [f"k{sub(i + 1)} {item['text']}"
              for i, item in enumerate(TOKENS["items"][:4])]
    return labels, list(ATTN["row_weights"])


def swiglu_demo():
    """(SiLU(gate), up, product) — gating is multiplicative, never a threshold."""
    gate = np.array([1.9, 0.4, -0.1, 2.6, 0.9, 0.05, 1.4, -0.3])
    up = np.array([0.8, -0.5, 1.2, 0.3, -0.9, 0.4, 0.7, -0.2])
    silu = gate / (1.0 + np.exp(-gate))
    return silu, up, silu * up


def cache_bar_counts():
    """Schematic cache quanta: two bars per KV head, so GQA draws half of MHA."""
    return 2 * N_HEADS, 2 * N_KV_HEADS


def period_label(tokens):
    """Compact token period for the RoPE frequency ladder (6.3, 26, 2.6M)."""
    if tokens >= 1e6:
        return f"{tokens / 1e6:.1f}M"
    if tokens >= 100:
        return f"{tokens:,.0f}"
    if tokens >= 10:
        return f"{tokens:.0f}"
    return f"{tokens:.1f}"


# §4 geometry: 1.5 keeps the axes and their caption inside the safe rect; the
# old 1.9 pushed the caption below the frame.
GEO_SIZE = 1.5
# §6 orbit demo range — a scene choice, not a model fact.
DEMO_LAST_M = 7


# The only timing source: exact per-sentence spans from the assembled tracks.
TIMINGS = json.loads((HERE / "audio" / "timings.json").read_text())
LEAD = TIMINGS["lead"]
TAIL = TIMINGS["tail"]

ORDER = list(TIMINGS["sections"])
BUDGETS = {k: TIMINGS["sections"][k]["duration"] + LEAD + TAIL for k in ORDER}
# sentence id -> (section key, start, end) in the section's own track
CUES = {sid: (sec, span["start"], span["end"])
        for sec, data in TIMINGS["sections"].items()
        for sid, span in data["sentences"].items()}
STARTS = {}
_acc = 0.0
for _k in ORDER:
    STARTS[_k] = _acc
    _acc += BUDGETS[_k]
TOTAL = _acc


# ---------------------------------------------------------------- typography
def T(text, size=TYPE["h3"], color=WHITE, weight="MEDIUM", font=FONT, **kw):
    return Text(text, font=font, font_size=pt(size), color=color, weight=weight,
                line_spacing=0.9, **kw)


def DIN_T(text, size=TYPE["tag"], color=MUTED, tracked=True):
    if tracked:
        text = "  ".join("\u200a".join(w) for w in text.split())
    return Text(text, font=DIN, font_size=pt(size), color=color, weight="BOLD",
                line_spacing=0.9)


def TY(s, size=TYPE["formula"], color=WHITE, **kw):
    return MathTypst(s, font_size=pt(size), color=color, **kw)


def lighter(c, f=0.35):
    from manim import ManimColor
    return interpolate_color(ManimColor(c), ManimColor("#FFFFFF"), f)


# ---------------------------------------------------------- surfaces & glow -
def glass(fill_op=0.05, stroke_op=0.32, radius=0.08):
    # Flat black panel. The outline carries the shape; a tinted fill and a
    # colored top-bar read as an app card, not as mathematics.
    return {"stroke_color": MUTED, "stroke_opacity": stroke_op,
            "stroke_width": 1.1, "fill_color": BLACK,
            "fill_opacity": 1.0, "corner_radius": radius}


def card(w, h, edge=None, shadow=True, fill_op=0.05, radius=0.08):
    # Returns one child: the panel. `edge` colors its outline. `shadow` and
    # `fill_op` stay for call-site compatibility; both are inert.
    box = RoundedRectangle(width=w, height=h, **glass(fill_op, radius=radius))
    if edge:
        box.set_stroke(edge, opacity=0.58, width=1.5)
    return VGroup(box)


def pill(label, color=WHITE, size=TYPE["body"], stroke=None, weight="MEDIUM"):
    txt = T(label, size=size, color=color, weight=weight)
    box = RoundedRectangle(corner_radius=0.06,
                           width=txt.width + 0.64, height=txt.height + 0.26,
                           stroke_color=stroke or color,
                           stroke_opacity=0.58 if stroke else 0.34,
                           stroke_width=1.25,
                           fill_color=BLACK, fill_opacity=1.0)
    return VGroup(box, txt.move_to(box))


def token_pill(word, id_label=None, color=WHITE, size=20):
    box = RoundedRectangle(corner_radius=0.07, width=len(word) * 0.155 + 0.46,
                           height=0.52, stroke_color=color, stroke_opacity=0.68,
                           stroke_width=1.45, fill_color=BLACK, fill_opacity=1.0)
    txt_c = WHITE if color == WHITE else lighter(color, 0.25)
    g = VGroup(box, T(word, size=size, color=txt_c, weight="MEDIUM").move_to(box))
    if id_label is not None:
        g.add(DIN_T(id_label, size=13, color=MUTED, tracked=False)
              .next_to(box, DOWN, buff=0.10))
    return g


def glow_dot(center, color, r=0.09):
    g = VGroup()
    for mult, op in [(4.4, 0.04), (3.0, 0.07), (2.0, 0.12), (1.35, 0.2)]:
        g.add(Circle(radius=r * mult, stroke_width=0, fill_color=color,
                     fill_opacity=op).move_to(center))
    g.add(Dot(center, radius=r, color=color).set_fill(color, 1.0))
    return g


def glow_ring(mobj, color):
    # One quiet halo. Stacked wide rings are neon, not emphasis.
    return VGroup(mobj.copy().set_stroke(color, width=2.6, opacity=0.22)
                  .set_fill(opacity=0))


class _MatrixGrid(VGroup):
    """VGroup of matrix cells whose set_fill preserves relative heatmap shading."""
    def set_fill(self, color=None, opacity=None, family=True):
        if opacity is not None and getattr(self, "_weights", None):
            for sq, w in zip(self.submobjects, self._weights):
                sq.set_fill(color=color, opacity=min(0.95, opacity * (0.45 + 1.15 * w)))
            return self
        return super().set_fill(color=color, opacity=opacity, family=family)


def grid(rows, cols, cell=0.24, gap=0.035):
    cells = _MatrixGrid()
    weights = []
    for r in range(rows):
        for c in range(cols):
            w = float(0.5 + 0.5 * np.sin(r * 1.9 + c * 2.7 + 0.4 * r * c))
            weights.append(w)
            sq = Square(side_length=cell, stroke_color=WHITE,
                        stroke_opacity=0.14, stroke_width=1.0,
                        fill_color=WHITE, fill_opacity=0.02 + 0.09 * w)
            sq.move_to(np.array([c * (cell + gap), -r * (cell + gap), 0]))
            cells.add(sq)
    cells._weights = weights
    return cells


def matrix_brackets(mob, color=MUTED, pad=0.07, tick=0.13, stroke_width=1.8, opacity=0.65):
    """Square 3Blue1Brown matrix brackets [ · ] framing a grid or table."""
    l, r, b, t = bbox(mob)
    lx, rx = l - pad, r + pad
    by, ty = b - pad, t + pad
    left_b = VMobject(stroke_color=color, stroke_width=stroke_width, stroke_opacity=opacity)
    left_b.set_points_as_corners([
        [lx + tick, ty, 0], [lx, ty, 0], [lx, by, 0], [lx + tick, by, 0]
    ])
    right_b = VMobject(stroke_color=color, stroke_width=stroke_width, stroke_opacity=opacity)
    right_b.set_points_as_corners([
        [rx - tick, ty, 0], [rx, ty, 0], [rx, by, 0], [rx - tick, by, 0]
    ])
    return VGroup(left_b, right_b)


def dial_ticks(dial_mob, color=MUTED, n=9, inner_ratio=0.80, outer_ratio=0.96):
    """Precision radial gauge ticks inside a circular dial."""
    c = dial_mob.get_center()
    r = dial_mob.width / 2
    ticks = VGroup()
    for a in np.linspace(-0.75 * np.pi, 0.75 * np.pi, n):
        u = np.array([np.cos(a), np.sin(a), 0])
        ticks.add(Line(c + u * r * inner_ratio, c + u * r * outer_ratio,
                       stroke_color=color, stroke_opacity=0.38, stroke_width=1.1))
    return ticks


def flat_fill(mobj, c):
    return mobj.set_fill(c, opacity=1)


def bars(labels, probs, colors, width=2.6, bh=0.30, size=19):
    g = VGroup()
    pmax = max(probs)
    for i, (lab, p, col) in enumerate(zip(labels, probs, colors)):
        y = -i * (bh + 0.16)
        lab_t = T(lab, size=size, color=lighter(col, 0.35), weight="MEDIUM")
        lab_t.move_to([0, y, 0], aligned_edge=LEFT)
        track = Rectangle(width=width, height=bh, stroke_width=0,
                          fill_color=WHITE, fill_opacity=0.06)
        track.next_to(lab_t, RIGHT, buff=0.18)
        track.set_y(lab_t.get_y())
        val = Rectangle(width=max(width * p / pmax, 1e-4), height=bh,
                        stroke_width=0, fill_color=col, fill_opacity=1)
        flat_fill(val, col)
        # A zero probability draws no fill: the mask removes the key, it does not
        # merely shrink its weight.
        val.set_fill(opacity=1.0 if p > 0 else 0.0)
        val.align_to(track, LEFT)
        val.set_y(track.get_y())
        pct = Text(f"{p:.2f}", font=DIN, font_size=pt(14),
                   color=lighter(col, 0.4),
                   weight="BOLD").next_to(track, RIGHT, buff=0.14)
        g.add(VGroup(lab_t, track, val, pct))
    return g


def head_badge(label, color, r=0.24):
    ring = Circle(radius=r, stroke_color=color, stroke_opacity=0.9,
                  stroke_width=2.0, fill_color=WHITE, fill_opacity=0.05)
    halo = glow_ring(ring, color)
    t = T(label, size=TYPE["tag"], color=color, weight="MEDIUM").move_to(ring)
    return VGroup(halo, ring, t)


def odot(color=MINT, r=0.17):
    c = Circle(radius=r, stroke_color=color, stroke_width=2.4)
    d = Dot(radius=r * 0.28, color=color)
    return VGroup(c, d)


def dim_arrow(a, b, c=WHITE, op=0.35, w=2.0, tip=0.13, buff=0.1):
    ar = Arrow(a, b, stroke_color=c, fill_color=c, stroke_width=w,
               tip_length=tip, buff=buff)
    ar.set_stroke(opacity=op)
    ar.set_fill(opacity=op)
    return ar


def data_arrow(a, b, c, w=2.2, op=0.85):
    return dim_arrow(a, b, c, op, w)


def flow(mobj, color, time_width=0.35, run_time=0.8):
    return ShowPassingFlash(mobj.copy().set_stroke(color, width=4.5, opacity=0.9),
                            time_width=time_width, run_time=run_time)


# ------------------------------------------------------------ layout guard --
# The region every glyph must stay inside: the 14.222 × 8 frame less a 0.4-unit
# margin. The bottom is 0.4 rather than 0.6 because the progress bar is chrome.
SAFE = (-6.7, 6.7, -3.6, 3.6)          # left, right, bottom, top
SAFE_TOL = 0.02                        # glyph bbox tolerance; text is not exact


def bbox(mob):
    """(left, right, bottom, top) from the critical points — the frame unit."""
    c = mob.get_critical_point
    return (c(LEFT)[0], c(RIGHT)[0], c(DOWN)[1], c(UP)[1])


def text_leaves(mobjects, skip=()):
    """Every Text/MathTypst glyph owner under each top-level mobject."""
    for top in mobjects:
        if any(top is s for s in skip):
            continue
        for leaf in top.get_family():
            if isinstance(leaf, (Text, MathTypst)):
                yield top, leaf


def text_label(mob, leaf=None):
    """A short human label for a layout message: its words, or its class."""
    source = leaf if leaf is not None else next(
        (l for l in mob.get_family() if isinstance(l, (Text, MathTypst))), None)
    if source is None:
        return type(mob).__name__
    raw = (getattr(source, "original_text", None)
           or getattr(source, "text", None)
           or getattr(source, "typst_code", None) or "")
    return " ".join(str(raw).replace("$", "").split())[:56]


def safe_violations(mobjects, safe=SAFE, tol=SAFE_TOL, skip=()):
    """Text whose bbox leaves the safe rect. The unit is one glyph object, so
    an overlap between two shapes is invisible here by construction — the
    group-overlap log covers that axis separately."""
    left, right, bottom, top = safe
    out = []
    for owner, leaf in text_leaves(mobjects, skip):
        l, r, b, t = bbox(leaf)
        over = {}
        if l < left - tol:
            over["left"] = round(left - l, 2)
        if r > right + tol:
            over["right"] = round(r - right, 2)
        if b < bottom - tol:
            over["bottom"] = round(bottom - b, 2)
        if t > top + tol:
            over["top"] = round(t - top, 2)
        if over:
            out.append({"text": text_label(owner, leaf), "over": over,
                        "owner": type(owner).__name__})
    return out


def _contains(outer, inner, tol=0.01):
    return (outer[0] <= inner[0] + tol and outer[1] >= inner[1] - tol
            and outer[2] <= inner[2] + tol and outer[3] >= inner[3] - tol)


def _has_text(mob):
    return any(isinstance(l, (Text, MathTypst)) for l in mob.get_family())


def group_overlaps(mobjects, skip=(), min_area=0.06, tol=0.01):
    """Top-level groups that partially overlap and both carry text.

    Containment is not a collision — a label sitting inside a card is the
    normal composition — so only partial intersections are reported. Groups
    without a glyph (cards, arrows, plots) are excluded: their overlap is the
    diagram, and logging it buried the readable collisions in noise.
    """
    tops = [m for m in mobjects
            if not any(m is s for s in skip) and _has_text(m)]
    boxes = [(m, bbox(m)) for m in tops]
    out = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            w = min(ba[1], bb[1]) - max(ba[0], bb[0])
            h = min(ba[3], bb[3]) - max(ba[2], bb[2])
            if w <= 0 or h <= 0 or w * h < min_area:
                continue
            if _contains(ba, bb, tol) or _contains(bb, ba, tol):
                continue
            out.append((text_label(a), text_label(b), round(w * h, 2)))
    return out


def operator_on_arrow(symbol, arrow, color=WHITE, size=26, pad=0.06):
    """A math operator centred on an arrow shaft, on a small knockout plate.

    The √ used to float beside the mean→RMS arrow with nothing tying it to the
    operation; sitting on the shaft it reads as "the arrow takes the root".
    """
    glyph = T(symbol, size=size, color=color, weight="BOLD")
    plate = RoundedRectangle(corner_radius=0.05, width=glyph.width + pad * 2,
                             height=glyph.height + pad,
                             stroke_width=0, fill_color=BLACK,
                             fill_opacity=0.92).move_to(arrow.get_center())
    glyph.move_to(plate)
    return VGroup(plate, glyph)


# ------------------------------------------------------------- base scene ---
class Section(Scene):
    SEC = ""
    CHAPTER = ""
    ACCENT = ACCENT

    def setup(self):
        self.camera.background_color = BLACK
        self.budget = BUDGETS[self.SEC]
        self._t = 0.0
        self._ft = 0.0                  # per-frame clock the updaters walk
        self._cues = []                   # clause starts, for the order guard
        self._pulse_at = -9.0              # last cue, for the flow pulse
        self.check_layout = os.environ.get("CHECK_LAYOUT") == "1"
        self._violations = []
        self._seen_overlaps = set()
        self._build_ambient()
        self._build_chrome()

    # ambient --------------------------------------------------------------
    def _build_ambient(self):
        # Canvas breath: a full-frame rect that lifts the black a few grey
        # levels and sinks back. The freeze gate reads pixels; the breath is
        # what keeps a held diagram from counting as a dead frame. It sits at
        # the bottom of the stack, so opaque content covers it where it lands.
        fw, fh = config.frame_width, config.frame_height
        base = Rectangle(width=fw + 0.1, height=fh + 0.1, stroke_width=0,
                         fill_color=WHITE, fill_opacity=0.0)
        self._breath = base
        self._ambient = VGroup(base)
        self.add(self._ambient)

    def _start_canvas_life(self):
        """Start the canvas breath: ~2.8 s period, 0..6 grey levels.

        Barely visible as decoration, far above freezedetect's noise floor.
        Cue coverage — a visual event per sentence — is the pacing check and
        is measured separately; the breath alone does not make pacing clean.
        """
        def breathe(m, dt):
            t = self._ft            # the same clock the playhead uses
            self._breath.set_fill(
                opacity=0.024 * (0.5 + 0.5 * np.sin(t * TAU / 2.8)))
        self._ambient.add_updater(breathe)

    # chrome ---------------------------------------------------------------
    def _build_chrome(self):
        self._chrome = VGroup()
        tag = VGroup(
            Dot(radius=0.028, color=self.ACCENT),
            DIN_T(self.CHAPTER, size=12, color=DIM))
        tag.arrange(RIGHT, buff=0.14).to_corner(UL, buff=0.42)
        tag.shift(DOWN * 0.06)
        fw = config.frame_width
        pw = fw - 0.9
        py = -3.85
        pr = 0.010
        track = RoundedRectangle(corner_radius=pr, width=pw, height=0.020,
                                 stroke_width=0, fill_color=WHITE,
                                 fill_opacity=0.08).move_to([0, py, 0])
        done_w = max(pw * STARTS[self.SEC] / TOTAL, 1e-3)
        cur_w = pw * self.budget / TOTAL
        done = RoundedRectangle(corner_radius=pr, width=done_w, height=0.020,
                                stroke_width=0, fill_color=WHITE,
                                fill_opacity=0.14)
        done.move_to([-pw / 2 + done_w / 2, py, 0])
        cur = RoundedRectangle(corner_radius=pr, width=cur_w, height=0.020,
                               stroke_width=0, fill_color=self.ACCENT,
                               fill_opacity=0.32)
        cur.move_to([-pw / 2 + done_w + cur_w / 2, py, 0])
        ticks = VGroup()
        for k in ORDER:   # one tick per chapter boundary
            x = -pw / 2 + pw * STARTS[k] / TOTAL
            ticks.add(Rectangle(width=0.010, height=0.065, stroke_width=0,
                                fill_color=WHITE, fill_opacity=0.18)
                      .move_to([x, py, 0]))
        # Clause playhead: slides across this scene's slice of the track and
        # flares at every cue. It is the moving highlight for the spoken
        # clause, and it is what keeps a wait from reading as a still frame.
        head = RoundedRectangle(corner_radius=0.015, width=0.055, height=0.16,
                                stroke_width=0, fill_color=self.ACCENT,
                                fill_opacity=0.80)
        halo = RoundedRectangle(corner_radius=0.04, width=0.24, height=0.28,
                                stroke_width=0, fill_color=self.ACCENT,
                                fill_opacity=0.0)
        self._playhead = VGroup(halo, head)
        self._play_x0 = -pw / 2 + done_w
        self._play_w = cur_w
        self._play_y = py

        def place(m, dt):
            self._ft += dt
            frac = min(1.0, max(0.0, self._ft / self.budget))
            m.move_to([self._play_x0 + self._play_w * frac, self._play_y, 0])
            age = self._ft - self._pulse_at
            flare = 1.0 if age < 0 else max(0.0, 1.0 - age / 0.7)
            halo.set_fill(opacity=0.20 * flare)
            halo.stretch_to_fit_width(0.34 + 0.42 * flare)
        self._playhead.add_updater(place)
        self._chrome.add(tag, track, done, cur, ticks, self._playhead)
        self.add(self._chrome)

    def watermark(self):
        """Corner mark. Only the end card carries it — v2 stamped every scene."""
        wm = DIN_T("LLaMA-3-Lite", size=12, color=DIM)
        return wm.to_corner(DR, buff=0.42).shift(UP * 0.16)

    # section header ---------------------------------------------------------
    def section_header(self, kicker, title, sub=None, run_time=1.2, ty=2.62,
                       size=40):
        kick = DIN_T(kicker, size=13, color=self.ACCENT)
        rule = RoundedRectangle(corner_radius=0.012, width=0.92, height=0.026,
                                stroke_width=0, fill_color=self.ACCENT,
                                fill_opacity=0.85)
        ttl = T(title, size=size, color=WHITE, weight="BOLD")
        if sub:
            sub_m = T(sub, size=19, color=MUTED, weight="MEDIUM")
            block = VGroup(ttl, sub_m).arrange(DOWN, buff=0.18)
        else:
            block = VGroup(ttl)
        block.move_to([0, ty, 0])
        kick.next_to(block, UP, buff=0.30).align_to(block, LEFT)
        rule.next_to(kick, DOWN, buff=0.14).align_to(kick, LEFT)
        anims = [Create(rule),
                 Write(kick),
                 Write(ttl)]
        if sub:
            anims.append(Write(sub_m))
        self.play(LaggedStart(*anims, lag_ratio=0.18), run_time=run_time)
        return VGroup(kick, rule, block)

    # timing wrappers -----------------------------------------------------
    # Run times snap to whole frames so the tracked clock equals the clock the
    # MP4 actually has. Without the snap, manim's frame rounding drifts a scene
    # up to a frame per play and cue targets land off the narration.
    def _snap(self, rt):
        fps = config.frame_rate
        return max(1, round(rt * fps)) / fps

    def play(self, *args, **kw):
        rt = kw.pop("run_time", None)
        if rt is None and args:
            rt = getattr(args[0], "run_time", None)
        if rt is None:
            rt = 1.0
        rt = self._snap(rt)
        self._ft = self._t        # re-anchor before frames render, so the
        super().play(*args, run_time=rt, **kw)   # playhead cannot drift off
        self._t += rt

    def wait(self, t):
        if t <= 0:
            return
        self.play(Wait(run_time=t))

    def cue(self, sid):
        """Hold until sentence `sid` starts to speak: its offset + LEAD.

        Cues self-correct against the absolute sentence time, so rounding never
        accumulates across a scene. Reaching a cue late is a content bug and is
        logged; sync_end() is what fails the scene.
        """
        sec, start, end = CUES[sid]
        if sec != self.SEC:
            raise ValueError(f"{self.SEC} cannot cue {sid} — it belongs to {sec}")
        if self._cues and start < self._cues[-1]:
            raise ValueError(f"{self.SEC}: cue {sid} is out of order")
        self._cues.append(start)
        rem = LEAD + start - self._t
        if rem > 0:
            self.wait(rem)
        elif rem < -1 / 30:
            print(f"!! [{self.SEC}] cue {sid} LATE by {-rem:.2f}s")
        self._pulse_at = self._t          # flow pulse fires on the clause
        self._layout_check(f"cue {sid}")

    # layout guard ---------------------------------------------------------
    def _layout_check(self, where):
        """Assert every visible glyph sits in SAFE and log real group collisions.

        Skipped unless CHECK_LAYOUT=1; sync_end() raises if anything is out,
        so a render cannot pass the gate by printing violations and continuing.
        """
        if not self.check_layout:
            return
        mobjects = [m for m in self.mobjects if m is not self._ambient]
        for item in safe_violations(mobjects, skip=(self._ambient,)):
            self._violations.append(item)
            print(f"!! [{self.SEC}] {where}: {item['text']!r} out of safe rect "
                  f"{item['over']}")
        # Chrome spans the frame by construction (progress bar), so it would
        # collide with everything; the scene content is what needs reviewing.
        skip = (self._ambient, self._chrome)
        for a, b, area in group_overlaps(mobjects, skip=skip):
            key = tuple(sorted((a, b)))
            if key in self._seen_overlaps:
                continue
            self._seen_overlaps.add(key)
            print(f"   [{self.SEC}] {where}: overlap {a!r} ↔ {b!r} ({area})")
            seen = set()
            for top, leaf in text_leaves(mobjects, skip):
                lab = text_label(top, leaf)
                if lab in (a, b) and id(top) not in seen:
                    seen.add(id(top))
                    print(f"        {lab!r} {type(top).__name__} "
                          f"bbox={tuple(round(float(v), 2) for v in bbox(top))}")

    def sync_end(self):
        """Fill the budget exactly, and fail the scene if content overran it.

        The wait runs down the remainder so the scene lands on its budget; the
        raise is what makes the timing contract real rather than a printout.
        """
        rem = self.budget - self._t
        print(f"[{self.SEC}] tracked {self._t:.2f} / budget {self.budget:.2f}")
        if rem > 0:
            self.wait(rem)
        err = self._t - self.budget
        # One frame of slack: the wait can only land on a frame boundary, so a
        # sub-frame residual is rounding, not a content overrun.
        if abs(err) > 1 / config.frame_rate + 1e-6:
            raise RuntimeError(
                f"{self.SEC}: ends at {self._t:.2f}s vs budget {self.budget:.2f}s "
                f"({err:+.2f}s) — content must fit LEAD + audio + TAIL")
        if self.check_layout:
            print(f"[{self.SEC}] layout: {len(self._violations)} safe-rect "
                  f"violations, {len(self._seen_overlaps)} partial overlaps")
            if self._violations:
                raise RuntimeError(
                    f"{self.SEC}: {len(self._violations)} text objects outside "
                    f"SAFE {SAFE} — run without CHECK_LAYOUT to inspect the frame")

    def fade_content(self, rt=0.6):
        content = [m for m in self.mobjects
                   if m is not self._chrome and m is not self._ambient]
        if content:
            self.play(*[FadeOut(m) for m in content], run_time=rt)


# =========================================================== scene 1 ========
class Section01(Section):
    SEC = "sec01_function"
    CHAPTER = "01 · THE FUNCTION"
    ACCENT = GOLD

    def construct(self):
        self._start_canvas_life()

        # s01.1 — the title card settles and the claim takes the glow
        self.cue("s01.1")
        kick = DIN_T("THE MODEL AS A FUNCTION", size=13, color=GOLD)
        rule = RoundedRectangle(corner_radius=0.012, width=0.92, height=0.026,
                                stroke_width=0, fill_color=GOLD,
                                fill_opacity=0.85)
        l1 = T("A large language model is", size=36, color=MUTED)
        l2 = T("one mathematical function", size=58, color=WHITE, weight="BOLD")
        title = VGroup(l1, l2).arrange(DOWN, buff=0.18).move_to([0, 1.9, 0])
        kick.next_to(title, UP, buff=0.34).align_to(title, LEFT)
        rule.next_to(kick, DOWN, buff=0.14).align_to(kick, LEFT)
        head = VGroup(kick, rule, title)
        self.play(Create(rule), Write(kick), run_time=0.6)
        self.play(Write(l1), run_time=0.7)
        self.play(Write(l2), run_time=0.8)
        self.play(Indicate(l2, scale_factor=1.05, color=GOLD), run_time=0.7)

        # s01.2 — tokens go in, the function runs, probabilities come out
        hdr = T("one mathematical function", size=19, color=MUTED)
        hdr.move_to([0, 3.15, 0])

        chips = VGroup(token_pill("t₁"), token_pill("t₂"), token_pill("t₃"))
        chips.arrange(RIGHT, buff=0.30)
        dots = T("…", size=24, color=DIM, weight="BOLD")
        inputs = VGroup(chips, dots).arrange(RIGHT, buff=0.28)
        inputs.move_to([-4.35, 0.45, 0])
        fbox = card(1.85, 1.55, edge=GOLD)
        fbox.move_to([-1.05, 0.45, 0])
        fmark = TY("f_theta", size=46, color=WHITE).move_to(fbox)
        fdom = TY("f_theta : cal(V)^* -> Delta^(|cal(V)| - 1)", size=16, color=GOLD)
        fdom.next_to(fbox, UP, buff=0.18)
        flab = DIN_T("THE MODEL", size=13, color=MUTED)
        flab.next_to(fbox, DOWN, buff=0.16)
        arr1 = dim_arrow(inputs.get_right() + RIGHT * 0.06,
                         fbox.get_left() + LEFT * 0.02, op=0.5, w=2.4, tip=0.16)
        prob = bars(["mat", "floor", "sofa", "…"],
                    [0.72, 0.12, 0.09, 0.07],
                    [GOLD, MUTED, MUTED, MUTED], width=2.15, bh=0.32, size=20)
        prob.move_to([2.75, 0.15, 0])
        prob_title = DIN_T("NEXT-TOKEN PROBABILITIES", size=TYPE["tag"],
                           color=MUTED)
        prob_title.next_to(prob, UP, buff=0.18).align_to(prob, LEFT)
        arr2 = dim_arrow(fbox.get_right() + RIGHT * 0.06,
                         prob.get_left() + LEFT * 0.02, op=0.5, w=2.4, tip=0.16)
        self.cue("s01.2")
        self.play(ReplacementTransform(head, hdr), run_time=0.55)
        self.play(LaggedStart(
            FadeIn(inputs, shift=RIGHT * 0.3), Create(arr1),
            FadeIn(fbox, shift=UP * 0.15), FadeIn(fmark), FadeIn(fdom), Write(flab),
            Create(arr2), GrowFromEdge(prob, DOWN), Write(prob_title),
            lag_ratio=0.14), run_time=3.4)

        # s01.3 — the two ends of the interface get their names
        tin = DIN_T("TEXT IN", size=TYPE["tag"], color=BLUE, tracked=False)
        tout = DIN_T("PROBABILITIES OUT", size=TYPE["tag"], color=GOLD,
                     tracked=False)
        tin.next_to(inputs, DOWN, buff=0.42)
        tout.next_to(prob, DOWN, buff=0.42).align_to(prob, LEFT)
        cap = T("one function:  tokens → probabilities", size=20, color=MUTED)
        cap.move_to([0, -1.85, 0])
        self.cue("s01.3")
        self.play(FadeOut(hdr), Write(tin), Write(tout), run_time=1.2)
        self.play(Write(cap), run_time=1.2)

        # s01.4 / s01.5 — the spec chips build in, then the row settles
        specs = VGroup(*[pill(s, color=WHITE, size=16)
                         for s in spec_chip_labels()])
        specs.arrange(RIGHT, buff=0.22).move_to([0, -2.55, 0])
        pair = VGroup(specs[3], specs[4])
        pair_home = pair.get_center()[0]
        pair.shift(LEFT * pair_home)   # open centred so the row has no hole
        self.cue("s01.4")
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.12) for s in pair],
                              lag_ratio=0.30), run_time=2.4)
        style_tag = T("built from scratch · in the LLaMA-3 style", size=17,
                      color=GOLD)
        style_tag.move_to([0, -3.25, 0])
        self.play(Write(style_tag), run_time=1.2)
        self.play(Indicate(pair[0], scale_factor=1.10, color=GOLD),
                  run_time=0.8)
        self.play(Indicate(pair[1], scale_factor=1.10, color=GOLD),
                  run_time=0.8)
        self.cue("s01.5")
        self.play(FadeOut(style_tag), run_time=0.5)
        self.play(pair.animate.shift(RIGHT * pair_home), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.12) for s in specs[:3]],
                              lag_ratio=0.14), run_time=1.6)
        srule = Line(specs.get_left(), specs.get_right(), stroke_width=1.6,
                     stroke_color=GOLD, stroke_opacity=0.5)
        srule.next_to(specs, DOWN, buff=0.12)
        self.play(Create(srule), run_time=0.9)
        self.play(LaggedStart(
            Indicate(specs[1], scale_factor=1.05, color=GOLD),
            Indicate(specs[2], scale_factor=1.05, color=GOLD),
            Indicate(specs[0], scale_factor=1.05, color=GOLD),
            lag_ratio=0.35), run_time=2.0)

        # s01.6 — the chips pulse once, in sequence
        self.cue("s01.6")
        self.play(LaggedStart(
            *[Indicate(s, scale_factor=1.05, color=WHITE) for s in specs],
            lag_ratio=0.32), run_time=2.4)

        # s01.7 — the demo sentence writes itself in under the chips
        demo = VGroup(inputs, arr1, fbox, fmark, fdom, flab, arr2, prob, prob_title,
                      tin, tout, specs)
        words = [it["text"] for it in TOKENS["items"]]
        toks = VGroup(*[token_pill(w, size=18) for w in words])
        toks.arrange(RIGHT, buff=0.22).move_to([0, -2.55, 0])
        self.cue("s01.7")
        self.play(FadeOut(cap), FadeOut(srule),
                  demo.animate.scale(0.8).shift(UP * 0.8), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in toks],
                              lag_ratio=0.18), run_time=1.5)

        # s01.8 — six IDs, and the open question
        ids = VGroup(*[
            DIN_T(str(item["id"]), size=13, color=MUTED, tracked=False)
            .next_to(toks[i], DOWN, buff=0.10)
            for i, item in enumerate(TOKENS["items"])])
        qm = T("?", size=34, color=GOLD, weight="BOLD")
        qm.next_to(toks, RIGHT, buff=0.35)
        self.cue("s01.8")
        self.play(LaggedStart(*[FadeIn(i, shift=UP * 0.10) for i in ids],
                              lag_ratio=0.14), run_time=1.6)
        self.play(Write(qm), run_time=1.0)

        # s01.9 — one token walks in as an ID and out as a distribution
        id_chip = pill(str(TOKENS["items"][-1]["id"]), color=GOLD, size=20)
        id_chip.move_to(toks[-1])
        self.cue("s01.9")
        self.play(ReplacementTransform(toks[-1], id_chip),
                  FadeOut(ids[-1]), run_time=0.8)
        in_pt = fbox.get_bottom() + DOWN * 0.35
        out_pt = fbox.get_right() + RIGHT * 0.12
        path = ArcBetweenPoints(id_chip.get_center(), in_pt,
                                angle=-TAU / 9)
        self.play(MoveAlongPath(id_chip, path), run_time=1.6, rate_func=linear)
        out_dot = glow_dot(out_pt, GOLD, r=0.06)
        self.play(FadeOut(id_chip), FadeIn(out_dot), run_time=0.5)
        path2 = ArcBetweenPoints(out_pt, prob.get_left() + LEFT * 0.35,
                                 angle=-TAU / 9)
        self.play(MoveAlongPath(out_dot, path2), run_time=1.2, rate_func=linear)
        over = DIN_T(f"OVER {VOCAB_LABEL} TOKENS", size=TYPE["tag"], color=GOLD,
                     tracked=False).scale(0.8)
        over.move_to(tout, aligned_edge=LEFT)
        self.play(FadeOut(out_dot), ReplacementTransform(tout, over),
                  Indicate(prob, scale_factor=1.05, color=GOLD), run_time=1.2)

        # s01.10 — the demo hands over to the walk's map: the block stack
        self.cue("s01.10")
        self.fade_content(0.5)
        stack = VGroup()
        for i in range(N_LAYERS):
            stack.add(RoundedRectangle(corner_radius=0.05, width=3.1,
                                       height=0.15, stroke_color=WHITE,
                                       stroke_opacity=0.13, stroke_width=1.1,
                                       fill_color=WHITE, fill_opacity=0.045))
        stack.arrange(DOWN, buff=0.045).move_to([-2.6, 0.0, 0])
        emb = pill(f"Embedding   {VOCAB_LABEL} → {D_MODEL:,}", color=BLUE,
                   size=17)
        emb.next_to(stack, UP, buff=0.35)
        unemb = pill(f"Unembedding   {D_MODEL:,} → {VOCAB_LABEL}", color=ROSE,
                     size=17)
        unemb.next_to(stack, DOWN, buff=0.35)
        raw = DIN_T("RAW TEXT", size=TYPE["tag"], color=MUTED)
        raw.next_to(emb, UP, buff=0.30)
        loop = pill("sampling loop", color=GOLD, size=17)
        loop.next_to(unemb, DOWN, buff=0.35)
        self.play(LaggedStart(FadeIn(raw), FadeIn(emb, shift=DOWN * 0.12),
                              *[Create(s) for s in stack],
                              FadeIn(unemb, shift=UP * 0.12),
                              FadeIn(loop, shift=UP * 0.12), lag_ratio=0.06),
                  run_time=2.4)
        ring = RoundedRectangle(corner_radius=0.06, width=3.28, height=0.22,
                                stroke_color=GOLD, stroke_width=2.4)
        ring.move_to(stack[0])   # block 1, the one the next scene opens
        lab = T("we'll open one block", size=19, color=GOLD)
        lab.next_to(ring, RIGHT, buff=0.45)
        self.play(Create(ring), Write(lab), run_time=0.9)

        # s01.11 — the stages light one by one along the silhouette
        tick_x = min(s.get_left()[0] for s in (emb, stack, unemb, loop))
        ticks = VGroup(*[
            Line([tick_x - 0.45, s.get_center()[1], 0],
                 [tick_x - 0.05, s.get_center()[1], 0],
                 stroke_width=2.6, stroke_color=GOLD, stroke_opacity=0.85)
            for s in (emb, stack, unemb, loop)])
        self.cue("s01.11")
        for tick, stage in zip(ticks, (emb, stack, unemb, loop)):
            self.play(Create(tick), run_time=0.5)
            self.play(Indicate(stage, scale_factor=1.06, color=GOLD),
                      run_time=0.6)

        # s01.12 — the loop turns one prediction into a paragraph
        para_lines = (
            VGroup(*[T(w, size=20, color=MUTED) for w in
                     ["The", "cat", "sat", "on", "the", "mat.", "It"]]
            ).arrange(RIGHT, buff=0.16),
            VGroup(*[T(w, size=20, color=MUTED) for w in
                     ["was", "warm", "there,", "and", "the", "light",
                      "stayed."]]
            ).arrange(RIGHT, buff=0.16),
        )
        para = VGroup(*para_lines).arrange(DOWN, buff=0.16,
                                           aligned_edge=LEFT)
        para.move_to([2.75, -2.2, 0])
        parw = [w for line in para_lines for w in line]
        self.cue("s01.12")
        self.play(LaggedStart(*[FadeIn(w, shift=UP * 0.10) for w in parw],
                              lag_ratio=0.12), run_time=1.9)

        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 2 ========
class Section02(Section):
    SEC = "sec02_objective"
    CHAPTER = "02 · THE OBJECTIVE"
    ACCENT = MINT

    def construct(self):
        self._start_canvas_life()

        # s02.1 — an empty objective card opens
        self.cue("s02.1")
        hdr = self.section_header("THE OBJECTIVE", "Why next-token prediction",
                                 run_time=1.1, ty=2.75, size=36)
        card_g = card(11.2, 3.15, edge=MINT)
        card_g.move_to([0, 0.62, 0])
        self.play(FadeIn(card_g), run_time=1.2)

        # s02.2 — the objective names itself on the card
        obj = T("next-token prediction", size=32, color=WHITE, weight="BOLD")
        obj.move_to([0, 1.62, 0])
        obj_sub = T("trained with cross-entropy  +  z-loss", size=18,
                    color=MUTED)
        obj_sub.move_to([0, 1.12, 0])
        self.cue("s02.2")
        self.play(Write(obj), run_time=1.6)
        self.play(Write(obj_sub), run_time=1.4)
        self.play(Indicate(obj, scale_factor=1.05, color=MINT), run_time=0.8)

        # s02.3 — real text, with the next token hidden behind a mask
        words = [it["text"] for it in TOKENS["items"]]
        sentence = T(TOKENS["sentence"], size=28, color=WHITE)
        sentence.move_to([-0.35, 0.15, 0])
        row = VGroup(*[token_pill(w, size=18) for w in words])
        row.arrange(RIGHT, buff=0.22).move_to([-0.35, 0.15, 0])
        self.cue("s02.3")
        self.play(Write(sentence), run_time=0.8)
        self.play(ReplacementTransform(sentence, row), run_time=1.1)
        mask = RoundedRectangle(corner_radius=0.14,
                                width=row[-1].width + 0.12,
                                height=row[-1].height + 0.06,
                                stroke_color=ROSE, stroke_width=1.7,
                                stroke_opacity=0.85, fill_color=BLACK,
                                fill_opacity=0.62)
        mask.move_to(row[-1])
        hidden = DIN_T("HIDDEN", size=13, color=ROSE, tracked=False)
        hidden.next_to(mask, DOWN, buff=0.12)
        self.play(FadeOut(row[-1]), FadeIn(mask), Write(hidden), run_time=0.9)

        # s02.4 — the model assigns the hidden token a low probability
        gbg = RoundedRectangle(corner_radius=0.08, width=0.42, height=1.6,
                               stroke_color=MUTED, stroke_opacity=0.4,
                               stroke_width=1.3, fill_color=WHITE,
                               fill_opacity=0.05)
        gbg.move_to([3.75, 0.2, 0])
        glab = DIN_T("p(true token)", size=13, color=MUTED, tracked=False)
        glab.next_to(gbg, UP, buff=0.16)
        gfill = RoundedRectangle(corner_radius=0.06, width=0.32, height=0.08,
                                 stroke_width=0, fill_color=MINT,
                                 fill_opacity=0.85)
        gfill.move_to(gbg.get_bottom(), aligned_edge=DOWN)
        glow = RoundedRectangle(corner_radius=0.06, width=0.32, height=0.34,
                                stroke_width=0, fill_color=MINT,
                                fill_opacity=0.85)
        glow.move_to(gbg.get_bottom(), aligned_edge=DOWN)
        val = T("0.14", size=20, color=MINT, weight="MEDIUM")
        val.next_to(gbg, RIGHT, buff=0.22)
        self.cue("s02.4")
        self.play(FadeIn(gbg), Write(glab), FadeIn(gfill), run_time=0.9)
        self.play(Transform(gfill, glow), Write(val), run_time=1.3)

        # s02.5 — the weight grid shuffles as the gauge climbs
        wg = grid(6, 9, cell=0.26, gap=0.045)
        wg.set_stroke(WHITE, opacity=0.18).set_fill(WHITE, 0.055)
        wg.move_to([-4.35, -2.25, 0])
        wg_br = matrix_brackets(wg, MINT, pad=0.06, tick=0.12)
        wlab = DIN_T("WEIGHTS", size=TYPE["tag"], color=MUTED)
        wlab.next_to(wg, DOWN, buff=0.18)
        self.cue("s02.5")
        self.play(FadeIn(wg), Create(wg_br), Write(wlab), run_time=0.9)
        gmid = RoundedRectangle(corner_radius=0.06, width=0.32, height=0.72,
                                stroke_width=0, fill_color=MINT,
                                fill_opacity=0.85)
        gmid.move_to(gbg.get_bottom(), aligned_edge=DOWN)
        val2 = T("0.38", size=20, color=MINT, weight="MEDIUM")
        val2.move_to(val)
        self.play(LaggedStart(*[Indicate(c, scale_factor=1.18, color=MINT)
                                for c in wg[::2]], lag_ratio=0.06),
                  Transform(gfill, gmid), ReplacementTransform(val, val2),
                  run_time=2.2)

        # s02.6 — "more likely" rises beside the text as the grid settles
        more = dim_arrow([-0.35, -0.85, 0], [-0.35, -0.28, 0], MINT,
                         op=0.55, w=2.4, tip=0.15, buff=0.02)
        more_lab = T("more likely", size=18, color=MINT)
        more_lab.next_to(more, RIGHT, buff=0.22)
        self.cue("s02.6")
        self.play(Create(more), Write(more_lab), run_time=1.4)
        self.play(Indicate(wg, scale_factor=1.04, color=MINT),
                  Indicate(gfill, scale_factor=1.10, color=MINT),
                  run_time=1.4)
        move_lab = T("every weight moves to make real text more likely",
                     size=16, color=MINT)
        move_lab.move_to([1.55, -3.25, 0])
        self.play(*[c.animate.set_fill(MINT, 0.11) for c in wg[::2]],
                  run_time=1.2)
        self.play(Write(move_lab), run_time=0.9)

        # s02.7 — the manual-label tag row fades out empty
        tags = VGroup(*[pill(s, color=DIM, size=15, stroke=DIM)
                        for s in ["sentiment tags", "answers", "rules"]])
        tags.arrange(RIGHT, buff=0.22).move_to([0.65, -1.62, 0])
        tlab = DIN_T("MANUAL LABELS", size=13, color=DIM)
        tlab.next_to(tags, UP, buff=0.16).align_to(tags, LEFT)
        self.cue("s02.7")
        self.play(FadeIn(tlab),
                  LaggedStart(*[FadeIn(t, shift=UP * 0.10) for t in tags],
                              lag_ratio=0.18), run_time=2.2)
        self.play(LaggedStart(*[FadeOut(t) for t in tags], lag_ratio=0.22),
                  FadeOut(tlab), FadeOut(move_lab), run_time=2.2)

        # s02.8 — the surprise dial reads high, then turns down
        dial = Circle(radius=0.78, stroke_color=MUTED, stroke_opacity=0.45,
                      stroke_width=2.2, fill_color=WHITE, fill_opacity=0.04)
        dial.move_to([4.35, -2.25, 0])
        dticks = dial_ticks(dial, MUTED)
        needle = Line(dial.get_center(),
                      dial.get_center() + RIGHT * 0.52 + UP * 0.52,
                      stroke_width=3.2, stroke_color=ROSE)
        dlab = DIN_T("SURPRISE", size=TYPE["tag"], color=MUTED)
        dlab.next_to(dial, DOWN, buff=0.18)
        hlab = DIN_T("HIGH", size=12, color=ROSE, tracked=False)
        hlab.next_to(dial.get_center() + RIGHT * 0.55 + UP * 0.55, UR, buff=0.12)
        llab = DIN_T("LOW", size=12, color=MINT, tracked=False)
        llab.next_to(dial.get_center() + LEFT * 0.55 + DOWN * 0.55, DL,
                     buff=0.12)
        self.cue("s02.8")
        self.play(FadeIn(dial), FadeIn(dticks), Create(needle), Write(dlab), Write(hlab),
                  Write(llab), run_time=1.0)
        ghigh = RoundedRectangle(corner_radius=0.06, width=0.32, height=1.32,
                                 stroke_width=0, fill_color=MINT,
                                 fill_opacity=0.85)
        ghigh.move_to(gbg.get_bottom(), aligned_edge=DOWN)
        val3 = T("0.81", size=20, color=MINT, weight="MEDIUM")
        val3.move_to(val2)
        self.play(Rotate(needle, angle=-TAU / 2,
                         about_point=dial.get_center()),
                  Transform(gfill, ghigh), ReplacementTransform(val2, val3),
                  run_time=2.4)

        # s02.9 — pushing surprise down shapes the parameters
        plab = DIN_T(f"{PARAMS_LABEL} PARAMETERS", size=TYPE["tag"], color=MINT)
        plab.move_to(wlab)
        self.cue("s02.9")
        self.play(ReplacementTransform(wlab, plab), run_time=0.8)
        self.play(Indicate(dial, scale_factor=1.05, color=MINT),
                  LaggedStart(*[c.animate.set_fill(MINT, 0.16) for c in wg],
                              lag_ratio=0.035), run_time=2.2)

        # s02.10 — grammar, style and world knowledge settle into the weights
        ghosts = VGroup(*[T(s, size=16, color=MINT) for s in
                          ["grammar", "style", "world knowledge"]])
        for g, pos in zip(ghosts, [[-5.75, -1.15, 0], [-2.85, -1.15, 0],
                                   [0.05, -1.15, 0]]):
            g.move_to(pos)
        self.cue("s02.10")
        self.play(LaggedStart(*[FadeIn(g, shift=DOWN * 0.15) for g in ghosts],
                              lag_ratio=0.18), run_time=1.6)
        self.play(LaggedStart(*[g.animate.shift(DOWN * 0.50).set_opacity(0.0)
                                for g in ghosts], lag_ratio=0.18), run_time=1.6)
        self.play(LaggedStart(*[c.animate.set_fill(MINT, 0.22) for c in wg[::3]],
                              lag_ratio=0.04), run_time=1.4)

        # s02.11 — a "noun" label tries to stick and slides off
        noun = pill("noun", color=DIM, size=15, stroke=DIM)
        noun.move_to(wg.get_center()).shift(UP * 0.25)
        self.cue("s02.11")
        self.play(FadeIn(noun, shift=DOWN * 0.12), run_time=0.6)
        self.play(noun.animate.shift(DOWN * 0.9 + LEFT * 0.7).set_opacity(0.0),
                  run_time=1.2)

        # s02.12 — training runs the prediction at every position at once
        marks = VGroup(*[glow_dot(p.get_top() + UP * 0.30, MINT, r=0.05)
                         for p in row])
        par_lab = DIN_T("IN PARALLEL", size=TYPE["tag"], color=MINT)
        par_lab.next_to(row, DOWN, buff=0.32)
        self.cue("s02.12")
        self.play(FadeOut(more), FadeOut(more_lab),
                  FadeIn(marks, scale=0.6), run_time=1.0)
        self.play(Indicate(marks, scale_factor=1.12, color=MINT), run_time=0.8)
        self.play(Write(par_lab), run_time=1.0)

        # s02.13 — generation reveals one new token at a time
        self.cue("s02.13")
        self.play(LaggedStart(*[FadeOut(m, shift=RIGHT * 0.12)
                                for m in marks[:-1]], lag_ratio=0.08),
                  run_time=1.3)
        one_lab = DIN_T("ONE AT A TIME", size=TYPE["tag"], color=MINT)
        one_lab.move_to(par_lab)
        revealed = token_pill(words[-1], size=18)
        revealed.move_to(row[-1])
        self.play(FadeOut(mask), FadeIn(revealed), FadeOut(hidden),
                  ReplacementTransform(par_lab, one_lab),
                  run_time=1.3)

        # s02.14 — the block silhouette ghosts in behind the objective card
        stack = VGroup()
        for _ in range(N_LAYERS):
            stack.add(RoundedRectangle(corner_radius=0.05, width=3.0,
                                       height=0.17, stroke_color=WHITE,
                                       stroke_opacity=0.06, stroke_width=1.0,
                                       fill_color=WHITE, fill_opacity=0.025))
        stack.arrange(DOWN, buff=0.145).move_to([0, 0, 0])
        self.cue("s02.14")
        self.play(*[FadeOut(m) for m in
                    (hdr, wg, wg_br, plab, dial, dticks, dlab, hlab, llab, needle, marks[-1],
                     card_g, obj, obj_sub, row, revealed, gbg, gfill, glab,
                     val3, one_lab)], run_time=0.9)
        self.play(LaggedStart(*[FadeIn(s) for s in stack], lag_ratio=0.05),
                  run_time=1.9)

        # s02.15 — the silhouette resolves to full colour as the card fades
        self.cue("s02.15")
        self.play(LaggedStart(*[s.animate.set_stroke(WHITE, opacity=0.18)
                                .set_fill(WHITE, 0.065) for s in stack],
                              lag_ratio=0.05), run_time=1.3)
        self.play(Indicate(stack, scale_factor=1.03, color=MINT),
                  run_time=0.9)

        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 3 ========
class Section03(Section):
    SEC = "sec03_tokens"
    CHAPTER = "03 · TOKENS"
    ACCENT = GOLD

    def construct(self):
        self._start_canvas_life()

        # s03.1 — text is refused by the matrix multiply; the tokenizer goes first
        self.cue("s03.1")
        hdr = self.section_header("TOKENS", "Text becomes IDs", run_time=0.9,
                                  ty=2.78, size=34)
        claim = T("Text cannot enter a matrix multiply", size=31, color=WHITE,
                  weight="BOLD")
        claim.move_to([0, 1.62, 0])
        self.play(Write(claim), run_time=1.3)

        txt = pill(TOKENS["sentence"], color=MUTED, size=17)
        txt.move_to([-3.55, 0.10, 0])
        mm = card(3.2, 1.30, edge=ROSE)
        mm.move_to([2.85, 0.10, 0])
        mm_lab = T("matrix multiply", size=17, color=MUTED)
        mm_lab.move_to(mm.get_center())
        blocked = dim_arrow(txt.get_right(), mm.get_left(), ROSE, 0.7, 2.4,
                            0.15, 0.12)
        cross = T("×", size=34, color=ROSE, weight="BOLD")
        cross.move_to(blocked.get_center())
        self.play(FadeIn(txt), run_time=0.6)
        self.play(Create(mm), Write(mm_lab), run_time=0.8)
        self.play(Create(blocked), FadeIn(cross), run_time=0.7)

        # s03.2 — the sentence splits into six subword chips with their IDs
        self.cue("s03.2")
        self.play(FadeOut(claim), FadeOut(blocked), FadeOut(cross),
                  FadeOut(mm), FadeOut(mm_lab), run_time=0.7)
        voc = DIN_T(f"VOCABULARY · {VOCAB_LABEL} TOKENS", size=TYPE["tag"],
                    color=GOLD)
        voc.move_to([0, 0.92, 0])
        chips = VGroup(*[token_pill(it["text"], str(it["id"]), color=GOLD)
                         for it in TOKENS["items"]])
        chips.arrange(RIGHT, buff=0.26).move_to([0, -0.35, 0])
        self.play(Write(voc), run_time=0.8)
        self.play(FadeOut(txt), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.14) for c in chips],
                              lag_ratio=0.22), run_time=2.6)
        self.play(LaggedStart(*[Indicate(c[1], scale_factor=1.12, color=GOLD)
                                for c in chips], lag_ratio=0.12),
                  run_time=1.6)

        # s03.3 — byte-pair encoding: the most frequent pair merges
        self.cue("s03.3")
        self.play(FadeOut(chips), FadeOut(voc), run_time=0.6)
        bpe = DIN_T("BYTE-PAIR ENCODING", size=TYPE["tag"], color=GOLD)
        bpe.move_to([0, 1.55, 0])
        a = token_pill("c", color=MUTED, size=20)
        b = token_pill("a", color=MUTED, size=20)
        plus = T("+", size=26, color=DIM, weight="BOLD")
        pair = VGroup(a, plus, b).arrange(RIGHT, buff=0.26)
        pair.move_to([-1.75, 0.35, 0])
        merged = token_pill("ca", color=GOLD, size=20)
        merged.move_to([2.05, 0.35, 0])
        arrow = dim_arrow(pair.get_right(), merged.get_left(), GOLD, 0.75, 2.4,
                          0.15, 0.14)
        stamp = DIN_T("NEW TOKEN", size=12, color=GOLD, tracked=False)
        stamp.next_to(merged, DOWN, buff=0.18)
        self.play(Write(bpe), run_time=0.7)
        self.play(LaggedStart(FadeIn(a), FadeIn(plus), FadeIn(b), lag_ratio=0.28),
                  run_time=0.9)
        self.play(Create(arrow), run_time=0.7)
        self.play(a.animate.move_to(merged.get_center()).set_opacity(0),
                  b.animate.move_to(merged.get_center()).set_opacity(0),
                  FadeOut(plus), run_time=1.0)
        self.play(FadeIn(merged, scale=0.7), Write(stamp), run_time=1.1)

        # s03.4 — a merge ladder builds from single characters to whole words
        self.cue("s03.4")
        self.play(FadeOut(bpe), FadeOut(pair), FadeOut(arrow), FadeOut(merged),
                  FadeOut(stamp), run_time=0.6)
        rungs = [
            (TOKENS["items"][1]["text"], GOLD,
             f"whole word · {TOKENS['items'][1]['id']}"),
            ("cat", WHITE, "merge"),
            ("ca   t", MUTED, "merge"),
            ("c   a   t", DIM, "characters"),
        ]
        ladder = VGroup()
        for lab, col, tag in rungs:
            row = VGroup(T(lab, size=21, color=col, weight="MEDIUM"),
                         DIN_T(tag, size=12, color=col, tracked=False))
            row.arrange(RIGHT, buff=0.55)
            ladder.add(row)
        ladder.arrange(DOWN, buff=0.40).move_to([0.35, -0.15, 0])
        up = dim_arrow(ladder[-1].get_left() + LEFT * 0.60,
                       ladder[0].get_left() + LEFT * 0.60, GOLD, 0.7, 2.4,
                       0.15, 0.10)
        self.play(LaggedStart(*[FadeIn(r, shift=UP * 0.12)
                                for r in ladder[::-1]], lag_ratio=0.30),
                  run_time=2.0)
        self.play(Create(up), run_time=0.7)
        self.play(Indicate(ladder[0], scale_factor=1.07, color=GOLD),
                  run_time=0.7)
        # s03.5 — common words stay whole, rare words break into known pieces
        self.cue("s03.5")
        self.play(FadeOut(ladder), FadeOut(up), run_time=0.5)
        self.play(FadeOut(hdr), run_time=0.5)
        common_lab = DIN_T("COMMON WORDS STAY WHOLE", size=TYPE["tag"],
                           color=GOLD)
        common_lab.move_to([0, 1.95, 0])
        commons = VGroup(*[token_pill(it["text"], str(it["id"]), color=GOLD)
                           for it in TOKENS["items"][:3]])
        commons.arrange(RIGHT, buff=0.32).move_to([0, 1.15, 0])
        rare_lab = DIN_T("RARE WORDS BREAK APART", size=TYPE["tag"], color=MUTED)
        rare_lab.move_to([0, -0.25, 0])
        rare = token_pill("unbelievable", color=MUTED, size=18)
        rare.move_to([0, -1.20, 0])
        frags = VGroup(*[token_pill(w, color=MINT, size=16)
                         for w in ["un", "believ", "able"]])
        frags.arrange(RIGHT, buff=0.26).move_to([0, -1.20, 0])
        self.play(Write(common_lab),
                  LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in commons],
                              lag_ratio=0.22), run_time=1.5)
        self.play(Indicate(commons, scale_factor=1.05, color=GOLD), run_time=0.9)
        self.play(Write(rare_lab), FadeIn(rare), run_time=1.0)
        self.play(ReplacementTransform(rare, frags), run_time=1.0)

        # s03.6 — characters make long sequences, whole words explode the vocab
        self.cue("s03.6")
        self.play(*[FadeOut(m) for m in
                    (common_lab, commons, rare_lab, frags)], run_time=0.7)
        left = card(5.7, 3.0, edge=MUTED)
        left.move_to([-3.45, 0.55, 0])
        left_lab = DIN_T("CHARACTERS", size=TYPE["tag"], color=MUTED)
        left_lab.move_to([left.get_center()[0], 1.62, 0])
        char_bar = VGroup(*[RoundedRectangle(corner_radius=0.025, width=0.115,
                                             height=0.42, stroke_width=0,
                                             fill_color=MUTED,
                                             fill_opacity=0.80)
                            for _ in range(26)])
        char_bar.arrange(RIGHT, buff=0.048)
        char_bar.move_to([left.get_center()[0], 0.72, 0])
        left_cap = T("sequences grow very long", size=15, color=MUTED)
        left_cap.move_to([left.get_center()[0], -0.05, 0])
        right = card(5.7, 3.0, edge=ROSE)
        right.move_to([3.45, 0.55, 0])
        right_lab = DIN_T("WHOLE WORDS", size=TYPE["tag"], color=ROSE)
        right_lab.move_to([right.get_center()[0], 1.62, 0])
        word_tbl = grid(6, 14, cell=0.205, gap=0.045)
        word_tbl.set_stroke(ROSE, opacity=0.28).set_fill(ROSE, 0.10)
        word_tbl.move_to([right.get_center()[0], 0.72, 0])
        word_br = matrix_brackets(word_tbl, ROSE, pad=0.05, tick=0.10)
        right_cap = T("the vocabulary explodes", size=15, color=ROSE)
        right_cap.move_to([right.get_center()[0], -0.05, 0])
        self.play(Create(left), Write(left_lab), FadeIn(char_bar),
                  Write(left_cap), run_time=1.6)
        self.play(Create(right), Write(right_lab), FadeIn(word_tbl), Create(word_br),
                  Write(right_cap), run_time=1.6)
        self.play(Indicate(char_bar, scale_factor=1.05, color=MUTED),
                  Indicate(word_tbl, scale_factor=1.04, color=ROSE),
                  run_time=1.4)

        # s03.7 — subwords sit in between and cover any input
        self.cue("s03.7")
        self.play(left.animate.set_stroke(opacity=0.22),
                  right.animate.set_stroke(opacity=0.22),
                  left_lab.animate.set_opacity(0.35),
                  right_lab.animate.set_opacity(0.35),
                  char_bar.animate.set_opacity(0.28),
                  word_tbl.animate.set_opacity(0.28),
                  word_br.animate.set_stroke(opacity=0.22),
                  FadeOut(left_cap), FadeOut(right_cap), run_time=0.7)
        sub_lab = DIN_T("SUBWORDS", size=TYPE["tag"], color=GOLD)
        sub_lab.move_to([0, -1.18, 0])
        mid = VGroup(*[token_pill(it["text"], str(it["id"]), color=GOLD)
                       for it in TOKENS["items"]])
        mid.arrange(RIGHT, buff=0.22).move_to([0, -1.95, 0])
        pieces = VGroup(*[token_pill(w, color=MINT, size=15)
                          for w in ["un", "seen", "-", "word"]])
        pieces.arrange(RIGHT, buff=0.18).move_to([-1.15, -2.95, 0])
        unseen = token_pill("unseen-word", color=MINT, size=16)
        unseen.move_to(pieces.get_center())
        pair_lab = T("even words never seen", size=15, color=MINT)
        pair_lab.next_to(pieces, RIGHT, buff=0.34)
        self.play(Write(sub_lab), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in mid],
                              lag_ratio=0.18), run_time=1.8)
        self.play(FadeIn(unseen), run_time=0.6)
        self.play(ReplacementTransform(unseen, pieces), Write(pair_lab),
                  run_time=1.1)

        # s03.8 — their IDs come straight from the vocabulary table
        self.cue("s03.8")
        self.play(FadeOut(sub_lab), FadeOut(pieces), FadeOut(pair_lab),
                  run_time=0.6)
        id_src = DIN_T("IDS FROM THE VOCABULARY TABLE", size=TYPE["tag"],
                       color=GOLD)
        id_src.move_to([0, -1.18, 0])
        self.play(Write(id_src), run_time=0.7)
        self.play(LaggedStart(*[c[2].animate.set_color(GOLD) for c in mid],
                              lag_ratio=0.12), run_time=1.1)
        # s03.9 — cat becomes 8415, mat becomes 5634
        self.cue("s03.9")
        cat_i, mat_i = 1, len(mid) - 1
        cat_id = T(f"{TOKENS['items'][cat_i]['id']}", size=30, color=GOLD,
                   weight="BOLD")
        cat_id.move_to([0, 0.25, 0])
        cat_cap = T("cat", size=17, color=MUTED)
        cat_cap.next_to(cat_id, LEFT, buff=0.34)
        mat_id = T(f"{TOKENS['items'][mat_i]['id']}", size=30, color=GOLD,
                   weight="BOLD")
        mat_id.move_to([0, -0.75, 0])
        mat_cap = T("mat", size=17, color=MUTED)
        mat_cap.next_to(mat_id, LEFT, buff=0.34)
        self.play(FadeOut(left), FadeOut(right), FadeOut(left_lab),
                  FadeOut(right_lab), FadeOut(char_bar), FadeOut(word_tbl),
                  FadeOut(word_br), run_time=0.5)
        self.play(Indicate(mid[cat_i], scale_factor=1.12, color=GOLD),
                  run_time=1.4)
        self.play(Write(cat_id), Write(cat_cap), run_time=1.0)
        self.play(Indicate(mid[mat_i], scale_factor=1.12, color=GOLD),
                  run_time=1.4)
        self.play(Write(mat_id), Write(mat_cap), run_time=1.0)
        self.play(Indicate(cat_id, scale_factor=1.06, color=GOLD),
                  Indicate(mat_id, scale_factor=1.06, color=GOLD),
                  run_time=0.7)

        # s03.10 — each ID is only a row number, with no meaning of its own
        self.cue("s03.10")
        self.play(FadeOut(id_src), FadeOut(cat_cap), FadeOut(mat_cap),
                  FadeOut(cat_id), FadeOut(mat_id), run_time=0.5)
        lifted = VGroup(*[c[2].copy() for c in mid])
        lifted.arrange(RIGHT, buff=0.62).move_to([0, -1.45, 0])
        self.play(*[ReplacementTransform(mid[i][2], lifted[i])
                    for i in range(len(mid))],
                  *[FadeOut(c[0]) for c in mid],
                  *[FadeOut(c[1]) for c in mid], run_time=1.4)
        self.play(*[m.animate.scale(1.6) for m in lifted], run_time=0.5)
        idx_lab = DIN_T("BARE ROW INDICES", size=TYPE["tag"], color=GOLD)
        idx_lab.move_to([0, -0.35, 0])
        self.play(Write(idx_lab), run_time=0.6)

        # s03.11 — the tokenizer knows nothing about cats
        self.cue("s03.11")
        self.play(FadeOut(lifted), FadeOut(idx_lab), run_time=0.4)
        face = Circle(radius=0.58, stroke_color=GOLD, stroke_opacity=0.85,
                      stroke_width=2.2, fill_color=GOLD, fill_opacity=0.10)
        face.move_to([-2.45, -0.15, 0])
        ear1 = Circle(radius=0.21, stroke_color=GOLD, stroke_opacity=0.85,
                      stroke_width=2.0, fill_color=GOLD, fill_opacity=0.18)
        ear2 = ear1.copy()
        ear1.move_to(face.get_center() + LEFT * 0.38 + UP * 0.50)
        ear2.move_to(face.get_center() + RIGHT * 0.38 + UP * 0.50)
        eye1 = Dot(face.get_center() + LEFT * 0.21 + UP * 0.11, radius=0.058,
                   color=GOLD)
        eye2 = Dot(face.get_center() + RIGHT * 0.21 + UP * 0.11, radius=0.058,
                   color=GOLD)
        nose = Dot(face.get_center() + DOWN * 0.12, radius=0.052, color=GOLD)
        cat = VGroup(face, ear1, ear2, eye1, eye2, nose)
        tbox = card(3.4, 1.35, edge=MUTED)
        tbox.move_to([2.15, -0.15, 0])
        tbox_lab = T("tokenizer", size=18, color=MUTED)
        tbox_lab.move_to(tbox.get_center())
        shrug = T("?", size=30, color=DIM, weight="BOLD")
        shrug.next_to(cat, RIGHT, buff=0.55)
        self.play(FadeIn(cat, scale=0.8), Create(tbox), Write(tbox_lab),
                  run_time=1.1)
        self.play(Write(shrug), run_time=0.5)
        self.play(FadeOut(cat), FadeOut(shrug), run_time=0.6)

        # s03.12 — its only job is to cut text into pieces the model has a row for
        self.cue("s03.12")
        self.play(FadeOut(tbox), FadeOut(tbox_lab), run_time=0.5)
        raw = T(TOKENS["sentence"], size=26, color=MUTED)
        raw.move_to([0, 0.85, 0])
        blade1 = Line(ORIGIN, UP * 0.40 + RIGHT * 0.15, stroke_color=ROSE,
                      stroke_width=2.6)
        blade2 = Line(ORIGIN, DOWN * 0.40 + RIGHT * 0.15, stroke_color=ROSE,
                      stroke_width=2.6)
        h1 = Circle(radius=0.11, stroke_color=ROSE, stroke_width=2.2)
        h1.move_to(blade1.get_end() + UP * 0.13 + RIGHT * 0.10)
        h2 = Circle(radius=0.11, stroke_color=ROSE, stroke_width=2.2)
        h2.move_to(blade2.get_end() + DOWN * 0.13 + RIGHT * 0.10)
        scissors = VGroup(blade1, blade2, h1, h2)
        scissors.move_to(raw.get_left() + DOWN * 0.80)
        cut_row = VGroup(*[token_pill(it["text"], size=15, color=GOLD)
                           for it in TOKENS["items"]])
        cut_row.arrange(RIGHT, buff=0.18).move_to([0, -0.95, 0])
        self.play(Write(raw), run_time=0.7)
        self.play(FadeIn(scissors), run_time=0.4)
        self.play(scissors.animate.move_to(raw.get_right() + DOWN * 0.80),
                  run_time=0.9)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.10) for c in cut_row],
                              lag_ratio=0.12), run_time=0.7)
        # s03.13 — the config lists fewer rows than the tokenizer holds
        self.cue("s03.13")
        self.play(FadeOut(raw), FadeOut(scissors), FadeOut(cut_row),
                  run_time=0.6)
        counter = card(8.6, 3.2, edge=GOLD)
        counter.move_to([0, -0.25, 0])
        ct_lab = DIN_T("VOCABULARY ROWS", size=TYPE["tag"], color=GOLD)
        ct_lab.move_to([0, 1.05, 0])
        cfg_num = T(VOCAB_CONFIG_LABEL, size=40, color=MUTED, weight="BOLD")
        cfg_num.move_to([-2.35, -0.15, 0])
        cfg_cap = DIN_T("CONFIG", size=13, color=MUTED)
        cfg_cap.next_to(cfg_num, DOWN, buff=0.20)
        tok_num = T(VOCAB_LABEL, size=40, color=GOLD, weight="BOLD")
        tok_num.move_to([2.35, -0.15, 0])
        tok_cap = DIN_T("TOKENIZER", size=13, color=GOLD)
        tok_cap.next_to(tok_num, DOWN, buff=0.20)
        diff = T(f"+{VOCAB_DELTA}", size=28, color=MINT, weight="BOLD")
        diff.move_to([0, -0.15, 0])
        self.play(Create(counter), Write(ct_lab), run_time=1.1)
        self.play(Write(cfg_num), Write(cfg_cap), run_time=1.3)
        self.play(Write(tok_num), Write(tok_cap), run_time=1.3)
        self.play(Write(diff), run_time=1.1)
        self.play(Indicate(diff, scale_factor=1.15, color=MINT), run_time=0.6)

        # s03.14 — the embedding table sizes itself to the tokenizer
        self.cue("s03.14")
        self.play(FadeOut(counter), FadeOut(ct_lab), FadeOut(cfg_num),
                  FadeOut(cfg_cap), FadeOut(diff), FadeOut(tok_num),
                  FadeOut(tok_cap), run_time=0.5)
        tbl = grid(10, 14, cell=0.205, gap=0.045)
        tbl.set_stroke(GOLD, opacity=0.26).set_fill(GOLD, 0.09)
        tbl.move_to([0, -0.15, 0])
        tbl_br = matrix_brackets(tbl, GOLD, pad=0.06, tick=0.12)
        tbl_cap = DIN_T("EMBEDDING TABLE", size=TYPE["tag"], color=GOLD)
        tbl_cap.next_to(tbl, UP, buff=0.28)
        grow = grid(1, 14, cell=0.205, gap=0.045)
        grow.set_stroke(MINT, opacity=0.34).set_fill(MINT, 0.16)
        grow.move_to(tbl.get_bottom() + DOWN * 0.13)
        grow_lab = T(f"+{VOCAB_DELTA} rows", size=16, color=MINT)
        grow_lab.next_to(grow, RIGHT, buff=0.32)
        stamp_num = T(VOCAB_LABEL, size=52, color=GOLD, weight="BOLD")
        stamp_num.move_to([0, 2.35, 0])
        stamp_cap = DIN_T("ROWS THE EMBEDDING TABLE HOLDS", size=13,
                          color=MUTED)
        stamp_cap.next_to(stamp_num, DOWN, buff=0.16)
        self.play(Write(tbl_cap), FadeIn(tbl), Create(tbl_br), run_time=0.9)
        self.play(FadeIn(grow), Write(grow_lab), run_time=0.8)
        self.play(Write(stamp_num), Write(stamp_cap), run_time=0.8)
        self.play(Indicate(stamp_num, scale_factor=1.08, color=GOLD),
                  run_time=0.5)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 4 ========
class Section04(Section):
    SEC = "sec04_embedding"
    CHAPTER = "04 · EMBEDDINGS"
    ACCENT = BLUE

    def construct(self):
        self._start_canvas_life()

        def num_cell(v, edge_color, text_color=WHITE):
            bx = RoundedRectangle(corner_radius=0.07, width=0.85, height=0.40,
                                  stroke_color=WHITE, stroke_opacity=0.13,
                                  stroke_width=1.1, fill_color=WHITE,
                                  fill_opacity=0.05)
            notch = Rectangle(width=0.05, height=0.26, stroke_width=0,
                              fill_color=edge_color, fill_opacity=0.95)
            notch.move_to(bx.get_left() + RIGHT * 0.06)
            tx = Text(v, font=DIN, font_size=pt(15), color=text_color,
                      weight="BOLD").move_to(bx)
            return VGroup(bx, notch, tx)

        # s04.1 — an ID is a row number; the row it points to carries the meaning
        self.cue("s04.1")
        hdr = self.section_header("EMBEDDINGS", "IDs become vectors", run_time=0.7,
                                  ty=2.78, size=34)
        mat = grid(8, 6, cell=0.235, gap=0.035)
        mat.set_stroke(WHITE, opacity=0.18).set_fill(WHITE, 0.055)
        mat.move_to([4.15, 0.30, 0])
        mat_br = matrix_brackets(mat, BLUE, pad=0.06, tick=0.12)
        mat_label = TY(f'E in RR^({VOCAB} times {D_MODEL})', size=21, color=MUTED)
        mat_label.next_to(mat, DOWN, buff=0.24)
        idx = token_pill(TOKENS["items"][1]["text"],
                         str(TOKENS["items"][1]["id"]), color=GOLD)
        idx.move_to([-4.15, 1.95, 0])
        idx_arrow = dim_arrow(idx.get_right() + DOWN * 0.30,
                              mat.get_left() + UP * 0.35, GOLD, 0.75, 2.4,
                              0.15, 0.12)
        self.play(Create(mat), Create(mat_br), run_time=0.7)
        self.play(Write(mat_label), FadeIn(idx), run_time=0.9)
        self.play(Create(idx_arrow), run_time=0.6)
        self.play(Indicate(idx, scale_factor=1.08, color=GOLD), run_time=0.6)

        # s04.2 — the matrix appears row by row and the cat row lights up
        self.cue("s04.2")
        self.play(FadeOut(hdr), run_time=0.5)
        self.play(LaggedStart(*[Indicate(mat[6 * r: 6 * r + 6],
                                         scale_factor=1.10, color=MUTED)
                                for r in range(8)], lag_ratio=0.10),
                  run_time=1.6)
        cat_row = mat[6:12]
        hl = RoundedRectangle(corner_radius=0.06, width=mat.width + 0.18,
                              height=0.355, stroke_color=GOLD, stroke_width=2.0,
                              stroke_opacity=0.95, fill_color=GOLD,
                              fill_opacity=0.18)
        hl.move_to(cat_row.get_center())
        self.play(GrowFromCenter(hl), run_time=1.1)
        row_lab = T("one row per vocabulary item", size=16, color=MUTED)
        row_lab.move_to([-2.30, 0.30, 0])
        self.play(Write(row_lab), run_time=0.8)

        # s04.3 — each row holds D_MODEL numbers
        self.cue("s04.3")
        dim_lab = T("numbers in one row", size=18, color=MUTED)
        dim_lab.move_to([-2.30, -0.85, 0])
        dim_num = T(f"{D_MODEL:,}", size=46, color=BLUE, weight="BOLD")
        dim_num.move_to([-2.30, -1.65, 0])
        self.play(Write(dim_lab), run_time=1.3)
        self.play(Write(dim_num), run_time=1.1)
        self.play(Indicate(hl, scale_factor=1.05, color=BLUE), run_time=1.0)
        self.play(Indicate(dim_num, scale_factor=1.08, color=BLUE), run_time=1.1)

        # s04.4 — the ID picks the row, and the row becomes the token's vector
        self.cue("s04.4")
        self.play(FadeOut(idx_arrow), FadeOut(row_lab), FadeOut(dim_lab),
                  FadeOut(dim_num), run_time=0.6)
        vec_vals = ["0.42", "-1.13", "0.07", "0.88", "-0.52", "…"]
        vec_cells = VGroup()
        for v in vec_vals:
            bx = RoundedRectangle(corner_radius=0.08, width=0.62, height=0.5,
                                  stroke_color=WHITE, stroke_opacity=0.16,
                                  stroke_width=1.1, fill_color=WHITE,
                                  fill_opacity=0.05)
            notch = Rectangle(width=0.055, height=0.34, stroke_width=0,
                              fill_color=BLUE, fill_opacity=0.95)
            notch.move_to(bx.get_left() + RIGHT * 0.07)
            vec_cells.add(VGroup(bx, notch,
                                 Text(v, font=DIN, font_size=pt(14),
                                      color=WHITE, weight="BOLD").move_to(bx)))
        vec_cells.arrange(RIGHT, buff=0.10).move_to([-1.45, 0.55, 0])
        vec_label = TY('x = E[t_2]', size=24, color=BLUE)
        vec_label.next_to(vec_cells, UP, buff=0.20)
        pull = dim_arrow(hl.get_left(), vec_cells.get_right() + UP * 0.10,
                         BLUE, 0.7, 2.4, 0.15, 0.12)
        self.play(Create(pull), run_time=0.8)
        self.play(ReplacementTransform(cat_row.copy(),
                                       VGroup(*[v[0] for v in vec_cells])),
                  LaggedStart(*[GrowFromCenter(VGroup(v[1], v[2]))
                                for v in vec_cells], lag_ratio=0.08),
                  Write(vec_label), run_time=2.3)
        self.play(Indicate(vec_cells, scale_factor=1.08, color=BLUE),
                  run_time=1.2)
        # s04.5 — no computation here yet, only a learned table
        self.cue("s04.5")
        learned = pill("learned table · no computation", color=MINT, size=15)
        learned.move_to([-1.45, -0.95, 0])
        t1 = Line(ORIGIN, RIGHT * 0.18 + UP * 0.20, stroke_color=MINT,
                  stroke_width=3.2)
        t2 = Line(RIGHT * 0.18 + UP * 0.20, RIGHT * 0.50 + DOWN * 0.26,
                  stroke_color=MINT, stroke_width=3.2)
        tick = VGroup(t1, t2).next_to(mat, LEFT, buff=0.38)
        self.play(Write(learned), run_time=1.2)
        self.play(FadeIn(tick, scale=0.7), run_time=0.7)
        self.play(Indicate(learned, scale_factor=1.05, color=MINT), run_time=0.6)

        # s04.6 — training moves rows so similar tokens land near each other
        self.cue("s04.6")
        self.play(FadeOut(learned), run_time=0.5)
        space = Axes(x_range=[-1, 1, 1], y_range=[-1, 1, 1], x_length=3.9,
                     y_length=2.15, tips=False,
                     axis_config={"stroke_color": MUTED, "stroke_opacity": 0.38,
                                  "stroke_width": 1.3})
        space.move_to([0.35, -2.15, 0])
        sgrid = VGroup()
        for v in (-0.5, 0.5):
            sgrid.add(Line(space.c2p(-1, v), space.c2p(1, v), stroke_color=WHITE,
                           stroke_opacity=0.07, stroke_width=0.9))
            sgrid.add(Line(space.c2p(v, -1), space.c2p(v, 1), stroke_color=WHITE,
                           stroke_opacity=0.07, stroke_width=0.9))
        pts = {"cat": (-0.65, 0.35, BLUE), "kitten": (-0.28, 0.82, BLUE),
               "dog": (-0.62, -0.42, BLUE), "car": (0.45, 0.58, ROSE),
               "truck": (0.78, 0.32, ROSE), "apple": (0.42, -0.52, MINT)}
        scatter = {"cat": (-0.75, -0.35), "kitten": (0.55, -0.55),
                   "dog": (-0.15, 0.65), "car": (-0.55, 0.35),
                   "truck": (0.15, -0.15), "apple": (0.75, 0.05)}
        dots, homes = [], []
        for name, (x, y, c) in pts.items():
            d = Dot(space.c2p(*scatter[name]), radius=0.075, color=c)
            lab = T(name, size=13, color=lighter(c, 0.2)).next_to(d, DOWN,
                                                                 buff=0.06)
            grp = VGroup(d, lab)
            dots.append(grp)
            homes.append(space.c2p(x, y) + (grp.get_center() - d.get_center()))
        space_cap = DIN_T(f"{D_MODEL:,}-D MEANING SPACE — 2-D SKETCH",
                          size=TYPE["micro"], color=MUTED)
        space_cap.next_to(space, DOWN, buff=0.16)
        self.play(FadeIn(sgrid), Create(space), run_time=0.8)
        self.play(*[FadeIn(d) for d in dots], run_time=1.0)
        self.play(*[dots[i].animate.move_to(homes[i]) for i in range(len(dots))],
                  run_time=1.5)
        self.play(Write(space_cap), run_time=0.7)

        # s04.7 — directions in that space carry meaning
        self.cue("s04.7")
        x_arrow = Arrow(space.c2p(-1, 0), space.c2p(1, 0), buff=0,
                        stroke_color=WHITE, fill_color=WHITE,
                        stroke_width=1.6, tip_length=0.14)
        y_arrow = Arrow(space.c2p(0, -1), space.c2p(0, 1), buff=0,
                        stroke_color=WHITE, fill_color=WHITE,
                        stroke_width=1.6, tip_length=0.14)
        x_lab = T("direction 1", size=13, color=MUTED)
        x_lab.next_to(x_arrow, RIGHT, buff=0.12)
        y_lab = T("direction 2", size=13, color=MUTED)
        y_lab.next_to(y_arrow, UP, buff=0.12)
        geo_note = T("each axis is one direction in the meaning space",
                     size=15, color=MUTED)
        geo_note.move_to([0.35, -0.25, 0])
        self.play(Create(x_arrow), Create(y_arrow), run_time=1.2)
        self.play(Write(x_lab), Write(y_lab), run_time=1.0)
        self.play(Write(geo_note), run_time=1.0)
        self.play(Indicate(space, scale_factor=1.04, color=BLUE), run_time=0.6)

        # s04.8 — an illustrative sketch: cat near kitten, far from car
        self.cue("s04.8")
        cat_d, kitten_d, car_d = dots[0][0], dots[1][0], dots[3][0]
        orig_pt = space.c2p(0, 0)
        v_cat = Arrow(orig_pt, cat_d.get_center(), buff=0.05, stroke_color=BLUE,
                      fill_color=BLUE, stroke_width=2.0, tip_length=0.11, stroke_opacity=0.65)
        v_kit = Arrow(orig_pt, kitten_d.get_center(), buff=0.05, stroke_color=BLUE,
                      fill_color=BLUE, stroke_width=2.0, tip_length=0.11, stroke_opacity=0.65)
        v_car = Arrow(orig_pt, car_d.get_center(), buff=0.05, stroke_color=ROSE,
                      fill_color=ROSE, stroke_width=2.0, tip_length=0.11, stroke_opacity=0.55)
        link = Line(cat_d.get_center(), kitten_d.get_center(),
                    stroke_color=BLUE, stroke_width=2.6, stroke_opacity=0.9)
        far = Line(cat_d.get_center(), car_d.get_center(),
                   stroke_color=ROSE, stroke_width=2.0, stroke_opacity=0.55)
        near_lab = T("near", size=14, color=BLUE)
        near_lab.move_to(link.get_center() + LEFT * 0.32)
        far_lab = T("far", size=14, color=ROSE)
        far_lab.move_to(far.get_center() + UP * 0.22 + RIGHT * 0.45)
        sketch_tag = DIN_T("ILLUSTRATIVE SKETCH", size=12, color=DIM)
        sketch_tag.move_to([-4.25, -1.35, 0])
        self.play(Create(v_cat), Create(v_kit), Create(link), Write(near_lab), run_time=1.1)
        self.play(Create(v_car), Create(far), Write(far_lab), run_time=1.1)
        self.play(Write(sketch_tag), run_time=0.7)
        self.play(Indicate(cat_d, scale_factor=1.25, color=BLUE),
                  Indicate(kitten_d, scale_factor=1.25, color=BLUE),
                  run_time=0.5)

        # s04.9 — the learned table embeds each token by how it is used
        self.cue("s04.9")
        self.play(FadeOut(v_cat), FadeOut(v_kit), FadeOut(v_car),
                  FadeOut(link), FadeOut(far), FadeOut(far_lab),
                  FadeOut(near_lab), run_time=0.6)
        extra = [("bird", -0.25, -0.68, BLUE), ("van", 0.35, -0.28, ROSE),
                 ("pear", 0.82, -0.55, MINT), ("bus", 0.80, 0.72, ROSE)]
        newcomers = []
        for name, x, y, c in extra:
            d = Dot(space.c2p(x, y), radius=0.065, color=c)
            lab = T(name, size=12, color=lighter(c, 0.2)).next_to(d, DOWN,
                                                                 buff=0.06)
            newcomers.append(VGroup(d, lab))
        # Multi-animation play, not LaggedStart: AnimationGroup leaves a wrapper
        # Group in the scene whose bbox spans every point, and FadeOut of the
        # children does not remove it.
        self.play(*[FadeIn(n, shift=UP * 0.10) for n in newcomers], run_time=2.4)
        self.play(*[Indicate(n, scale_factor=1.10, color=BLUE)
                    for n in newcomers], run_time=1.0)
        # s04.10 — the vector is a point in D_MODEL dimensions
        self.cue("s04.10")
        cloud = VGroup()
        rng = np.random.default_rng(7)
        for _ in range(160):
            p = rng.normal(0, 1, 3)
            p = p / (np.linalg.norm(p) + 1e-9) * (0.8 + 0.5 * rng.random())
            d = Dot(point=[p[0] * 3.1, p[1] * 0.90 - 1.25, 0], radius=0.035,
                    color=WHITE)
            d.set_fill(WHITE, opacity=0.10 + 0.22 * rng.random())
            cloud.add(d)
        hd_lab = T(f"a point in {D_MODEL:,} dimensions", size=22, color=WHITE,
                   weight="MEDIUM")
        hd_lab.move_to([0, -0.10, 0])
        self.play(FadeOut(x_arrow), FadeOut(y_arrow), FadeOut(x_lab),
                  FadeOut(y_lab), FadeOut(geo_note), FadeOut(sketch_tag),
                  FadeOut(space_cap), FadeOut(sgrid), run_time=0.7)
        self.play(FadeOut(space), FadeOut(*dots), FadeOut(*newcomers),
                  run_time=0.8)
        self.play(FadeIn(cloud, scale=1.3), run_time=1.3)
        self.play(Write(hd_lab), run_time=0.9)
        self.play(Indicate(cloud, scale_factor=1.05, color=BLUE), run_time=0.9)

        # s04.11 — two dimensions stand in for the rest
        self.cue("s04.11")
        self.play(FadeOut(hd_lab), run_time=0.5)
        flat = VGroup(*[d.copy().move_to([d.get_center()[0],
                                          -2.15 + d.get_center()[1] * 0.06, 0])
                        for d in cloud])
        self.play(Transform(cloud, flat), run_time=1.2)
        self.play(Create(space), Write(space_cap), run_time=1.0)
        honest = pill("2-D stands in for the rest", color=MINT, size=15)
        honest.move_to([-3.90, -0.55, 0])
        self.play(Write(honest), run_time=0.9)

        # s04.12 — from here the model sees only vectors, never characters
        self.cue("s04.12")
        self.play(FadeOut(mat), FadeOut(mat_br), FadeOut(mat_label), FadeOut(hl), FadeOut(idx),
                  FadeOut(tick), FadeOut(pull),
                  FadeOut(vec_label), FadeOut(cloud), FadeOut(space),
                  FadeOut(space_cap), FadeOut(honest), run_time=0.9)
        hero = vec_cells.copy().move_to([0, 0.35, 0]).scale(1.3)
        self.play(ReplacementTransform(vec_cells, hero), run_time=1.0)
        hero_halo = glow_ring(hero[0][0], BLUE)
        self.play(FadeIn(hero_halo),
                  Indicate(hero, scale_factor=1.06, color=BLUE), run_time=1.1)

        # s04.13 — the vector rides down through every block
        self.cue("s04.13")
        self.play(FadeOut(hero_halo), FadeOut(hero), run_time=0.5)
        rail = Rectangle(width=0.34, height=6.4, stroke_color=WHITE,
                         stroke_opacity=0.16, stroke_width=1.2,
                         fill_color=WHITE, fill_opacity=0.045)
        rail.move_to([-2.85, -0.15, 0])
        rail_lines = VGroup()
        for dx in (-0.10, 0.0, 0.10):
            rail_lines.add(Line(rail.get_top() + DOWN * 0.22 + RIGHT * dx,
                                rail.get_bottom() + UP * 0.22 + RIGHT * dx,
                                stroke_color=WHITE, stroke_opacity=0.10,
                                stroke_width=1.0))
        top_y = rail.get_top()[1] - 0.22
        bot_y = rail.get_bottom()[1] + 0.22
        blocks = VGroup()
        for i in range(N_LAYERS):
            frac = i / (N_LAYERS - 1)
            b = Rectangle(width=0.52, height=0.10, stroke_color=WHITE,
                          stroke_opacity=0.16, stroke_width=0.9,
                          fill_color=WHITE, fill_opacity=0.06)
            b.move_to([rail.get_center()[0] + 0.72,
                       top_y * (1 - frac) + bot_y * frac, 0])
            blocks.add(b)
        x16 = pill(f"× {N_LAYERS} blocks", color=WHITE, size=15)
        x16.move_to([3.90, 2.35, 0])
        self.play(Create(rail), Create(rail_lines), run_time=1.1)
        self.play(LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.05),
                  run_time=1.4)
        self.play(Write(x16), run_time=0.7)
        rider = glow_dot(rail.get_top() + DOWN * 0.35, BLUE, r=0.11)
        self.play(FadeIn(rider), run_time=0.5)

        # s04.14 — that carried vector is the residual stream
        self.cue("s04.14")
        band_lab = DIN_T("THE RESIDUAL STREAM", size=TYPE["tag"], color=BLUE)
        band_lab.next_to(rail, LEFT, buff=0.35)
        self.play(Write(band_lab), run_time=1.0)
        self.play(flow(rail_lines[1], BLUE), run_time=0.8)
        # s04.15 — attention and SwiGLU each read a normalized copy of the stream
        self.cue("s04.15")
        att = card(3.5, 0.85, edge=BLUE)
        att.move_to([1.55, 1.45, 0])
        att_lab = T("Attention sublayer", size=18, color=lighter(BLUE, 0.25))
        att_lab.move_to(att)
        ffn = card(3.5, 0.85, edge=VIOLET)
        ffn.move_to([1.55, -1.05, 0])
        ffn_lab = T("SwiGLU sublayer", size=18, color=lighter(VIOLET, 0.25))
        ffn_lab.move_to(ffn)
        rms_a = pill("RMSNorm", color=GOLD, size=15)
        rms_a.move_to([-1.15, 1.45, 0])
        rms_b = pill("RMSNorm", color=GOLD, size=15)
        rms_b.move_to([-1.15, -1.05, 0])
        tap1 = dim_arrow(rail.get_right() + UP * 1.60, rms_a.get_left(),
                         GOLD, 0.55, 2.0, 0.13, 0.10)
        tap2 = dim_arrow(rail.get_right() + UP * -0.50, rms_b.get_left(),
                         GOLD, 0.55, 2.0, 0.13, 0.10)
        link1 = dim_arrow(rms_a.get_right(), att.get_left(), BLUE, 0.7, 2.2,
                          0.14, 0.12)
        link2 = dim_arrow(rms_b.get_right(), ffn.get_left(), VIOLET, 0.7, 2.2,
                          0.14, 0.12)
        mean_chip = pill("mean square", color=MUTED, size=14)
        mean_chip.move_to([2.30, -2.55, 0])
        rms_chip = pill("RMS", color=WHITE, size=14, stroke=WHITE)
        rms_chip.move_to([4.75, -2.55, 0])
        mean2rms = dim_arrow(mean_chip.get_right(), rms_chip.get_left(),
                             op=0.45, w=2.0, tip=0.12, buff=0.12)
        root = operator_on_arrow("√", mean2rms, size=22)
        self.play(FadeOut(blocks), Create(att), Write(att_lab), Create(tap1),
                  Create(link1), FadeIn(rms_a), run_time=1.3)
        self.play(Create(ffn), Write(ffn_lab), Create(tap2), Create(link2),
                  FadeIn(rms_b), run_time=1.3)
        self.play(Write(mean_chip), Write(rms_chip), Create(mean2rms),
                  Write(root), run_time=1.2)
        self.play(Indicate(rms_a, scale_factor=1.06, color=GOLD),
                  Indicate(rms_b, scale_factor=1.06, color=GOLD),
                  run_time=0.8)

        # s04.16 — each writes its result back onto the stream by addition
        self.cue("s04.16")
        p1_center = rail.get_right() + RIGHT * 0.26 + UP * 0.45
        plus1 = VGroup(Circle(radius=0.19, stroke_color=MINT, stroke_width=1.6,
                              fill_color=BLACK,
                              fill_opacity=1.0).move_to(p1_center),
                       T("+", size=18, color=MINT,
                         weight="BOLD").move_to(p1_center))
        add1 = data_arrow(att.get_bottom() + LEFT * 1.2,
                          plus1[0].get_right(), MINT, w=2.6)
        p2_center = rail.get_right() + RIGHT * 0.26 + DOWN * 1.95
        plus2 = VGroup(Circle(radius=0.19, stroke_color=MINT, stroke_width=1.6,
                              fill_color=BLACK,
                              fill_opacity=1.0).move_to(p2_center),
                       T("+", size=18, color=MINT,
                         weight="BOLD").move_to(p2_center))
        add2 = data_arrow(ffn.get_bottom() + LEFT * 1.2,
                          plus2[0].get_right(), MINT, w=2.6)
        self.play(Create(add1), GrowFromCenter(plus1), run_time=1.2)
        self.play(Create(add2), GrowFromCenter(plus2), run_time=1.2)

        # s04.17 — this pre-norm shape is the structure of every decoder block
        self.cue("s04.17")
        frame = DashedVMobject(RoundedRectangle(corner_radius=0.16, width=6.6,
                                                height=3.1, stroke_color=GOLD,
                                                stroke_width=1.6),
                               num_dashes=40)
        frame.move_to([0.85, 0.20, 0])
        stamp = DIN_T("PRE-NORM", size=TYPE["tag"], color=GOLD)
        stamp.move_to([0.85, 1.98, 0])
        repeat = pill(f"same frame × {N_LAYERS}", color=GOLD, size=15)
        repeat.move_to([0.85, -1.72, 0])
        self.play(Create(frame), run_time=1.0)
        self.play(Write(stamp), run_time=0.8)
        self.play(Write(repeat), Indicate(frame, scale_factor=1.03, color=GOLD),
                  run_time=1.1)

        # s04.18 — a last RMSNorm cleans that stream before scoring
        self.cue("s04.18")
        self.play(FadeOut(frame), FadeOut(stamp), FadeOut(repeat), run_time=0.6)
        fin = pill("final RMSNorm", color=GOLD, size=15)
        fin.move_to([-0.85, -2.55, 0])
        fin_arrow = data_arrow(rail.get_bottom() + UP * 0.15 + RIGHT * 0.30,
                               fin.get_left(), GOLD, w=2.2)
        self.play(Create(fin_arrow), run_time=0.9)
        self.play(Write(fin), run_time=1.0)
        self.play(Circumscribe(rms_chip, color=GOLD, buff=0.12), run_time=1.0)
        self.play(Indicate(fin, scale_factor=1.08, color=GOLD), run_time=0.8)

        # s04.19 — a separate, untied table of the same kind runs at the output
        self.cue("s04.19")
        self.play(*[FadeOut(m) for m in
                    (mean_chip, mean2rms, root, rms_chip, att, att_lab, ffn,
                     ffn_lab, rms_a, rms_b, tap1, tap2, link1, link2, add1,
                     add2, plus1, plus2, x16, fin, fin_arrow)],
                  FadeIn(blocks), run_time=0.9)
        echo = grid(6, 5, cell=0.235, gap=0.035)
        echo.set_stroke(ROSE, opacity=0.22).set_fill(ROSE, 0.07)
        echo.move_to([2.95, -1.15, 0])
        echo_br = matrix_brackets(echo, ROSE, pad=0.06, tick=0.12)
        echo_lab = TY(f'W_U   {D_MODEL:,} → {VOCAB_LABEL}', size=20, color=ROSE)
        echo_lab.next_to(echo, DOWN, buff=0.24)
        echo_tag = DIN_T("A SEPARATE UNTIED TABLE", size=13, color=ROSE)
        echo_tag.next_to(echo, UP, buff=0.22)
        self.play(FadeIn(echo, scale=0.8), Create(echo_br), run_time=1.2)
        self.play(Write(echo_lab), Write(echo_tag), run_time=1.2)
        self.play(Indicate(echo, scale_factor=1.05, color=ROSE), run_time=0.7)
        self.play(flow(rail_lines[1], BLUE), run_time=0.7)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 5 ========
class Section05(Section):
    SEC = "sec05_matching"
    CHAPTER = "05 · ATTENTION: MATCHING"
    ACCENT = BLUE

    def construct(self):
        self._start_canvas_life()

        # s05.1 — one vector knows itself and nothing about its neighbours
        self.cue("s05.1")
        hdr = self.section_header("ATTENTION: MATCHING", "Queries, keys and scores",
                                  run_time=0.7, ty=2.78, size=32)
        lone = RoundedRectangle(corner_radius=0.08, width=1.5, height=0.72,
                                stroke_color=BLUE, stroke_width=2.0,
                                stroke_opacity=0.9, fill_color=BLUE,
                                fill_opacity=0.12)
        lone.move_to([0, 0.55, 0])
        lone_lab = T("x₂", size=24, color=BLUE, weight="BOLD").move_to(lone)
        lone_halo = glow_ring(lone, BLUE)
        self.play(FadeIn(lone, scale=0.8), Write(lone_lab), run_time=1.2)
        self.play(FadeIn(lone_halo), run_time=0.6)
        self.play(Indicate(lone, scale_factor=1.08, color=BLUE), run_time=0.8)

        # s05.2 — the next problem is context
        self.cue("s05.2")
        self.play(FadeOut(hdr), run_time=0.5)
        ghosts = VGroup()
        for dx, dy in [(-3.5, 1.45), (3.5, 1.45), (-3.5, -1.05), (3.5, -1.05),
                       (0, 2.05), (0, -1.70)]:
            g = RoundedRectangle(corner_radius=0.08, width=1.2, height=0.58,
                                 stroke_color=WHITE, stroke_opacity=0.20,
                                 stroke_width=1.2, fill_color=WHITE,
                                 fill_opacity=0.04)
            g.move_to([dx, 0.55 + dy, 0])
            ghosts.add(g)
        ctx_lab = T("context: the tokens around it", size=17, color=MUTED)
        ctx_lab.move_to([0, -2.70, 0])
        self.play(*[FadeIn(g) for g in ghosts], run_time=1.8)
        self.play(Write(ctx_lab), run_time=0.9)

        # s05.3 — every token looks back at the tokens before it
        self.cue("s05.3")
        self.play(FadeOut(lone), FadeOut(lone_lab), FadeOut(lone_halo),
                  FadeOut(ghosts), FadeOut(ctx_lab), run_time=0.7)
        words = ["The", "cat", "sat", "on"]
        toks = VGroup(*[T(w, size=21, color=WHITE) for w in words])
        toks.arrange(RIGHT, buff=1.15).move_to([0, 1.85, 0])
        q_ring = Circle(radius=0.62, stroke_color=BLUE, stroke_width=2.2,
                        stroke_opacity=0.9, fill_color=BLUE, fill_opacity=0.10)
        q_ring.move_to(toks[2])
        q_tag = DIN_T("QUERY", size=13, color=BLUE)
        q_tag.next_to(q_ring, UP, buff=0.18)
        arcs = VGroup(*[CurvedArrow(toks[2].get_bottom(), toks[j].get_bottom(),
                                    stroke_color=BLUE, stroke_width=1.8,
                                    tip_length=0.12, angle=-0.55,
                                    stroke_opacity=0.6, fill_opacity=0)
                        for j in range(2)])
        back_lab = T("looks back at earlier tokens", size=16, color=BLUE)
        back_lab.move_to([0, 0.55, 0])
        self.play(*[Write(t) for t in toks], run_time=1.5)
        self.play(Create(q_ring), Write(q_tag), run_time=1.0)
        self.play(*[Create(a) for a in arcs], run_time=1.2)
        self.play(Write(back_lab), run_time=0.9)

        # s05.4 — each vector is projected three times
        self.cue("s05.4")
        self.play(FadeOut(arcs), FadeOut(back_lab), run_time=0.5)
        labels = [("query", BLUE, "W_Q"), ("key", GOLD, "W_K"),
                  ("value", MINT, "W_V")]
        qkv = VGroup()
        for i, (name, col, wname) in enumerate(labels):
            bx = card(2.15, 1.05, edge=col)
            bx.move_to([-3.55 + i * 2.75, -0.55, 0])
            lab = T(name, size=19, color=lighter(col, 0.25))
            lab.move_to(bx.get_center() + UP * 0.13)
            wlab = DIN_T(wname, size=13, color=col)
            wlab.move_to(bx.get_center() + DOWN * 0.22)
            qkv.add(VGroup(bx, lab, wlab))
        arrows = VGroup(*[dim_arrow(toks[2].get_bottom() + DOWN * 0.55,
                                    qkv[i][0].get_top(), labels[i][1], 0.65,
                                    2.0, 0.13, 0.10) for i in range(3)])
        self.play(*[Create(a) for a in arrows], run_time=1.5)
        self.play(*[FadeIn(g) for g in qkv], run_time=1.9)
        self.play(*[Indicate(g, scale_factor=1.05, color=labels[i][1])
                    for i, g in enumerate(qkv)], run_time=1.1)

        # s05.5 — the query is what a token asks for
        self.cue("s05.5")
        qmark = T("?", size=28, color=BLUE, weight="BOLD")
        qmark.next_to(qkv[0][0], RIGHT, buff=0.28)
        self.play(FadeIn(qmark, scale=0.6), run_time=0.7)
        self.play(Indicate(qkv[0], scale_factor=1.12, color=BLUE), run_time=0.7)

        # s05.6 — the key is what a token offers for matching
        self.cue("s05.6")
        self.play(FadeOut(qmark), run_time=0.4)
        key_badges = VGroup(*[pill("k", color=GOLD, size=13)
                              for i, _ in enumerate(words) if i != 2])
        for b, t in zip(key_badges, [toks[0], toks[1], toks[3]]):
            b.move_to(t.get_top() + UP * 0.34)
        self.play(*[FadeIn(b, scale=0.6) for b in key_badges], run_time=0.7)
        self.play(*[Indicate(b, scale_factor=1.18, color=GOLD)
                    for b in key_badges], run_time=0.6)

        # s05.7 — the value is what a token hands over once it is chosen
        self.cue("s05.7")
        self.play(FadeOut(key_badges), run_time=0.5)
        v_badge = pill("v — the information handed over", color=MINT, size=15)
        v_badge.move_to([1.95, -1.65, 0])
        self.play(FadeIn(v_badge, shift=UP * 0.15), run_time=1.0)
        self.play(v_badge.animate.move_to([0, -1.75, 0]), run_time=1.0)
        self.play(Indicate(v_badge, scale_factor=1.06, color=MINT), run_time=0.7)

        # s05.8 — all three projections are learned matrices
        self.cue("s05.8")
        tiles = VGroup()
        for i, (_, col, _) in enumerate(labels):
            g = grid(4, 3, cell=0.16, gap=0.035)
            g.set_stroke(col, opacity=0.36).set_fill(col, 0.15)
            g.move_to(qkv[i][0].get_center() + DOWN * 1.12)
            g_br = matrix_brackets(g, col, pad=0.04, tick=0.08)
            tiles.add(VGroup(g, g_br))
        learn_lab = DIN_T("LEARNED MATRICES", size=TYPE["tag"], color=MUTED)
        learn_lab.move_to([0, -2.45, 0])
        self.play(*[FadeIn(g, scale=0.8) for g in tiles], run_time=1.6)
        self.play(*[Indicate(g, scale_factor=1.08, color=labels[i][1])
                    for i, g in enumerate(tiles)], run_time=1.5)
        self.play(Write(learn_lab), run_time=0.9)

        # s05.9 — a query takes a dot product with every key
        self.cue("s05.9")
        self.play(*[FadeOut(m) for m in
                    (qkv, arrows, tiles, learn_lab, v_badge)], run_time=0.6)
        self.play(VGroup(toks, q_ring, q_tag).animate.shift(UP * 0.55),
                  run_time=0.4)
        mat = grid(4, 4, cell=0.62, gap=0.10)
        mat.set_stroke(WHITE, opacity=0.18).set_fill(WHITE, 0.055)
        mat.move_to([0, -0.25, 0])
        mat_br = matrix_brackets(mat, BLUE, pad=0.08, tick=0.16)
        raw, _, _ = attention_demo()
        snums = VGroup()
        for r in range(4):
            for c in range(4):
                v = raw[r][c]
                cell_sq = mat[4 * r + c]
                if v > 0:
                    cell_sq.set_fill(BLUE, opacity=min(0.28, 0.04 + 0.22 * (v / 21.0)))
                else:
                    cell_sq.set_fill(WHITE, opacity=0.025)
                snums.add(Text(f"{v:+.1f}" if abs(v) < 100 else f"{v:+.0f}",
                               font=DIN, font_size=pt(13), color=MUTED,
                               weight="BOLD").move_to(cell_sq))
        matcap = DIN_T("SCORE MATRIX  ·  s = q · k", size=TYPE["tag"],
                       color=MUTED)
        matcap.next_to(mat, DOWN, buff=0.22)
        self.play(Create(mat), Create(mat_br), run_time=0.6)
        self.play(*[FadeIn(t) for t in snums], run_time=1.0)
        self.play(Write(matcap), run_time=0.7)

        # s05.10 — a large dot product means similar direction, long length, or both
        self.cue("s05.10")
        pair = VGroup(T("q₃", size=20, color=BLUE, weight="BOLD"),
                      T("·", size=20, color=DIM),
                      T("k₂", size=20, color=GOLD, weight="BOLD"))
        pair.arrange(RIGHT, buff=0.22).move_to([-4.05, 0.65, 0])
        orig_v = np.array([-5.25, -0.25, 0])
        tip_q = np.array([-4.15, 0.38, 0])
        tip_k = np.array([-3.95, -0.10, 0])
        a1 = Arrow(orig_v, tip_q, stroke_color=BLUE,
                   fill_color=BLUE, stroke_width=3.0, tip_length=0.16, buff=0)
        a2 = Arrow(orig_v, tip_k, stroke_color=GOLD,
                   fill_color=GOLD, stroke_width=3.0, tip_length=0.16, buff=0)
        uk = (tip_k - orig_v) / np.linalg.norm(tip_k - orig_v)
        proj_pt = orig_v + np.dot(tip_q - orig_v, uk) * uk
        proj_line = DashedVMobject(
            Line(tip_q, proj_pt, stroke_color=WHITE, stroke_opacity=0.65, stroke_width=1.4),
            num_dashes=7)
        ang_arc = ArcBetweenPoints(orig_v + 0.42 * uk,
                                   orig_v + 0.42 * (tip_q - orig_v) / np.linalg.norm(tip_q - orig_v),
                                   angle=0.45, stroke_color=MUTED, stroke_width=1.4)
        align_lab = T("aligned → large score", size=15, color=MUTED)
        align_lab.move_to([-4.35, -0.85, 0])
        cell = mat[4 * 2 + 1]
        self.play(Write(pair), run_time=0.8)
        self.play(Create(a1), Create(a2), Create(proj_line), Create(ang_arc), run_time=1.0)
        self.play(Write(align_lab), run_time=0.8)
        self.play(Indicate(cell, scale_factor=1.22, color=BLUE),
                  Indicate(snums[4 * 2 + 1], scale_factor=1.22, color=BLUE),
                  run_time=0.8)

        # s05.11 — each dot product fills one cell of the score matrix
        self.cue("s05.11")
        self.play(FadeOut(pair), FadeOut(a1), FadeOut(a2), FadeOut(proj_line),
                  FadeOut(ang_arc), FadeOut(align_lab),
                  run_time=0.5)
        last = mat[4 * 3 + 3]
        self.play(Indicate(last, scale_factor=1.22, color=GOLD),
                  Indicate(snums[4 * 3 + 3], scale_factor=1.22, color=GOLD),
                  run_time=0.9)
        self.play(Indicate(mat, scale_factor=1.03, color=MUTED), run_time=1.0)

        # s05.12 — the scores are divided by the square root of the key dimension
        self.cue("s05.12")
        scale_pill = pill(f"÷ √d_k     d_k = {ATTN['d_k']}     "
                          f"scale {ATTN['scale']}", color=GOLD,
                          size=17, stroke=GOLD)
        scale_pill.move_to([-4.0, 1.15, 0])
        self.play(FadeIn(scale_pill, shift=DOWN * 0.16), run_time=1.1)
        self.play(mat.animate.set_fill(WHITE, 0.035), run_time=0.8)

        # s05.13 — without scaling, scores grow with the dimension
        self.cue("s05.13")
        spike = VGroup()
        for i, h in enumerate([0.45, 0.95, 1.55, 2.35]):
            b = Rectangle(width=0.36, height=h, stroke_width=0,
                          fill_color=ROSE, fill_opacity=0.85)
            b.move_to([-4.15 + i * 0.52, -2.85 + h / 2, 0])
            spike.add(b)
        spike_lab = DIN_T("SCORES GROW", size=13, color=ROSE)
        spike_lab.move_to([-3.35, -3.25, 0])
        self.play(*[GrowFromEdge(b, DOWN) for b in spike], run_time=1.5)
        self.play(Write(spike_lab), run_time=0.7)
        self.play(Indicate(spike, scale_factor=1.05, color=ROSE), run_time=1.0)

        # s05.14 — scaling keeps the softmax away from saturation
        self.cue("s05.14")
        flat = VGroup()
        for i, h in enumerate([0.55, 0.50, 0.58, 0.52]):
            b = Rectangle(width=0.36, height=h, stroke_width=0,
                          fill_color=MINT, fill_opacity=0.85)
            b.move_to([2.65 + i * 0.52, -2.85 + h / 2, 0])
            flat.add(b)
        flat_lab = DIN_T("SCALED SCORES SETTLE", size=13, color=MINT)
        flat_lab.move_to([3.45, -3.25, 0])
        self.play(*[GrowFromEdge(b, DOWN) for b in flat], run_time=1.4)
        self.play(Write(flat_lab), run_time=0.7)
        self.play(Indicate(scale_pill, scale_factor=1.08, color=GOLD),
                  Indicate(flat, scale_factor=1.05, color=MINT), run_time=1.2)

        # s05.15 — the matrix is the map of who will listen to whom
        self.cue("s05.15")
        self.play(FadeOut(spike), FadeOut(spike_lab), FadeOut(flat),
                  FadeOut(flat_lab), FadeOut(toks), FadeOut(q_ring),
                  FadeOut(q_tag), FadeOut(scale_pill), run_time=0.6)
        mat_glow = glow_ring(mat, BLUE)
        map_lab = T("who will listen to whom", size=20, color=BLUE,
                    weight="MEDIUM")
        map_lab.move_to([0, -2.55, 0])
        banner = TY('"Attention"(Q, K, V) = "softmax"((Q K^T)/sqrt(d_k) + M) V',
                    size=24, color=WHITE)
        banner_card = card(11.4, 1.15, edge=BLUE)
        banner_card.move_to([0, 2.45, 0])
        banner.move_to(banner_card)
        self.play(FadeIn(mat_glow), run_time=0.4)
        self.play(Write(map_lab), run_time=0.5)
        self.play(Create(banner_card), FadeIn(banner), run_time=0.9)

        self.fade_content(0.40)
        self.sync_end()

class Section06(Section):
    SEC = "sec06_blend"
    CHAPTER = "06 · ATTENTION: BLENDING"
    ACCENT = GOLD

    def construct(self):
        self._start_canvas_life()

        # s06.1 — scores are raw numbers; they need to become weights
        self.cue("s06.1")
        hdr = self.section_header("ATTENTION: BLENDING", "Mask, softmax and heads",
                                  run_time=0.6, ty=2.78, size=32)
        mat = grid(4, 4, cell=0.58, gap=0.09)
        mat.set_stroke(WHITE, opacity=0.18).set_fill(WHITE, 0.055)
        mat.move_to([-3.05, 0.55, 0])
        mat_br = matrix_brackets(mat, GOLD, pad=0.08, tick=0.15)
        raw, _, _ = attention_demo()
        snums = VGroup()
        for r in range(4):
            for c in range(4):
                v = raw[r][c]
                cell_sq = mat[4 * r + c]
                if v > 0:
                    cell_sq.set_fill(BLUE, opacity=min(0.24, 0.03 + 0.18 * (v / 21.0)))
                snums.add(Text(f"{v:+.1f}" if abs(v) < 100 else f"{v:+.0f}",
                               font=DIN, font_size=pt(12), color=MUTED,
                               weight="BOLD").move_to(cell_sq))
        matcap = DIN_T("SCORES", size=TYPE["tag"], color=MUTED)
        matcap.next_to(mat, DOWN, buff=0.22)
        wlabels, wprobs = attention_weight_bars()
        wcols = [BLUE, GOLD, MINT, VIOLET]
        wbars = VGroup()
        for i, (lab, col) in enumerate(zip(wlabels, wcols)):
            y = -i * 0.46
            lab_t = T(lab, size=16, color=lighter(col, 0.35), weight="MEDIUM")
            lab_t.move_to([0, y, 0], aligned_edge=LEFT)
            track = Rectangle(width=2.6, height=0.30, stroke_width=0,
                              fill_color=WHITE, fill_opacity=0.06)
            track.next_to(lab_t, RIGHT, buff=0.18)
            track.set_y(lab_t.get_y())
            wbars.add(VGroup(lab_t, track))
        wbars.move_to([1.35, 0.65, 0])
        wcap = TY('w_(3 j) = "softmax"(s_(3 j) + M)', size=17, color=MUTED)
        wcap.next_to(wbars, DOWN, buff=0.24)
        self.play(Create(mat), Create(mat_br), run_time=0.7)
        self.play(*[FadeIn(t) for t in snums], Write(matcap), run_time=0.9)
        self.play(FadeIn(wbars), run_time=0.7)

        # s06.2 — a causal mask hides the future from every query
        self.cue("s06.2")
        self.play(FadeOut(hdr), run_time=0.4)
        hatch = VGroup()
        for r in range(4):
            for c in range(4):
                if c > r:
                    cell = mat[4 * r + c].copy()
                    cell.set_fill(ROSE, 0.22).set_stroke(ROSE, 1.3, 0.75)
                    hatch.add(cell)
        fut = DIN_T("FUTURE HIDDEN", size=13, color=ROSE, tracked=False)
        fut.move_to([-3.05, -1.45, 0])
        self.play(*[FadeIn(h) for h in hatch], run_time=1.1)
        self.play(Write(fut), run_time=0.8)

        # s06.3 — a token may attend to itself and to earlier tokens only
        self.cue("s06.3")
        self.play(*[h.animate.set_fill(ROSE, 0.06).set_stroke(ROSE, 1.1, 0.30)
                    for h in hatch], run_time=1.1)
        live = VGroup()
        for r in range(4):
            for c in range(4):
                if c <= r:
                    cell = mat[4 * r + c].copy()
                    cell.set_fill(GOLD, 0.16).set_stroke(GOLD, 1.3, 0.65)
                    live.add(cell)
        live_lab = T("self  +  earlier tokens", size=16, color=GOLD)
        live_lab.move_to([-3.05, -2.15, 0])
        self.play(*[FadeIn(c) for c in live], run_time=1.4)
        self.play(Write(live_lab), run_time=0.8)
        self.play(Indicate(live, scale_factor=1.03, color=GOLD), run_time=0.9)

        # s06.4 — training never lets a token read the answer it must predict
        self.cue("s06.4")
        ans = pill(TOKENS["items"][3]["text"], color=ROSE, size=17, stroke=ROSE)
        ans.move_to([3.95, 1.95, 0])
        ans_tag = DIN_T("THE ANSWER", size=12, color=ROSE, tracked=False)
        ans_tag.next_to(ans, DOWN, buff=0.12)
        self.play(FadeIn(ans, scale=0.8), Write(ans_tag), run_time=1.1)
        self.play(Indicate(ans, scale_factor=1.10, color=ROSE), run_time=0.7)
        self.play(FadeOut(ans), FadeOut(ans_tag), run_time=0.8)
        self.play(*[Indicate(h, scale_factor=1.10, color=ROSE)
                    for h in hatch[::3]], run_time=0.8)

        # s06.5 — masked scores go to minus infinity before the softmax
        self.cue("s06.5")
        inf_lab = T("−∞", size=22, color=ROSE, weight="BOLD")
        inf_lab.move_to([-3.05, -1.45, 0])
        self.play(ReplacementTransform(fut, inf_lab), run_time=0.8)
        self.play(*[snums[4 * r + c].animate.set_color(ROSE)
                    for r in range(4) for c in range(4) if c > r],
                  run_time=1.2)
        self.play(Indicate(inf_lab, scale_factor=1.15, color=ROSE),
                  run_time=0.7)

        # s06.6 — softmax turns the rest into weights that sum to one
        self.cue("s06.6")
        real = bars(wlabels, wprobs, wcols, width=2.6, size=16)
        real.move_to(wbars.get_center())
        sum_lab = T("weights sum to 1", size=16, color=MINT)
        sum_lab.move_to([1.35, -1.65, 0])
        self.play(ReplacementTransform(wbars, real), run_time=1.5)
        self.play(Write(wcap), run_time=0.8)
        self.play(Write(sum_lab), run_time=0.8)
        self.play(Indicate(real, scale_factor=1.04, color=MINT), run_time=0.8)

        # s06.7 — look at the row for the word sat
        self.cue("s06.7")
        row_hl = RoundedRectangle(corner_radius=0.05,
                                  width=mat.width + 0.14, height=0.72,
                                  stroke_color=GOLD, stroke_width=2.0,
                                  stroke_opacity=0.95, fill_color=GOLD,
                                  fill_opacity=0.14)
        row_hl.move_to(mat[8:12].get_center())
        self.play(GrowFromCenter(row_hl), run_time=0.8)
        self.play(Indicate(real, scale_factor=1.05, color=GOLD), run_time=0.7)

        # s06.8 — it puts 0.86 of its weight on cat
        self.cue("s06.8")
        big = real[1][2]
        self.play(big.animate.stretch_to_fit_height(0.38),
                  run_time=1.1)
        self.play(Indicate(real[1], scale_factor=1.10, color=GOLD),
                  run_time=0.8)
        self.play(Indicate(real[1][3], scale_factor=1.35, color=GOLD),
                  run_time=0.8)

        # s06.9 — the masked future cell holds zero
        self.cue("s06.9")
        zero_note = T("masked key → exactly 0.00", size=15, color=ROSE)
        zero_note.move_to([1.35, -2.35, 0])
        self.play(Write(zero_note), run_time=0.8)
        self.play(Indicate(real[3], scale_factor=1.10, color=ROSE),
                  run_time=0.7)

        # s06.10 — the weights blend the value vectors into one output
        self.cue("s06.10")
        self.play(FadeOut(row_hl), FadeOut(zero_note), FadeOut(sum_lab),
                  FadeOut(wcap), run_time=0.5)
        obox = Arrow([2.55, -2.55, 0], [3.75, -2.55, 0], stroke_color=MINT,
                     fill_color=MINT, stroke_width=3.0, tip_length=0.18,
                     buff=0)
        otxt = T("o₃", size=24, color=MINT, weight="BOLD")
        otxt.next_to(obox, RIGHT, buff=0.16)
        olab = TY('o_i = sum_j w_(i j) v_j', size=18, color=MINT)
        olab.move_to([0.25, -2.55, 0])
        wsum = VGroup()
        for i, col in enumerate(wcols):
            wsum.add(dim_arrow([0.45 + i * 0.52, -0.55, 0],
                               obox.get_start() + UP * 0.20, col, 0.55, 1.8,
                               0.12, 0.10))
        self.play(*[Create(a) for a in wsum], run_time=0.9)
        self.play(Create(obox), Write(otxt), run_time=0.7)
        self.play(Write(olab), run_time=0.7)

        # s06.11 — the output stays near the neighbourhood of past values
        self.cue("s06.11")
        cloud = VGroup()
        for i, (dx, dy) in enumerate([(-0.85, 0.35), (-0.35, -0.30),
                                      (0.15, 0.45), (0.70, -0.15)]):
            cloud.add(Dot(point=[4.55 + dx, -1.35 + dy, 0], radius=0.09,
                          color=wcols[i]))
        o_pt = glow_dot([4.55, -1.30, 0], MINT, r=0.12)
        blend_spokes = VGroup(*[
            DashedVMobject(
                Line(cloud[i].get_center(), [4.55, -1.30, 0], stroke_color=wcols[i],
                     stroke_width=1.6, stroke_opacity=0.22 + 0.65 * wprobs[i]),
                num_dashes=6)
            for i in range(3)])
        neigh = T("weighted average of past values", size=15, color=MUTED)
        neigh.move_to([4.55, -2.35, 0])
        self.play(*[FadeIn(d) for d in cloud], run_time=1.1)
        self.play(*[Create(s) for s in blend_spokes], FadeIn(o_pt), run_time=0.8)
        self.play(Write(neigh), run_time=0.9)
        self.play(Indicate(o_pt, scale_factor=1.20, color=MINT), run_time=0.8)

        # s06.12 — each token leaves with a summary of its own past
        self.cue("s06.12")
        self.play(FadeOut(wsum), FadeOut(olab), FadeOut(neigh), run_time=0.5)
        summary = pill("summary of its own past", color=MINT, size=16)
        summary.move_to([3.15, -2.95, 0])
        self.play(Write(summary), run_time=1.0)
        self.play(Indicate(obox, scale_factor=1.12, color=MINT),
                  Indicate(otxt, scale_factor=1.12, color=MINT), run_time=1.0)

        # s06.13 — this whole computation runs once per head
        self.cue("s06.13")
        self.play(FadeOut(cloud), FadeOut(blend_spokes), FadeOut(o_pt), FadeOut(summary),
                  FadeOut(obox), FadeOut(otxt), FadeOut(live_lab),
                  FadeOut(inf_lab), run_time=0.6)
        hcols = [BLUE, GOLD, MINT, VIOLET] * (N_HEADS // 4)
        heads = VGroup(*[head_badge(f"h{i + 1}", hcols[i], r=0.18)
                         for i in range(N_HEADS)])
        heads.arrange(RIGHT, buff=0.34).move_to([0, -2.35, 0])
        hlab = pill(f"× {N_HEADS} heads", color=WHITE, size=13)
        hlab.move_to([4.35, -2.35, 0])
        self.play(*[FadeIn(h, shift=UP * 0.14) for h in heads], run_time=1.4)
        self.play(Write(hlab), run_time=0.7)
        self.play(Indicate(heads, scale_factor=1.06, color=WHITE), run_time=1.0)

        # s06.14 — Q is per head; K and V are shared by pairs of heads
        self.cue("s06.14")
        pair_lab = DIN_T("K, V SHARED BY PAIRS OF HEADS", size=TYPE["micro"],
                         color=MUTED)
        pair_lab.move_to([-3.85, -3.15, 0])
        pair_links = VGroup()
        for i in range(0, N_HEADS, 2):
            pair_links.add(Line(heads[i].get_bottom(), heads[i + 1].get_bottom(),
                                stroke_color=GOLD, stroke_width=2.2,
                                stroke_opacity=0.8))
        self.play(Write(pair_lab), run_time=1.0)
        self.play(*[Create(l) for l in pair_links], run_time=1.2)
        self.play(*[Indicate(heads[i], scale_factor=1.14, color=hcols[i])
                    for i in range(N_HEADS)], run_time=1.4)

        # s06.15 — eight outputs, concatenated, times one output matrix
        self.cue("s06.15")
        hcap = T("independent projections → concat\n→ W_O", size=13,
                 color=MUTED)
        hcap.next_to(heads, DOWN, buff=0.55)
        w_o = pill(f"W_O   {D_MODEL:,} → {D_MODEL:,}", color=ROSE, size=14,
                   stroke=ROSE)
        w_o.move_to([4.45, -1.25, 0])
        self.play(Write(hcap), run_time=1.1)
        self.play(Write(w_o), run_time=1.0)
        self.play(Indicate(w_o, scale_factor=1.08, color=ROSE), run_time=0.8)

        # s06.16 — the result is written back into the residual stream
        self.cue("s06.16")
        self.play(FadeOut(mat), FadeOut(mat_br), FadeOut(snums), FadeOut(matcap),
                  FadeOut(live), FadeOut(hatch), FadeOut(real), run_time=0.5)
        band = card(9.6, 0.95, fill_op=0.045, radius=0.2)
        band.move_to([0, 1.72, 0])
        band_lab = DIN_T("THE RESIDUAL STREAM", size=TYPE["micro"], color=MUTED)
        band_lab.next_to(band, DOWN, buff=0.16).align_to(band, LEFT).shift(
            RIGHT * 0.15)
        p_center = np.array([0, 1.05, 0])
        plus = VGroup(Circle(radius=0.20, stroke_color=MINT, stroke_width=1.8,
                             fill_color=BLACK,
                             fill_opacity=1.0).move_to(p_center),
                      T("+", size=20, color=MINT,
                        weight="BOLD").move_to(p_center))
        write_back = data_arrow([0, -1.55, 0], plus[0].get_bottom(), MINT, w=3.0)
        self.play(Create(band), Write(band_lab), run_time=1.0)
        self.play(Create(write_back), GrowFromCenter(plus), run_time=0.9)

        self.fade_content(0.40)
        self.sync_end()

class Section07(Section):
    SEC = "sec07_rope"
    CHAPTER = "07 · ROTARY POSITIONS"
    ACCENT = VIOLET

    def construct(self):
        self._start_canvas_life()

        # s07.1 — attention as written cannot see word order
        self.cue("s07.1")
        hdr = self.section_header("POSITION, ROTATING", "Rotary positions — RoPE",
                                  run_time=0.8, ty=2.78, size=32)
        row1 = VGroup(T("the", size=22, color=BLUE), T("cat", size=22,
                                                       color=GOLD),
                      T("sat", size=22, color=MINT))
        row1.arrange(RIGHT, buff=0.45).move_to([0, 1.35, 0])
        b1 = VGroup(T("[", size=26, color=MUTED).next_to(row1, LEFT, buff=0.15),
                    T("]", size=26, color=MUTED).next_to(row1, RIGHT,
                                                         buff=0.15))
        row2 = VGroup(T("sat", size=22, color=MINT), T("cat", size=22,
                                                       color=GOLD),
                      T("the", size=22, color=BLUE))
        row2.arrange(RIGHT, buff=0.45).move_to([0, 0.15, 0])
        b2 = VGroup(T("[", size=26, color=MUTED).next_to(row2, LEFT, buff=0.15),
                    T("]", size=26, color=MUTED).next_to(row2, RIGHT,
                                                         buff=0.15))
        eqt = TY('q_i dot k_j = "const"   ("order-blind")', size=18,
                 color=MUTED)
        eqt.move_to([0, -1.25, 0])
        self.play(Write(row1), FadeIn(b1), run_time=1.1)
        self.play(Write(row2), FadeIn(b2), run_time=1.1)
        self.play(Write(eqt), run_time=0.9)
        self.play(Indicate(row1, scale_factor=1.06, color=WHITE),
                  Indicate(row2, scale_factor=1.06, color=WHITE),
                  run_time=0.9)

        # s07.2 — the vectors carry meaning without position
        self.cue("s07.2")
        self.play(FadeOut(hdr), run_time=0.5)
        meaning = VGroup(*[pill(w, color=c, size=14)
                           for w, c in [("meaning", BLUE), ("meaning", GOLD),
                                        ("meaning", MINT)]])
        meaning.arrange(RIGHT, buff=0.45).move_to([0, -2.05, 0])
        slot = RoundedRectangle(corner_radius=0.08, width=1.35, height=0.62,
                                stroke_color=DIM, stroke_opacity=0.6,
                                stroke_width=1.6, fill_color=BLACK,
                                fill_opacity=1.0)
        slot.move_to([4.05, -2.05, 0])
        slot_lab = DIN_T("POSITION", size=12, color=DIM, tracked=False)
        slot_lab.next_to(slot, DOWN, buff=0.12)
        self.play(*[FadeIn(m) for m in meaning], run_time=1.2)
        self.play(FadeIn(slot), Write(slot_lab), run_time=1.0)
        self.play(Indicate(meaning, scale_factor=1.05, color=BLUE),
                  run_time=1.0)
        self.play(Indicate(slot, scale_factor=1.10, color=DIM), run_time=0.9)

        # s07.3 — the next piece adds position without touching the meaning
        self.cue("s07.3")
        tile = pill("m · θ", color=VIOLET, size=17, stroke=VIOLET)
        tile.move_to([2.85, -0.95, 0])
        self.play(FadeIn(tile), run_time=0.8)
        self.play(tile.animate.move_to(slot.get_center()), run_time=1.0)
        self.play(Indicate(tile, scale_factor=1.12, color=VIOLET), run_time=0.6)

        # s07.4 — rotary embeddings rotate q and k by position
        self.cue("s07.4")
        self.play(*[FadeOut(m) for m in
                    (hdr, row1, b1, row2, b2, eqt, meaning, slot, slot_lab,
                     tile)], run_time=0.8)
        cc = np.array([-3.35, -0.35, 0])
        circ = Circle(radius=1.15, stroke_color=WHITE, stroke_opacity=0.35,
                      stroke_width=1.6).move_to(cc)
        inner_circ = DashedVMobject(
            Circle(radius=0.58, stroke_color=WHITE, stroke_opacity=0.14,
                   stroke_width=1.0).move_to(cc), num_dashes=20)
        c_axes = VGroup(
            Line(cc + LEFT * 1.25, cc + RIGHT * 1.25, stroke_color=WHITE,
                 stroke_opacity=0.18, stroke_width=1.2),
            Line(cc + DOWN * 1.25, cc + UP * 1.25, stroke_color=WHITE,
                 stroke_opacity=0.18, stroke_width=1.2),
            inner_circ)
        for deg in range(0, 360, 30):
            u = np.array([np.cos(deg * TAU / 360), np.sin(deg * TAU / 360), 0])
            c_axes.add(Line(cc + u * 1.07, cc + u * 1.15, stroke_color=WHITE,
                            stroke_opacity=0.26, stroke_width=1.0))
        vec = Arrow(cc, cc + RIGHT * 1.15, stroke_color=VIOLET,
                    fill_color=VIOLET, stroke_width=3.0, tip_length=0.18,
                    buff=0)
        vec_halo = glow_ring(vec, VIOLET)
        mtrack = ValueTracker(0)
        mlab = always_redraw(
            lambda: T(f"m = {int(mtrack.get_value())}", size=18, color=VIOLET,
                      weight="MEDIUM").next_to(circ, DOWN, buff=0.22))
        self.play(Create(circ), Create(c_axes), run_time=1.0)
        self.play(GrowFromEdge(vec, LEFT), FadeIn(vec_halo), run_time=1.0)
        self.play(FadeIn(mlab), run_time=0.7)
        self.play(Rotate(vec, angle=-TAU / 5, about_point=cc), run_time=1.4)

        # s07.5 — each vector splits into pairs of dimensions
        self.cue("s07.5")
        pairs_n = FACTS["rope"]["pairs"]
        pair_lab = T(f"{pairs_n} pairs per head", size=20, color=VIOLET,
                     weight="MEDIUM")
        pair_lab.move_to([1.65, 1.55, 0])
        split = VGroup()
        for i in range(8):
            split.add(Rectangle(width=0.16, height=0.34, stroke_width=0,
                                fill_color=VIOLET, fill_opacity=0.75))
        split.arrange(RIGHT, buff=0.12).move_to([1.65, 0.55, 0])
        pair_mark = VGroup(
            Line(split[2].get_left() + UP * 0.30, split[3].get_right() + UP * 0.30,
                 stroke_color=WHITE, stroke_opacity=0.55, stroke_width=1.6),
            T("( x₂ₖ , x₂ₖ₊₁ )", size=15, color=MUTED).next_to(split, DOWN,
                                                                buff=0.30))
        self.play(Write(pair_lab), run_time=1.0)
        self.play(FadeIn(split), run_time=1.1)
        self.play(Create(pair_mark[0]), Write(pair_mark[1]), run_time=1.0)

        # s07.6 — each pair is a 2-D point, or a complex number
        self.cue("s07.6")
        zx = Line([3.35, -1.05, 0], [5.95, -1.05, 0], stroke_color=WHITE,
                  stroke_opacity=0.22, stroke_width=1.3)
        zy = Line([4.65, -2.35, 0], [4.65, 0.25, 0], stroke_color=WHITE,
                  stroke_opacity=0.22, stroke_width=1.3)
        zpt = glow_dot([5.15, -0.55, 0], VIOLET, r=0.11)
        zlab = TY('z = x_(2k) + i x_(2k+1)', size=19, color=VIOLET)
        zlab.move_to([4.65, -2.75, 0])
        re_lab = T("real", size=13, color=MUTED).next_to(zx, RIGHT, buff=0.12)
        im_lab = T("imag", size=13, color=MUTED).next_to(zy, UP, buff=0.10)
        self.play(Create(zx), Create(zy), run_time=0.9)
        self.play(FadeIn(zpt), Write(re_lab), Write(im_lab), run_time=1.1)
        self.play(Write(zlab), run_time=1.0)
        self.play(Indicate(zpt, scale_factor=1.25, color=VIOLET), run_time=0.8)

        # s07.7 — position m rotates every pair by m times a base frequency
        self.cue("s07.7")
        self.play(FadeOut(pair_mark), run_time=0.5)
        rot = TY('R(m theta) = mat(cos m theta, - sin m theta;'
                 ' sin m theta, cos m theta)', size=24, color=WHITE)
        rot.move_to([1.55, -1.55, 0])
        rot2 = TY('e^(i m theta)', size=24, color=VIOLET)
        rot2.move_to([1.55, -2.45, 0])
        self.play(Write(rot), run_time=1.5)
        self.play(Write(rot2), run_time=1.1)
        self.play(mtrack.animate(rate_func=linear).set_value(DEMO_LAST_M),
                  Rotate(vec, angle=-TAU / 5 * DEMO_LAST_M, about_point=cc),
                  run_time=1.6)

        # s07.8 — the rotation matrix is the pair version of that turn
        self.cue("s07.8")
        self.play(Indicate(rot, scale_factor=1.05, color=WHITE), run_time=0.9)
        self.play(Indicate(vec, scale_factor=1.12, color=VIOLET),
                  Indicate(zpt, scale_factor=1.12, color=VIOLET), run_time=1.0)

        # s07.9 — different pairs turn at different speeds
        self.cue("s07.9")
        self.play(FadeOut(rot), FadeOut(rot2), FadeOut(split),
                  FadeOut(pair_lab), FadeOut(zx), FadeOut(zy), FadeOut(zpt),
                  FadeOut(zlab), FadeOut(re_lab), FadeOut(im_lab),
                  run_time=0.6)
        rings = VGroup()
        for x in (0.55, 2.85, 5.15):
            rings.add(Circle(radius=0.95, stroke_color=WHITE,
                             stroke_opacity=0.30, stroke_width=1.5)
                      .move_to([x, -0.55, 0]))
        self.play(*[GrowFromCenter(r) for r in rings], run_time=1.0)

        # s07.10 — fast pairs pin nearby structure, slow pairs mark distance
        self.cue("s07.10")
        samples = FACTS["rope"]["samples"]
        last_pair = FACTS["rope"]["pairs"] - 1
        hands, plabs = VGroup(), VGroup()
        for r, sample in zip(rings, samples):
            hands.add(Arrow(r.get_center(),
                            r.get_center() + RIGHT * 0.85,
                            stroke_color=VIOLET, fill_color=VIOLET,
                            stroke_width=2.6, tip_length=0.15, buff=0))
            plabs.add(T(f"pair {sample['pair']}   "
                        f"{period_label(sample['period_tokens'])} tokens",
                        size=13, color=MUTED).next_to(r, DOWN, buff=0.20))
        band_lab = DIN_T("FREQUENCY LADDER", size=TYPE["tag"], color=VIOLET)
        band_lab.move_to([2.85, 1.15, 0])
        self.play(Write(band_lab), run_time=0.7)
        self.play(*[GrowFromEdge(h, LEFT) for h in hands], run_time=1.2)
        self.play(*[Write(p) for p in plabs], run_time=1.2)
        self.play(Rotate(hands[0], angle=-TAU, about_point=rings[0].get_center()),
                  run_time=1.0)

        # s07.11 — the fastest pair shown completes a turn in about 6.3 tokens
        self.cue("s07.11")
        fast_note = T(f"pair 0 turns once every "
                      f"{period_label(samples[0]['period_tokens'])} tokens",
                      size=17, color=VIOLET)
        fast_note.move_to([2.85, 1.85, 0])
        self.play(Write(fast_note), run_time=1.0)
        self.play(Rotate(hands[0], angle=-TAU, about_point=rings[0].get_center()),
                  Rotate(hands[1], angle=-TAU / 4,
                         about_point=rings[1].get_center()),
                  Rotate(hands[2], angle=-TAU / 60,
                         about_point=rings[2].get_center()),
                  run_time=1.6)
        self.play(Indicate(hands[0], scale_factor=1.12, color=VIOLET),
                  run_time=0.8)

        # s07.12 — a rotated query meets a rotated key: one relative turn
        self.cue("s07.12")
        self.play(FadeOut(band_lab), FadeOut(fast_note), run_time=0.5)
        qa = Arrow([-3.35, -2.95, 0], [-4.45, -2.25, 0], stroke_color=BLUE,
                   fill_color=BLUE, stroke_width=3.0, tip_length=0.16, buff=0)
        ka = Arrow([-3.35, -2.95, 0], [-2.25, -2.25, 0], stroke_color=GOLD,
                   fill_color=GOLD, stroke_width=3.0, tip_length=0.16, buff=0)
        rel_arc = ArcBetweenPoints([-2.85, -2.63, 0], [-3.85, -2.63, 0],
                                   angle=0.85, stroke_color=VIOLET, stroke_width=2.0)
        q_lab = T("R_m q", size=17, color=BLUE).move_to([-4.95, -2.55, 0])
        k_lab = T("R_n k", size=17, color=GOLD).move_to([-1.75, -2.55, 0])
        self.play(GrowFromEdge(qa, RIGHT), GrowFromEdge(ka, LEFT),
                  Create(rel_arc), Write(q_lab), Write(k_lab), run_time=1.4)
        self.play(Rotate(qa, angle=TAU / 18, about_point=qa.get_start()),
                  Rotate(ka, angle=-TAU / 18, about_point=ka.get_start()),
                  run_time=1.2)
        combine = TY('(R_m q)^T (R_n k) = g(m - n)', size=26, color=WHITE)
        combine.move_to([2.85, -3.05, 0])
        self.play(Write(combine), run_time=1.4)
        self.play(Indicate(qa, scale_factor=1.10, color=BLUE),
                  Indicate(ka, scale_factor=1.10, color=GOLD), run_time=1.0)

        # s07.13 — the positional part depends only on the distance m − n
        self.cue("s07.13")
        pcap = T("depends only on the distance  m − n", size=17, color=VIOLET)
        pcap.move_to([-1.65, 1.85, 0])
        self.play(Write(pcap), run_time=1.1)
        self.play(Indicate(combine, scale_factor=1.05, color=VIOLET),
                  run_time=0.9)

        # s07.14 — content still shapes the score
        self.cue("s07.14")
        self.play(FadeOut(qa), FadeOut(ka), FadeOut(rel_arc), FadeOut(q_lab), FadeOut(k_lab),
                  FadeOut(combine), run_time=0.6)
        content = pill("content  ·  direction and length", color=MINT, size=16)
        content.move_to([-3.35, -2.35, 0])
        dist = pill("position  ·  how far apart", color=VIOLET, size=16,
                    stroke=VIOLET)
        dist.move_to([1.35, -2.35, 0])
        self.play(FadeIn(content), run_time=1.0)
        self.play(FadeIn(dist), run_time=1.0)
        self.play(Indicate(content, scale_factor=1.06, color=MINT),
                  run_time=0.8)
        self.play(Indicate(dist, scale_factor=1.06, color=VIOLET), run_time=0.8)

        # s07.15 — theta keeps the slowest pairs smooth across the context
        self.cue("s07.15")
        theta_chip = pill(f"θ = {THETA_LABEL}", color=GOLD, size=18,
                          stroke=GOLD)
        theta_chip.move_to([2.85, 1.85, 0])
        ctx_bar = Rectangle(width=6.2, height=0.30, stroke_color=WHITE,
                            stroke_opacity=0.22, stroke_width=1.2,
                            fill_color=WHITE, fill_opacity=0.07)
        ctx_bar.move_to([2.85, -3.05, 0])
        ctx_lab = DIN_T(f"CONTEXT {SEQ_LEN:,}", size=TYPE["micro"], color=MUTED)
        ctx_lab.next_to(ctx_bar, UP, buff=0.14)
        slow_arc = Arc(radius=0.95, angle=TAU / 3, stroke_color=VIOLET,
                       stroke_width=4.0).move_to(rings[2].get_center())
        self.play(Write(theta_chip), run_time=1.0)
        self.play(Create(ctx_bar), Write(ctx_lab), run_time=1.2)
        self.play(Create(slow_arc), run_time=1.3)
        self.play(Indicate(hands[2], scale_factor=1.12, color=VIOLET),
                  run_time=0.9)

        # s07.16 — theta sets the slowest wavelength
        self.cue("s07.16")
        self.play(FadeOut(content), FadeOut(dist), FadeOut(pcap),
                  FadeOut(ctx_bar), FadeOut(ctx_lab), FadeOut(circ),
                  FadeOut(c_axes), FadeOut(vec), FadeOut(vec_halo),
                  FadeOut(mlab), run_time=0.5)
        dial = Circle(radius=0.85, stroke_color=MUTED, stroke_opacity=0.45,
                      stroke_width=2.2, fill_color=WHITE,
                      fill_opacity=0.04).move_to([-3.35, -0.35, 0])
        dticks = dial_ticks(dial, MUTED)
        needle = Line(dial.get_center(),
                      dial.get_center() + RIGHT * 0.55 + UP * 0.55,
                      stroke_width=3.2, stroke_color=GOLD)
        dial_lab = DIN_T("THETA", size=TYPE["tag"], color=MUTED)
        dial_lab.next_to(dial, DOWN, buff=0.20)
        longer = T("larger θ → longer slowest wavelength", size=17, color=GOLD)
        longer.move_to([2.85, 2.45, 0])
        self.play(FadeIn(dial), FadeIn(dticks), Write(dial_lab), run_time=0.9)
        self.play(GrowFromEdge(needle, LEFT), run_time=0.6)
        self.play(Write(longer), run_time=0.9)
        self.play(Rotate(needle, angle=TAU / 5, about_point=dial.get_center()),
                  Transform(slow_arc,
                            Arc(radius=0.95, angle=TAU / 2,
                                stroke_color=GOLD,
                                stroke_width=4.0).move_to(
                                    rings[2].get_center())),
                  run_time=1.2)

        self.fade_content(0.40)
        self.sync_end()

class Section08(Section):
    SEC = "sec08_qknorm"
    CHAPTER = "08 · QK-NORM"
    ACCENT = MINT

    def construct(self):
        self._start_canvas_life()

        # s08.1 — one Lite detail lands between projection and rotation
        self.cue("s08.1")
        hdr = self.section_header("A LITE DETAIL", "QK-norm", run_time=0.7,
                                  ty=2.78, size=34)
        q_in = pill("q", color=BLUE, size=18)
        q_in.move_to([-5.15, 0.85, 0])
        k_in = pill("k", color=GOLD, size=18)
        k_in.move_to([-5.15, -0.85, 0])
        rope_q = pill("RoPE", color=VIOLET, size=16, stroke=VIOLET)
        rope_q.move_to([4.35, 0.85, 0])
        rope_k = pill("RoPE", color=VIOLET, size=16, stroke=VIOLET)
        rope_k.move_to([4.35, -0.85, 0])
        self.play(FadeIn(q_in), FadeIn(k_in), run_time=0.8)
        self.play(FadeIn(rope_q), FadeIn(rope_k), run_time=0.8)
        self.play(Indicate(q_in, scale_factor=1.10, color=BLUE),
                  Indicate(k_in, scale_factor=1.10, color=GOLD), run_time=0.7)

        # s08.2 — q and k each pass through RMSNorm before RoPE
        self.cue("s08.2")
        self.play(FadeOut(hdr), run_time=0.5)
        nq = pill("RMSNorm", color=MINT, size=16, stroke=MINT)
        nq.move_to([-0.65, 0.85, 0])
        nk = pill("RMSNorm", color=MINT, size=16, stroke=MINT)
        nk.move_to([-0.65, -0.85, 0])
        a1 = dim_arrow(q_in.get_right(), nq.get_left(), BLUE, 0.7, 2.2, 0.14,
                       0.12)
        a2 = dim_arrow(nq.get_right(), rope_q.get_left(), VIOLET, 0.7, 2.2,
                       0.14, 0.12)
        a3 = dim_arrow(k_in.get_right(), nk.get_left(), GOLD, 0.7, 2.2, 0.14,
                       0.12)
        a4 = dim_arrow(nk.get_right(), rope_k.get_left(), VIOLET, 0.7, 2.2,
                       0.14, 0.12)
        self.play(Create(a1), Create(a3), run_time=0.9)
        self.play(FadeIn(nq), FadeIn(nk), run_time=0.9)
        self.play(Create(a2), Create(a4), run_time=0.9)
        self.play(Indicate(nq, scale_factor=1.08, color=MINT),
                  Indicate(nk, scale_factor=1.08, color=MINT), run_time=0.8)

        # s08.3 — the step controls scale, so the logits stay well behaved
        self.cue("s08.3")
        bars_g = VGroup()
        for i, h in enumerate([1.10, 0.75, 1.20, 0.65, 1.00]):
            b = Rectangle(width=0.42, height=h, stroke_width=0, fill_color=MINT,
                          fill_opacity=0.80)
            b.move_to([-2.85 + i * 0.62, -3.20 + h / 2, 0])
            bars_g.add(b)
        b_grid = VGroup(*[
            Line([-3.35, y, 0], [0.35, y, 0], stroke_color=WHITE,
                 stroke_opacity=0.10, stroke_width=0.9)
            for y in (-3.20, -2.80, -2.40)])
        bound = DashedVMobject(
            Line([-3.35, -1.95, 0], [0.35, -1.95, 0], stroke_color=WHITE,
                 stroke_width=1.8), num_dashes=18)
        bound.set_stroke(opacity=0.55)
        bound_lab = DIN_T("BOUNDED", size=12, color=MUTED, tracked=False)
        bound_lab.move_to([0.95, -1.95, 0])
        self.play(FadeIn(b_grid), *[GrowFromEdge(b, DOWN) for b in bars_g], run_time=1.3)
        self.play(Create(bound), Write(bound_lab), run_time=1.0)
        self.play(Indicate(bars_g, scale_factor=1.05, color=MINT), run_time=1.0)

        # s08.4 — the norm runs per head, on each q and k vector on its own
        self.cue("s08.4")
        self.play(FadeOut(b_grid), FadeOut(bars_g), FadeOut(bound), FadeOut(bound_lab),
                  run_time=0.5)
        hcols = [BLUE, GOLD, MINT, VIOLET] * (N_HEADS // 4)
        heads = VGroup(*[head_badge(f"h{i + 1}", hcols[i], r=0.17)
                         for i in range(N_HEADS)])
        heads.arrange(RIGHT, buff=0.68).move_to([0, -2.55, 0])
        strip_y = -2.15
        chips = VGroup()
        for h in heads:
            txt = T("norm", size=12, color=MINT, weight="MEDIUM")
            bx = RoundedRectangle(corner_radius=0.07, width=0.68, height=0.32,
                                  stroke_color=MINT, stroke_opacity=0.75,
                                  stroke_width=1.3, fill_color=BLACK,
                                  fill_opacity=1.0)
            chip = VGroup(bx, txt.move_to(bx))
            chip.move_to(h.get_center() + UP * 0.52)
            chips.add(chip)
        self.play(*[FadeIn(h, shift=UP * 0.10) for h in heads], run_time=1.2)
        self.play(*[FadeIn(c, scale=0.7) for c in chips], run_time=1.1)
        self.play(Indicate(chips, scale_factor=1.10, color=MINT), run_time=0.8)

        # s08.5 — one shared gain per layer, for queries and for keys
        self.cue("s08.5")
        gamma = pill(f"γ   {HEAD_DIM} numbers", color=MINT, size=16,
                     stroke=MINT)
        gamma.move_to([heads[0].get_center()[0], -0.35, 0])
        gamma_q = T("one gain for q", size=15, color=BLUE)
        gamma_q.move_to([-2.35, 0.35, 0])
        gamma_k = T("one gain for k", size=15, color=GOLD)
        gamma_k.move_to([2.35, 0.35, 0])
        self.play(Write(gamma), run_time=1.1)
        self.play(Write(gamma_q), Write(gamma_k), run_time=1.2)
        self.play(Indicate(gamma, scale_factor=1.10, color=MINT), run_time=0.9)
        self.play(Indicate(gamma_q, scale_factor=1.06, color=BLUE),
                  Indicate(gamma_k, scale_factor=1.06, color=GOLD), run_time=1.0)

        # s08.6 — training stability, at a tiny parameter cost
        self.cue("s08.6")
        self.play(FadeOut(gamma_q), FadeOut(gamma_k), FadeOut(gamma),
                  FadeOut(heads), FadeOut(chips), run_time=0.5)
        sliver = Rectangle(width=0.10, height=1.45, stroke_width=0,
                           fill_color=MINT, fill_opacity=0.85)
        sliver.move_to([-3.35, -2.05, 0])
        block = Rectangle(width=2.4, height=1.45, stroke_width=0,
                          fill_color=WHITE, fill_opacity=0.10)
        block.move_to([-1.55, -2.05, 0])
        cost_lab = T("tiny parameter cost", size=15, color=MINT)
        cost_lab.move_to([-2.35, -3.25, 0])
        pts = [[1.55, -1.80, 0], [2.45, -2.30, 0], [3.35, -2.60, 0],
               [4.25, -2.77, 0], [5.15, -2.85, 0]]
        loss = VMobject(stroke_color=MINT, stroke_width=3.0).set_points_as_corners(
            [np.array(p) for p in pts])
        loss_lab = T("steady loss curve", size=15, color=MINT)
        loss_lab.move_to([3.35, -3.25, 0])
        self.play(FadeIn(sliver), FadeIn(block), run_time=0.8)
        self.play(Write(cost_lab), run_time=0.7)
        self.play(Create(loss), run_time=0.8)
        self.play(Write(loss_lab), run_time=0.7)

        # s08.7 — LLaMA 3 itself has no QK-norm
        self.cue("s08.7")
        self.play(*[FadeOut(m) for m in
                    (sliver, block, cost_lab, loss, loss_lab,
                     q_in, k_in, nq, nk, rope_q, rope_k, a1, a2, a3, a4)],
                  run_time=0.6)
        caveat = card(9.4, 1.55, edge=MINT)
        caveat.move_to([0, 0, 0])
        caveat_t = T("a Lite addition — LLaMA 3 itself has none", size=22,
                     color=WHITE, weight="MEDIUM")
        caveat_t.move_to(caveat)
        self.play(Create(caveat), run_time=0.8)
        self.play(Write(caveat_t), run_time=1.2)
        self.play(Indicate(caveat_t, scale_factor=1.05, color=MINT),
                  run_time=0.8)

        self.fade_content(0.40)
        self.sync_end()

class Section09(Section):
    SEC = "sec09_gqa"
    CHAPTER = "09 · GQA + KV CACHE"
    ACCENT = GOLD

    def construct(self):
        self._start_canvas_life()

        def cache_stack(nbars, color, label):
            bars_g = VGroup()
            for _ in range(nbars):
                bars_g.add(RoundedRectangle(corner_radius=0.03, width=0.95,
                                            height=0.11, stroke_color=color,
                                            stroke_opacity=0.5, stroke_width=1.0,
                                            fill_color=color, fill_opacity=0.22))
            bars_g.arrange(DOWN, buff=0.035)
            lab = DIN_T(label, size=TYPE["micro"], color=MUTED)
            return VGroup(bars_g, lab)

        # s09.1 — training attention sees the whole sentence at once
        self.cue("s09.1")
        hdr = self.section_header("SHARING THE CACHE", "Grouped-query attention",
                                  run_time=0.7, ty=2.78, size=32)
        words = [it["text"] for it in TOKENS["items"]]
        row = VGroup(*[token_pill(w, size=18) for w in words])
        row.arrange(RIGHT, buff=0.28).move_to([0, 1.75, 0])
        self.play(*[FadeIn(t, shift=UP * 0.10) for t in row], run_time=1.2)
        self.play(Indicate(row, scale_factor=1.05, color=GOLD), run_time=0.8)

        # s09.2 — generation writes one token, then the next
        self.cue("s09.2")
        self.play(FadeOut(hdr), run_time=0.5)
        full_strip = VGroup(*[token_pill(w, size=17, color=GOLD) for w in words])
        full_strip.arrange(RIGHT, buff=0.26).move_to([-1.45, 0.25, 0])
        strip = VGroup(*full_strip[:4])
        cursor = RoundedRectangle(corner_radius=0.03, width=0.11, height=0.42,
                                  stroke_width=0, fill_color=ROSE,
                                  fill_opacity=1)
        cursor.next_to(strip, RIGHT, buff=0.16)
        self.play(*[FadeIn(t, shift=UP * 0.10) for t in strip], run_time=1.6)
        self.play(FadeIn(cursor), run_time=0.6)
        for chip in full_strip[4:]:
            self.play(FadeIn(chip, shift=UP * 0.10), run_time=0.55)
            strip.add(chip)
        self.play(cursor.animate.next_to(strip, RIGHT, buff=0.16), run_time=0.6)

        # s09.3 — each new token still sees everything before it
        self.cue("s09.3")
        arcs = VGroup()
        for t in strip[:-1]:
            arcs.add(CurvedArrow(strip[-1].get_bottom(), t.get_bottom(),
                                 stroke_color=GOLD, stroke_width=1.5,
                                 tip_length=0.10, angle=-0.45,
                                 stroke_opacity=0.55, fill_opacity=0))
        self.play(*[Create(a) for a in arcs], run_time=1.4)
        self.play(Indicate(strip[-1], scale_factor=1.10, color=GOLD),
                  run_time=0.8)

        # s09.4 — recomputing every past key and value would waste work
        self.cue("s09.4")
        ghosts = VGroup(*[token_pill(w, size=17, color=MUTED) for w in words])
        ghosts.arrange(RIGHT, buff=0.26).move_to(strip.get_center() + DOWN * 0.72)
        ghosts.set_opacity(0.40)
        waste = T("recompute everything, every step", size=16, color=ROSE)
        waste.move_to([-1.45, -1.35, 0])
        self.play(*[FadeIn(g) for g in ghosts], run_time=1.1)
        self.play(Write(waste), run_time=0.8)
        self.play(LaggedStart(*[FadeOut(g) for g in ghosts], lag_ratio=0.08),
                  FadeOut(waste), run_time=1.3)

        # s09.5 — a KV cache stores those vectors instead
        self.cue("s09.5")
        cache_x = 4.25
        cache_title = T("KV cache — per layer × past token", size=14,
                        color=MUTED).move_to([cache_x, 1.15, 0])
        stack = VGroup()
        for i in range(6):
            stack.add(RoundedRectangle(corner_radius=0.03, width=1.55,
                                       height=0.11, stroke_color=MINT,
                                       stroke_opacity=0.5, stroke_width=1.0,
                                       fill_color=MINT, fill_opacity=0.22))
        stack.arrange(DOWN, buff=0.10).move_to([cache_x, -0.25, 0])
        self.play(Write(cache_title), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.12) for b in stack],
                              lag_ratio=0.12), run_time=1.4)

        # s09.6 — two vectors per token for every key-value head
        self.cue("s09.6")
        kt = pill("K", color=GOLD, size=15, stroke=GOLD)
        vt = pill("V", color=MINT, size=15, stroke=MINT)
        pair = VGroup(kt, vt).arrange(RIGHT, buff=0.22)
        pair.move_to([cache_x, -1.45, 0])
        pair_lab = T("one K and one V per token", size=14, color=MUTED)
        pair_lab.next_to(pair, DOWN, buff=0.18)
        self.play(FadeIn(kt), FadeIn(vt), run_time=0.9)
        self.play(Write(pair_lab), run_time=0.8)
        self.play(Indicate(pair, scale_factor=1.08, color=GOLD), run_time=0.8)

        # s09.7 — one token costs 16 × 2 × 4 × 128 stored numbers
        self.cue("s09.7")
        self.play(*[FadeOut(m) for m in
                    (row, strip, cursor, arcs, cache_title, stack, pair,
                     pair_lab)], run_time=0.9)
        math_card = card(10.2, 2.35, edge=GOLD)
        math_card.move_to([0, 0.85, 0])
        terms = [f"{N_LAYERS}", "2", f"{N_KV_HEADS}", f"{HEAD_DIM}"]
        names = ["layers", "K + V", "KV heads", "per head"]
        trow, nrow = VGroup(), VGroup()
        for t, n in zip(terms, names):
            trow.add(T(t, size=34, color=GOLD, weight="BOLD"))
            nrow.add(DIN_T(n, size=12, color=MUTED, tracked=False))
        trow.arrange(RIGHT, buff=0.85).move_to([0, 1.15, 0])
        nrow.arrange(RIGHT, buff=1.15).move_to([0, 0.55, 0])
        eq = T(f"= {N_LAYERS * 2 * N_KV_HEADS * HEAD_DIM:,}", size=34,
               color=WHITE, weight="BOLD")
        eq.move_to([0, -0.15, 0])
        xlab = VGroup(*[T("×", size=22, color=DIM) for _ in range(3)])
        for i, x in enumerate(xlab):
            x.move_to([(trow[i].get_right()[0] + trow[i + 1].get_left()[0]) / 2,
                       1.15, 0])
        self.play(Create(math_card), run_time=0.8)
        self.play(*[Write(t) for t in trow], run_time=1.3)
        self.play(*[Write(x) for x in xlab], Write(eq), run_time=1.2)
        self.play(*[Write(n) for n in nrow], run_time=1.2)
        self.play(Indicate(eq, scale_factor=1.08, color=GOLD), run_time=0.9)

        # s09.8 — that count assumes HEAD_DIM numbers per head
        self.cue("s09.8")
        hl = RoundedRectangle(corner_radius=0.06, width=1.55, height=0.72,
                              stroke_color=MINT, stroke_width=2.2,
                              stroke_opacity=0.95, fill_color=MINT,
                              fill_opacity=0.16)
        hl.move_to(trow[3])
        self.play(GrowFromCenter(hl), run_time=0.9)
        self.play(Indicate(trow[3], scale_factor=1.12, color=MINT), run_time=0.8)

        # s09.9 — at full context the cache passes thirty-three million
        self.cue("s09.9")
        self.play(FadeOut(math_card), FadeOut(trow), FadeOut(nrow),
                  FadeOut(xlab), FadeOut(eq), FadeOut(hl), run_time=0.8)
        ruler = Rectangle(width=10.4, height=0.26, stroke_color=WHITE,
                          stroke_opacity=0.22, stroke_width=1.2,
                          fill_color=WHITE, fill_opacity=0.07)
        ruler.move_to([0, -0.15, 0])
        ruler_lab = DIN_T(f"CONTEXT {SEQ_LEN:,}", size=TYPE["micro"],
                          color=MUTED)
        ruler_lab.next_to(ruler, UP, buff=0.18)
        growth = Rectangle(width=10.4, height=1.55, stroke_color=MINT,
                           stroke_opacity=0.5, stroke_width=1.4,
                           fill_color=MINT, fill_opacity=0.16)
        growth.move_to([0, -1.15, 0])
        grow_lab = T(f"cache at full context  ·  "
                     f"{N_LAYERS * 2 * N_KV_HEADS * HEAD_DIM * SEQ_LEN:,} numbers",
                     size=17, color=MINT)
        grow_lab.move_to([0, -2.45, 0])
        self.play(Create(ruler), Write(ruler_lab), run_time=1.0)
        self.play(GrowFromEdge(growth, DOWN), run_time=1.4)
        self.play(Write(grow_lab), run_time=1.0)
        self.play(Indicate(growth, scale_factor=1.03, color=MINT), run_time=0.8)

        # s09.10 — the cache is a growing memory cost at inference
        self.cue("s09.10")
        self.play(FadeOut(ruler), FadeOut(ruler_lab), FadeOut(growth),
                  FadeOut(grow_lab), run_time=0.5)
        meter = Rectangle(width=9.2, height=1.35, stroke_color=WHITE,
                          stroke_opacity=0.28, stroke_width=1.4,
                          fill_color=WHITE, fill_opacity=0.05)
        meter.move_to([0, -0.35, 0])
        meter_fill = Rectangle(width=7.6, height=1.35, stroke_width=0,
                               fill_color=MINT, fill_opacity=0.30)
        meter_fill.move_to([0, -0.35, 0])
        meter_lab = DIN_T("MEMORY AT INFERENCE", size=TYPE["tag"], color=MUTED)
        meter_lab.next_to(meter, DOWN, buff=0.22)
        self.play(Create(meter), run_time=0.7)
        self.play(GrowFromEdge(meter_fill, LEFT), run_time=0.8)
        self.play(Write(meter_lab), run_time=0.7)

        # s09.11 — Lite runs N_HEADS query heads and N_KV_HEADS key-value heads
        self.cue("s09.11")
        self.play(FadeOut(meter), FadeOut(meter_fill), FadeOut(meter_lab),
                  run_time=0.5)
        qrow = VGroup(*[head_badge(f"q{i + 1}", BLUE, r=0.18)
                        for i in range(N_HEADS)])
        qrow.arrange(RIGHT, buff=0.34).move_to([0, 1.55, 0])
        kvrow = VGroup(*[head_badge(f"kv{i + 1}", GOLD, r=0.20)
                         for i in range(N_KV_HEADS)])
        kvrow.arrange(RIGHT, buff=0.95).move_to([0, -0.55, 0])
        qlab = DIN_T(f"{N_HEADS} QUERY HEADS", size=TYPE["tag"], color=BLUE)
        qlab.move_to([0, 2.35, 0])
        kvl = DIN_T(f"{N_KV_HEADS} KEY-VALUE HEADS", size=TYPE["tag"],
                    color=GOLD)
        kvl.move_to([0, -1.45, 0])
        self.play(*[FadeIn(h, shift=UP * 0.12) for h in qrow],
                  Write(qlab), run_time=1.3)
        self.play(*[FadeIn(h, shift=UP * 0.12) for h in kvrow],
                  Write(kvl), run_time=1.3)
        self.play(Indicate(qrow, scale_factor=1.05, color=BLUE),
                  Indicate(kvrow, scale_factor=1.05, color=GOLD), run_time=1.1)

        # s09.12 — each key-value head is shared by two query heads
        self.cue("s09.12")
        per_group = N_HEADS // N_KV_HEADS
        pair_cols = [BLUE, VIOLET]
        arcs = VGroup()
        for i in range(N_HEADS):
            arcs.add(CurvedArrow(qrow[i].get_bottom(),
                                 kvrow[i // per_group].get_top(),
                                 stroke_color=pair_cols[i % 2], stroke_width=1.8,
                                 tip_length=0.11, angle=-0.35,
                                 stroke_opacity=0.65, fill_opacity=0))
        share = T(f"each KV head shared by {per_group} query heads", size=15,
                  color=MUTED)
        share.move_to([0, -2.15, 0])
        self.play(*[Create(a) for a in arcs], run_time=1.4)
        self.play(Write(share), run_time=0.8)

        # s09.13 — this layout is grouped-query attention
        self.cue("s09.13")
        glab = pill(f"grouped-query attention   {N_HEADS} Q : {N_KV_HEADS} KV",
                    color=GOLD, size=15, stroke=GOLD)
        glab.move_to([0, -2.85, 0])
        self.play(FadeIn(glab, shift=DOWN * 0.12), run_time=0.8)
        self.play(Indicate(glab, scale_factor=1.06, color=GOLD), run_time=0.8)

        # s09.14 — four KV heads halve the cache against eight
        self.cue("s09.14")
        self.play(FadeOut(qrow), FadeOut(kvrow), FadeOut(arcs),
                  FadeOut(qlab), FadeOut(kvl), FadeOut(share), FadeOut(glab),
                  run_time=0.7)
        n_mha, n_gqa = cache_bar_counts()
        cache_x = -2.2
        mha_cache = cache_stack(n_mha, ROSE, f"MHA · {N_HEADS} KV heads")
        mha_cache[0].move_to([cache_x - 1.35, 0.15, 0])
        mha_cache[1].move_to([mha_cache[0].get_center()[0], -1.85, 0])
        gqa_cache = cache_stack(n_gqa, MINT, f"GQA · {N_KV_HEADS} KV heads")
        gqa_cache[0].move_to([cache_x + 1.85, -0.35, 0])
        gqa_cache[1].move_to([gqa_cache[0].get_center()[0], -1.85, 0])
        arrow_halve = data_arrow(mha_cache[0].get_right() + RIGHT * 0.15,
                                 gqa_cache[0].get_left() + LEFT * 0.15, MINT,
                                 w=2.4)
        halve_lab = T("÷ 2", size=20, color=MINT, weight="BOLD")
        halve_lab.move_to(arrow_halve.get_center() + UP * 0.32)
        self.play(FadeIn(mha_cache[0]), Write(mha_cache[1]), run_time=1.1)
        self.play(FadeIn(gqa_cache[0]), Write(gqa_cache[1]), run_time=1.1)
        self.play(Create(arrow_halve), Write(halve_lab), run_time=1.1)
        self.play(Indicate(gqa_cache[0], scale_factor=1.04, color=MINT),
                  run_time=0.9)

        # s09.15 — the attention operation itself is unchanged
        self.cue("s09.15")
        same = T("the attention math is unchanged", size=19, color=WHITE,
                 weight="MEDIUM")
        same.move_to([4.15, 1.35, 0])
        warn = T("a smaller KV set can still\nchange model quality", size=15,
                 color=GOLD)
        warn.move_to([4.15, 0.35, 0])
        self.play(Write(same), run_time=1.0)
        self.play(Write(warn), run_time=1.1)
        self.play(Indicate(same, scale_factor=1.05, color=WHITE), run_time=0.8)
        self.play(Indicate(gqa_cache[1], scale_factor=1.10, color=MINT),
                  run_time=0.8)

        self.fade_content(0.40)
        self.sync_end()

# =========================================================== scene 10 ======
# =========================================================== scene 10 ======
class Section10(Section):
    SEC = "sec10_swiglu"
    CHAPTER = "10 · SWIGLU"
    ACCENT = VIOLET

    def construct(self):
        self._start_canvas_life()

        # s10.1 — two sublayers sit in every block
        self.cue("s10.1")
        hdr = self.section_header("THE GATED MLP", "SwiGLU — the gated feed-forward",
                                  run_time=0.8, ty=2.78, size=32)
        block = DashedVMobject(RoundedRectangle(corner_radius=0.16, width=4.6,
                                                height=3.4, stroke_color=WHITE,
                                                stroke_width=1.6),
                               num_dashes=36)
        block.set_stroke(opacity=0.35)
        block.move_to([-3.85, -0.15, 0])
        slot1 = card(3.7, 1.15, edge=BLUE)
        slot1.move_to([-3.85, 1.05, 0])
        slot1_lab = T("attention", size=18, color=lighter(BLUE, 0.25))
        slot1_lab.move_to(slot1)
        slot2 = card(3.7, 1.15, edge=VIOLET)
        slot2.move_to([-3.85, -1.35, 0])
        slot2_lab = T("feed-forward", size=18, color=lighter(VIOLET, 0.25))
        slot2_lab.move_to(slot2)
        self.play(Create(block), run_time=0.9)
        self.play(FadeIn(slot1), Write(slot1_lab), run_time=1.0)
        self.play(FadeIn(slot2), Write(slot2_lab), run_time=1.0)

        # s10.2 — the feed-forward is applied to each token on its own
        self.cue("s10.2")
        self.play(FadeOut(hdr), run_time=0.5)
        x_box = pill(f"x   ({D_MODEL:,})", color=WHITE, size=16, stroke=WHITE)
        x_box.move_to([1.15, -0.15, 0])
        out = pill(f"out   ({D_MODEL:,})", color=WHITE, size=16, stroke=WHITE)
        out.move_to([5.55, -0.15, 0])
        mid = card(1.55, 1.05, edge=VIOLET)
        mid.move_to([3.35, -0.15, 0])
        mid_lab = DIN_T("FFN", size=TYPE["tag"], color=VIOLET)
        mid_lab.move_to(mid)
        fa1 = dim_arrow(x_box.get_right(), mid.get_left(), WHITE, 0.55, 2.0,
                        0.13, 0.12)
        fa2 = dim_arrow(mid.get_right(), out.get_left(), WHITE, 0.55, 2.0,
                        0.13, 0.12)
        self.play(FadeIn(x_box), FadeIn(out), run_time=0.9)
        self.play(Create(fa1), Create(fa2), FadeIn(mid), Write(mid_lab),
                  run_time=1.0)

        # s10.3 — attention mixes tokens; the FFN works on each vector alone
        self.cue("s10.3")
        top_row = VGroup(*[token_pill(w, size=16, color=BLUE) for w in
                           [it["text"] for it in TOKENS["items"][:4]]])
        top_row.arrange(RIGHT, buff=0.30).move_to([3.35, 1.65, 0])
        mix = VGroup(*[CurvedArrow(top_row[i].get_bottom(),
                                   top_row[i + 1].get_bottom(),
                                   stroke_color=BLUE, stroke_width=1.6,
                                   tip_length=0.11, angle=-0.5,
                                   stroke_opacity=0.6, fill_opacity=0)
                       for i in range(3)])
        mix_lab = T("attention mixes tokens", size=15, color=BLUE)
        mix_lab.move_to([3.35, 2.55, 0])
        alone_lab = T("feed-forward: one vector at a time", size=15,
                      color=VIOLET)
        alone_lab.move_to([3.35, -1.55, 0])
        self.play(*[FadeIn(t, shift=UP * 0.10) for t in top_row],
                  Write(mix_lab), run_time=1.3)
        self.play(*[Create(a) for a in mix], run_time=1.1)
        self.play(Write(alone_lab), run_time=0.8)
        self.play(Indicate(x_box, scale_factor=1.08, color=VIOLET),
                  Indicate(out, scale_factor=1.08, color=VIOLET), run_time=0.8)

        # s10.4 — the vector widens to D_FF, then compresses back
        self.cue("s10.4")
        self.play(FadeOut(top_row), FadeOut(mix), FadeOut(mix_lab),
                  FadeOut(alone_lab), FadeOut(slot1), FadeOut(slot1_lab),
                  FadeOut(slot2), FadeOut(slot2_lab), FadeOut(block),
                  FadeOut(fa1), FadeOut(fa2), FadeOut(mid), FadeOut(mid_lab),
                  run_time=0.8)
        self.play(x_box.animate.move_to([-5.15, -0.15, 0]),
                  out.animate.move_to([5.55, -0.15, 0]), run_time=0.7)
        wide = Rectangle(width=2.45, height=3.15, stroke_color=VIOLET,
                         stroke_opacity=0.65, stroke_width=1.8,
                         fill_color=VIOLET, fill_opacity=0.10)
        wide.move_to([0.35, -0.15, 0])
        narrow = Rectangle(width=0.75, height=1.05, stroke_color=WHITE,
                           stroke_opacity=0.35, stroke_width=1.4,
                           fill_color=WHITE, fill_opacity=0.06)
        narrow.move_to([-2.75, -0.15, 0])
        wide_lab = T(f"{D_FF:,}", size=30, color=VIOLET, weight="BOLD")
        wide_lab.move_to(wide)
        narrow_lab = T(f"{D_MODEL:,}", size=22, color=WHITE, weight="BOLD")
        narrow_lab.move_to(narrow)
        wa1 = dim_arrow(narrow.get_right(), wide.get_left(), WHITE, 0.55, 2.2,
                        0.14, 0.12)
        wa2 = dim_arrow(wide.get_right(), out.get_left(), WHITE, 0.55, 2.2,
                        0.14, 0.12)
        self.play(FadeIn(narrow), Write(narrow_lab), run_time=0.8)
        self.play(Create(wa1), FadeIn(wide), Write(wide_lab), run_time=1.2)
        self.play(Create(wa2), run_time=0.7)
        self.play(Indicate(wide, scale_factor=1.05, color=VIOLET), run_time=0.9)

        # s10.5 — the wide middle gives room to transform features
        self.cue("s10.5")
        marks = VGroup()
        for i in range(22):
            marks.add(Rectangle(width=0.14, height=0.14, stroke_width=0,
                                fill_color=VIOLET, fill_opacity=0.70))
        marks.arrange_in_grid(4, 6, buff=(0.28, 0.30))
        marks.move_to(wide)
        self.play(*[FadeIn(m, scale=0.5) for m in marks], run_time=1.4)
        self.play(marks.animate.set_fill(opacity=0.45), run_time=0.8)
        self.play(Indicate(marks, scale_factor=1.08, color=VIOLET), run_time=0.9)

        # s10.6 — the feed-forward dominates the block's parameters
        self.cue("s10.6")
        self.play(FadeOut(marks), FadeOut(narrow), FadeOut(narrow_lab),
                  FadeOut(wide), FadeOut(wide_lab), FadeOut(wa1), FadeOut(wa2),
                  FadeOut(x_box), FadeOut(out), run_time=0.8)
        pb = FACTS["params_breakdown"]["per_block"]
        ffn_n = pb["gate_up_proj"] + pb["down_proj"]
        attn_n = (pb["q_proj"] + pb["k_proj"] + pb["v_proj"]
                  + pb["out_proj"])
        ffn_bar = Rectangle(width=6.4 * ffn_n / ffn_n, height=0.72,
                            stroke_width=0, fill_color=VIOLET,
                            fill_opacity=0.80)
        ffn_bar.move_to([-4.35 + ffn_bar.width / 2, 0.55, 0])
        attn_bar = Rectangle(width=6.4 * attn_n / ffn_n, height=0.72,
                             stroke_width=0, fill_color=BLUE,
                             fill_opacity=0.80)
        attn_bar.move_to([-4.35 + attn_bar.width / 2, -0.75, 0])
        ffn_lab = T(f"feed-forward   {ffn_n:,}", size=17, color=VIOLET)
        ffn_lab.next_to(ffn_bar, RIGHT, buff=0.35)
        attn_lab = T(f"attention   {attn_n:,}", size=17, color=BLUE)
        attn_lab.next_to(attn_bar, RIGHT, buff=0.35)
        param_cap = DIN_T("PARAMETERS PER BLOCK", size=TYPE["tag"], color=MUTED)
        param_cap.move_to([0, 1.75, 0])
        self.play(Write(param_cap), run_time=0.7)
        self.play(GrowFromEdge(ffn_bar, LEFT), Write(ffn_lab), run_time=1.3)
        self.play(GrowFromEdge(attn_bar, LEFT), Write(attn_lab), run_time=1.3)
        self.play(Indicate(ffn_bar, scale_factor=1.05, color=VIOLET),
                  run_time=1.0)

        # s10.7 — LLaMA gates this expansion with SwiGLU
        self.cue("s10.7")
        self.play(FadeOut(ffn_bar), FadeOut(ffn_lab), FadeOut(attn_bar),
                  FadeOut(attn_lab), FadeOut(param_cap), run_time=0.6)
        sw = pill("SwiGLU", color=VIOLET, size=26, stroke=VIOLET)
        sw.move_to([0, 0.35, 0])
        sw_sub = T("gated feed-forward", size=18, color=MUTED)
        sw_sub.move_to([0, -0.75, 0])
        self.play(FadeIn(sw, scale=0.8), run_time=1.0)
        self.play(Write(sw_sub), run_time=0.8)

        # s10.8 — the wide layer forks into a gate path and an up path
        self.cue("s10.8")
        self.play(FadeOut(sw), FadeOut(sw_sub), run_time=0.5)
        xin = pill(f"x   ({D_MODEL:,})", color=WHITE, size=16, stroke=WHITE)
        xin.move_to([-5.15, 0.55, 0])
        gate = card(2.35, 0.85, edge=GOLD)
        gate.move_to([-0.85, 1.75, 0])
        gate_lab = T(f"W_gate  → {D_FF:,}", size=16, color=lighter(GOLD, 0.25))
        gate_lab.move_to(gate)
        up = card(2.35, 0.85, edge=BLUE)
        up.move_to([-0.85, -0.65, 0])
        up_lab = T(f"W_up  → {D_FF:,}", size=16, color=lighter(BLUE, 0.25))
        up_lab.move_to(up)
        f1 = dim_arrow(xin.get_right() + UP * 0.10, gate.get_left(), GOLD,
                       0.65, 2.0, 0.13, 0.12)
        f2 = dim_arrow(xin.get_right() + DOWN * 0.10, up.get_left(), BLUE,
                       0.65, 2.0, 0.13, 0.12)
        self.play(FadeIn(xin), run_time=0.7)
        self.play(Create(f1), Create(f2), run_time=1.0)
        self.play(FadeIn(gate), Write(gate_lab), FadeIn(up), Write(up_lab),
                  run_time=1.2)

        # s10.9 — both paths start from the same input vector
        self.cue("s10.9")
        fork = glow_dot(xin.get_right() + RIGHT * 0.14, MINT, r=0.07)
        self.play(FadeIn(fork), run_time=0.8)
        self.play(Indicate(fork, scale_factor=1.35, color=MINT), run_time=0.7)

        # s10.10 — the gate path passes through SiLU
        self.cue("s10.10")
        plot_card = card(4.15, 2.55, edge=GOLD)
        plot_card.move_to([3.85, 1.45, 0])
        ax = Axes(x_range=[-3.5, 3.5, 1], y_range=[-1.5, 4, 1], x_length=2.6,
                  y_length=1.55, tips=False,
                  axis_config={"stroke_color": MUTED, "stroke_opacity": 0.35,
                               "stroke_width": 1.1})
        ax.move_to([3.85, 1.45, 0])
        pgrid = VGroup(
            *[Line(ax.c2p(-3.5, y), ax.c2p(3.5, y), stroke_color=WHITE,
                   stroke_opacity=0.07, stroke_width=0.8) for y in (1, 2, 3)],
            *[Line(ax.c2p(x, -1.5), ax.c2p(x, 4), stroke_color=WHITE,
                   stroke_opacity=0.07, stroke_width=0.8) for x in (-2, 2)])
        silu = ax.plot(lambda t: t / (1 + np.exp(-t)), color=GOLD,
                       stroke_width=3.0)
        relu = DashedVMobject(ax.plot(lambda t: np.maximum(0, t),
                                      color=MUTED, stroke_width=1.8),
                              num_dashes=22)
        sl = TY('"SiLU"(x) = x dot sigma(x)', size=17, color=GOLD)
        sl.move_to([3.85, 2.35, 0])
        rl = DIN_T("RELU — DASHED", size=TYPE["micro"], color=MUTED)
        rl.move_to([3.85, 0.45, 0])
        self.play(Create(plot_card), FadeIn(pgrid), Create(ax), run_time=0.9)
        self.play(Create(relu), Write(rl), run_time=1.0)
        self.play(Create(silu), Write(sl), run_time=1.1)

        # s10.11 — near zero SiLU passes about half; large positives pass through
        self.cue("s10.11")
        m1 = glow_dot(ax.c2p(0.0, 0.0), GOLD, r=0.10)
        m2 = glow_dot(ax.c2p(3.0, 3.0 / (1 + np.exp(-3.0))), GOLD, r=0.10)
        m3 = glow_dot(ax.c2p(-2.5, -2.5 / (1 + np.exp(2.5))), ROSE, r=0.10)
        lab1 = T("near 0 → about half", size=13, color=GOLD)
        lab1.move_to([3.15, -1.05, 0])
        lab2 = T("large + → nearly unchanged", size=13, color=GOLD)
        lab2.move_to([3.15, -1.55, 0])
        self.play(FadeIn(m1), FadeIn(m2), run_time=1.0)
        self.play(Write(lab1), run_time=0.8)
        self.play(Write(lab2), run_time=0.8)
        self.play(FadeIn(m3), run_time=0.8)
        self.play(Indicate(m1, scale_factor=1.25, color=GOLD),
                  Indicate(m2, scale_factor=1.25, color=GOLD), run_time=0.8)

        # s10.12 — element by element, the gate path scales the up path
        self.cue("s10.12")
        self.play(FadeOut(plot_card), FadeOut(pgrid), FadeOut(ax), FadeOut(silu), FadeOut(relu),
                  FadeOut(sl), FadeOut(rl), FadeOut(m1), FadeOut(m2),
                  FadeOut(m3), FadeOut(lab1), FadeOut(lab2), run_time=0.5)
        sil = pill("SiLU", color=GOLD, size=15, stroke=GOLD)
        sil.move_to([1.05, 1.75, 0])
        mult = odot(MINT, r=0.22)
        mult.move_to([2.55, 0.55, 0])
        mult_lab = DIN_T("ELEMENT-WISE", size=TYPE["micro"], color=MINT)
        mult_lab.next_to(mult, RIGHT, buff=0.28)
        a3 = data_arrow(gate.get_right(), sil.get_left(), GOLD, w=2.2)
        a4 = data_arrow(sil.get_right(), mult.get_center() + UP * 0.30, GOLD,
                        w=2.2)
        a5 = data_arrow(up.get_right(), mult.get_center() + DOWN * 0.30, BLUE,
                        w=2.2)
        self.play(Create(a3), FadeIn(sil), run_time=0.8)
        self.play(Create(a4), Create(a5), FadeIn(mult), Write(mult_lab),
                  run_time=1.0)

        # s10.13 — the gate can shrink, pass, or amplify a feature
        self.cue("s10.13")
        silu_vals, up_vals, prod_vals = swiglu_demo()
        rows = VGroup()
        for vals, col, tag in ((silu_vals, GOLD, "gate"),
                               (up_vals, BLUE, "up"),
                               (prod_vals, MINT, "product")):
            cells = VGroup()
            for v in vals:
                cells.add(Text(f"{v:+.2f}", font=DIN, font_size=pt(13),
                               color=lighter(col, 0.30), weight="BOLD"))
            cells.arrange(RIGHT, buff=0.24)
            rows.add(VGroup(cells, DIN_T(tag, size=12, color=col,
                                         tracked=False)))
        for r, y in zip(rows, (-1.15, -1.85, -2.55)):
            r[0].move_to([1.15, y, 0])
            r[1].move_to([-2.05, y, 0])
        gcap = T("the gate scales each feature — nothing is switched off",
                 size=15, color=MINT)
        gcap.move_to([0, -3.35, 0])
        self.play(*[Write(r[0]) for r in rows[:2]],
                  *[Write(r[1]) for r in rows[:2]], run_time=1.5)
        self.play(Write(rows[2][0]), Write(rows[2][1]), run_time=1.0)
        self.play(Write(gcap), run_time=0.8)

        # s10.14 — only the product of the two paths flows onward
        self.cue("s10.14")
        self.play(FadeOut(rows[0]), FadeOut(rows[1]), run_time=0.8)
        self.play(Indicate(rows[2], scale_factor=1.06, color=MINT),
                  run_time=0.8)

        # s10.15 — multiplicative gating lets one path decide what to keep
        self.cue("s10.15")
        dial = Circle(radius=0.72, stroke_color=MUTED, stroke_opacity=0.45,
                      stroke_width=2.2, fill_color=WHITE,
                      fill_opacity=0.04).move_to([-4.15, -2.25, 0])
        dticks = dial_ticks(dial, MUTED)
        needle = Line(dial.get_center(),
                      dial.get_center() + RIGHT * 0.48 + UP * 0.40,
                      stroke_width=3.2, stroke_color=GOLD)
        dial_lab = DIN_T("GATE", size=TYPE["tag"], color=MUTED)
        dial_lab.next_to(dial, LEFT, buff=0.24)
        decide = T("the gate decides how much of the up path to keep",
                   size=16, color=GOLD)
        decide.move_to([1.35, -3.35, 0])
        self.play(FadeOut(gcap), FadeIn(dial), FadeIn(dticks), Write(dial_lab), run_time=1.0)
        self.play(GrowFromEdge(needle, LEFT), run_time=0.7)
        self.play(Write(decide), run_time=0.9)
        self.play(Rotate(needle, angle=TAU / 4, about_point=dial.get_center()),
                  run_time=1.0)
        self.play(Rotate(needle, angle=-TAU / 6, about_point=dial.get_center()),
                  run_time=0.8)

        # s10.16 — a final matrix projects back into the residual stream
        self.cue("s10.16")
        self.play(FadeOut(dial), FadeOut(dticks), FadeOut(needle), FadeOut(dial_lab),
                  FadeOut(decide), FadeOut(mult_lab), FadeOut(rows[2]),
                  run_time=0.5)
        down = card(2.35, 0.85, edge=MINT)
        down.move_to([4.55, 0.55, 0])
        down_lab = T(f"W_down  → {D_MODEL:,}", size=16,
                     color=lighter(MINT, 0.25))
        down_lab.move_to(down)
        out2 = pill(f"out   ({D_MODEL:,})", color=WHITE, size=16,
                    stroke=WHITE)
        out2.move_to([4.55, -0.85, 0])
        fx_card = card(5.6, 0.85, edge=MINT)
        fx_card.move_to([3.45, -3.1, 0])
        fx = TY('"FFN"(x) = W_("down")("SiLU"(W_("gate") x) dot W_("up") x)',
                size=22, color=WHITE).move_to(fx_card)
        a6 = data_arrow(mult.get_right(), down.get_left(), MINT, w=2.4)
        a7 = dim_arrow(down.get_bottom(), out2.get_top(), WHITE, 0.55, 2.0,
                       0.13, 0.12)
        self.play(Create(a6), FadeIn(down), Write(down_lab), run_time=0.9)
        self.play(Create(a7), FadeIn(out2), run_time=0.8)
        self.play(Create(fx_card), Write(fx), run_time=0.8)

        self.fade_content(0.40)
        self.sync_end()

# =========================================================== scene 10 ======
# =========================================================== scene 11 ======
class Section11(Section):
    SEC = "sec11_loop"
    CHAPTER = "11 · THE GENERATION LOOP"
    ACCENT = ROSE

    def construct(self):
        self._start_canvas_life()

        # s11.1 — after N_LAYERS blocks, one last RMSNorm settles the stream
        self.cue("s11.1")
        hdr = self.section_header("CLOSING THE LOOP",
                                  "Unembedding and autoregression",
                                  run_time=0.8, ty=2.78, size=32)
        stack = VGroup()
        for _ in range(N_LAYERS):
            stack.add(RoundedRectangle(corner_radius=0.045, width=2.35,
                                       height=0.115, stroke_color=WHITE,
                                       stroke_opacity=0.18, stroke_width=1.0,
                                       fill_color=WHITE, fill_opacity=0.055))
        stack.arrange(DOWN, buff=0.10).move_to([-4.35, 0.35, 0])
        slab = DIN_T(f"× {N_LAYERS} BLOCKS", size=TYPE["micro"], color=MUTED)
        slab.next_to(stack, RIGHT, buff=0.30)
        fin = pill("final RMSNorm", color=GOLD, size=15, stroke=GOLD)
        fin.move_to([-4.35, -2.15, 0])
        self.play(LaggedStart(*[FadeIn(s) for s in stack], lag_ratio=0.05),
                  Write(slab), run_time=1.3)
        self.play(Write(fin), run_time=0.8)
        self.play(Indicate(fin, scale_factor=1.08, color=GOLD), run_time=0.7)

        # s11.2 — a separate, untied matrix scores every candidate
        self.cue("s11.2")
        self.play(FadeOut(hdr), run_time=0.5)
        wu = pill(f"W_U   {D_MODEL:,} → {VOCAB_LABEL}", color=ROSE, size=15,
                  stroke=ROSE)
        wu.move_to([0, 2.55, 0])
        logits = VGroup(*[Text(v, font=DIN, font_size=pt(13), color=MUTED,
                               weight="BOLD")
                          for v in ["17.2", "15.8", "14.1", "13.6", "12.9",
                                    "…"]])
        logits.arrange(RIGHT, buff=0.42).move_to([1.65, 0.75, 0])
        ruler = Rectangle(width=8.4, height=0.22, stroke_color=WHITE,
                          stroke_opacity=0.22, stroke_width=1.1,
                          fill_color=WHITE, fill_opacity=0.06)
        ruler.move_to([1.65, 0.05, 0])
        ruler_lab = DIN_T(f"VOCABULARY  {VOCAB_LABEL}", size=TYPE["micro"],
                          color=MUTED)
        ruler_lab.next_to(ruler, DOWN, buff=0.16)
        self.play(Write(wu), run_time=0.9)
        self.play(Create(ruler), Write(ruler_lab), run_time=1.0)
        self.play(*[Write(l) for l in logits], run_time=1.4)
        self.play(Indicate(wu, color=ROSE), run_time=0.8)

        # s11.3 — each logit is one dot product with a candidate row
        self.cue("s11.3")
        one = logits[0].copy()
        one.move_to([-0.35, -1.05, 0]).scale(1.9)
        a1 = dim_arrow(logits[0].get_bottom(), one.get_top(), ROSE, 0.7, 2.2,
                       0.14, 0.12)
        eq = TY('"logit"_i = x dot W_U[i]', size=22, color=WHITE)
        eq.move_to([-0.35, -2.05, 0])
        self.play(Create(a1), ReplacementTransform(logits[0].copy(), one),
                  run_time=1.3)
        self.play(Write(eq), run_time=1.1)
        self.play(Indicate(one, scale_factor=1.10, color=ROSE), run_time=0.8)

        # s11.4 — the unembedding table is as large as the embedding table
        self.cue("s11.4")
        self.play(FadeOut(a1), FadeOut(one), FadeOut(eq), FadeOut(ruler),
                  FadeOut(ruler_lab), run_time=0.6)
        emb = card(3.35, 2.35, edge=BLUE)
        emb.move_to([-0.25, -1.25, 0])
        emb_lab = T("embedding", size=19, color=lighter(BLUE, 0.25))
        emb_lab.move_to(emb.get_center() + UP * 0.68)
        emb_n = T(f"{FACTS['params_breakdown']['top_level']['input_embedding']:,}",
                  size=24, color=BLUE, weight="BOLD")
        emb_n.move_to(emb.get_center())
        emb_c = DIN_T("PARAMETERS", size=12, color=MUTED, tracked=False)
        emb_c.move_to(emb.get_center() + DOWN * 0.68)
        un = card(3.35, 2.35, edge=ROSE)
        un.move_to([3.65, -1.25, 0])
        un_lab = T("unembedding", size=19, color=lighter(ROSE, 0.25))
        un_lab.move_to(un.get_center() + UP * 0.72)
        un_n = T(f"{FACTS['params_breakdown']['top_level']['output_proj']:,}",
                 size=24, color=ROSE, weight="BOLD")
        un_n.move_to(un.get_center())
        un_c = DIN_T("PARAMETERS", size=12, color=MUTED, tracked=False)
        un_c.move_to(un.get_center() + DOWN * 0.72)
        self.play(Create(emb), Write(emb_lab), Write(emb_n), Write(emb_c),
                  run_time=1.4)
        self.play(Create(un), Write(un_lab), Write(un_n), Write(un_c),
                  run_time=1.4)
        self.play(Indicate(emb_n, scale_factor=1.08, color=BLUE),
                  Indicate(un_n, scale_factor=1.08, color=ROSE), run_time=1.1)

        # s11.5 — together the two tables hold just over half the model
        self.cue("s11.5")
        self.play(FadeOut(emb), FadeOut(emb_lab), FadeOut(emb_n), FadeOut(emb_c),
                  FadeOut(un), FadeOut(un_lab), FadeOut(un_n), FadeOut(un_c),
                  run_time=0.6)
        tb = FACTS["params_breakdown"]["top_level"]
        tot = FACTS["params"]
        tables = tb["input_embedding"] + tb["output_proj"]
        bar_bg = Rectangle(width=6.8, height=0.75, stroke_color=WHITE,
                           stroke_opacity=0.24, stroke_width=1.2,
                           fill_color=WHITE, fill_opacity=0.05)
        bar_bg.move_to([1.35, -1.35, 0])
        bar_t = Rectangle(width=6.8 * tables / tot, height=0.75, stroke_width=0,
                          fill_color=ROSE, fill_opacity=0.80)
        bar_t.move_to(bar_bg.get_left() + RIGHT * bar_t.width / 2)
        bar_t.set_y(bar_bg.get_y())
        half_lab = T(f"the two tables  {tables / tot:.0%} of the model",
                     size=18, color=ROSE)
        half_lab.move_to([1.35, -2.15, 0])
        self.play(Create(bar_bg), run_time=0.8)
        self.play(GrowFromEdge(bar_t, LEFT), run_time=1.0)
        self.play(FadeIn(half_lab, shift=UP * 0.10), run_time=0.5)

        # s11.6 — large logits mean the candidate fits the context well
        self.cue("s11.6")
        self.play(FadeOut(bar_bg), FadeOut(bar_t), FadeOut(half_lab),
                  run_time=0.5)
        tall = logits[0].copy()
        tall.move_to([1.65, -0.85, 0]).scale(1.7)
        fit = T("large logit → fits the context well", size=17, color=ROSE)
        fit.move_to([1.65, -1.85, 0])
        self.play(ReplacementTransform(logits[0].copy(), tall), run_time=0.9)
        self.play(Write(fit), run_time=0.8)
        self.play(Indicate(tall, scale_factor=1.12, color=ROSE), run_time=0.7)

        # s11.7 — softmax turns the logits into probabilities
        self.cue("s11.7")
        self.play(FadeOut(tall), FadeOut(fit), FadeOut(logits), run_time=0.5)
        pb = bars(["mat", "floor", "rug", "…"], [0.82, 0.07, 0.04, 0.07],
                  [ROSE, MUTED, MUTED, MUTED], width=2.6, size=16)
        pb.move_to([1.65, -0.35, 0])
        softmax_lab = TY('p = "softmax"("logits")', size=20, color=MUTED)
        softmax_lab.move_to([1.65, -2.35, 0])
        self.play(GrowFromEdge(pb, DOWN), run_time=1.4)
        self.play(Write(softmax_lab), run_time=1.0)
        self.play(Indicate(pb, scale_factor=1.04, color=ROSE), run_time=0.8)

        # s11.8 — sampling picks one token and appends it to the context
        self.cue("s11.8")
        words = [it["text"] for it in TOKENS["items"]]
        ctx = VGroup(*[token_pill(w, color=BLUE) for w in words[:5]])
        ctx.arrange(RIGHT, buff=0.24).move_to([1.85, 1.45, 0])
        mat_chip = token_pill(words[-1], color=ROSE)
        mat_chip.next_to(ctx, RIGHT, buff=0.42)
        self.play(*[FadeIn(c, shift=UP * 0.10) for c in ctx], run_time=1.2)
        self.play(ReplacementTransform(pb[0][2].copy(), mat_chip[0]),
                  ReplacementTransform(pb[0][0].copy(), mat_chip[1]),
                  run_time=1.0)
        self.play(Indicate(mat_chip, scale_factor=1.10, color=ROSE),
                  run_time=0.8)

        # s11.9 — the full network runs again for the next one
        self.cue("s11.9")
        loop = CurvedArrow(mat_chip.get_top(), ctx[0].get_top(),
                           stroke_color=ROSE, fill_color=ROSE,
                           stroke_width=2.2, tip_length=0.14, angle=0.64,
                           stroke_opacity=0.85, fill_opacity=0)
        loop.tip.set_fill(ROSE, 0.95)
        loop_halo = glow_ring(loop, ROSE)
        self.play(Create(loop), FadeIn(loop_halo), run_time=1.1)

        # s11.10 — one forward pass per generated token
        self.cue("s11.10")
        loop_cap = T("one forward pass per generated token", size=17,
                     color=ROSE)
        loop_cap.move_to([-3.35, -3.15, 0])
        self.play(Write(loop_cap), run_time=1.0)
        self.play(ShowPassingFlash(loop.copy().set_stroke(ROSE, 6.0, 0.9),
                                   time_width=0.35), run_time=1.2)
        self.play(Indicate(loop, scale_factor=1.06, color=ROSE), run_time=0.8)

        # s11.11 — a KV cache would compute only the newest token
        self.cue("s11.11")
        self.play(FadeOut(pb), FadeOut(softmax_lab),
                  run_time=0.5)
        cache = VGroup()
        for i in range(5):
            cache.add(RoundedRectangle(corner_radius=0.03, width=2.15,
                                       height=0.13, stroke_color=MINT,
                                       stroke_opacity=0.5, stroke_width=1.0,
                                       fill_color=MINT, fill_opacity=0.28))
        cache.arrange(DOWN, buff=0.13).move_to([4.35, -0.85, 0])
        cache_lab = T("a cache would keep past K and V", size=15, color=MINT)
        cache_lab.next_to(cache, DOWN, buff=0.22)
        self.play(LaggedStart(*[FadeIn(b) for b in cache], lag_ratio=0.12),
                  run_time=1.2)
        self.play(Write(cache_lab), run_time=1.0)
        self.play(Indicate(cache, scale_factor=1.05, color=MINT), run_time=1.0)
        only = T("only the newest token runs the stack", size=15, color=MINT)
        only.move_to([4.35, -2.55, 0])
        self.play(Write(only), run_time=1.0)

        # s11.12 — this code re-reads the whole sequence at every step
        self.cue("s11.12")
        self.play(FadeOut(cache), FadeOut(cache_lab), FadeOut(only), run_time=0.5)
        ghost = VGroup(*[token_pill(w, size=16, color=MUTED) for w in words])
        ghost.arrange(RIGHT, buff=0.22).move_to([1.85, -1.15, 0])
        ghost.set_opacity(0.35)
        honest = T("this code re-reads the whole sequence", size=17,
                   color=ROSE)
        honest.move_to([1.85, -2.55, 0])
        self.play(*[FadeIn(g) for g in ghost], run_time=1.2)
        self.play(Write(honest), run_time=1.0)
        self.play(LaggedStart(*[FadeOut(g) for g in ghost], lag_ratio=0.08),
                  run_time=1.4)
        self.play(Indicate(honest, scale_factor=1.05, color=ROSE), run_time=0.8)

        # s11.13 — one function on a table of numbers now writes a paragraph
        self.cue("s11.13")
        self.play(FadeOut(honest), FadeOut(loop_cap), run_time=0.6)
        para = VGroup()
        for i, w in enumerate(["The", "model", "now", "writes", "a",
                               "paragraph", "from", "numbers"]):
            para.add(T(w, size=17, color=WHITE))
        para.arrange_in_grid(2, 4, buff=(0.55, 0.34))
        para.move_to([-0.35, -1.45, 0])
        para_cap = T("one function, from tokens to probabilities", size=17,
                     color=MUTED)
        para_cap.move_to([-0.35, -2.85, 0])
        self.play(*[Write(t) for t in para], run_time=1.6)
        self.play(Write(para_cap), run_time=1.0)
        self.play(Indicate(para, scale_factor=1.04, color=WHITE), run_time=0.8)

        self.fade_content(0.40)
        self.sync_end()

# =========================================================== scene 10 ======
# =========================================================== scene 12 ======
class Section12(Section):
    SEC = "sec12_recap"
    CHAPTER = "12 · THE WHOLE PICTURE"
    ACCENT = GOLD

    def construct(self):
        self._start_canvas_life()

        # s12.1 — walk the whole path once more
        self.cue("s12.1")
        hdr = self.section_header("THE WHOLE PICTURE", "The path end to end",
                                  run_time=0.8, ty=2.78, size=32)
        row1 = VGroup(pill("Embedding", color=BLUE, size=13),
                      T("→", size=15, color=DIM),
                      pill("RMSNorm · RoPE · GQA  +", color=GOLD, size=13),
                      T("→", size=15, color=DIM),
                      pill("RMSNorm · SwiGLU  +", color=VIOLET, size=13),
                      T("→", size=15, color=DIM),
                      pill("Norm", color=GOLD, size=13),
                      T("→", size=15, color=DIM),
                      pill("Unembed · softmax", color=ROSE, size=13))
        row1.arrange(RIGHT, buff=0.12).move_to([0, 2.05, 0])
        chain = Line(row1[0].get_right(), row1[-1].get_left(),
                     stroke_color=WHITE, stroke_opacity=0.12, stroke_width=1.0)
        brace = Brace(VGroup(row1[2], row1[4]), UP, buff=0.10)
        brace_lab = DIN_T(f"× {N_LAYERS} BLOCKS", size=TYPE["micro"],
                          color=MUTED)
        brace_lab.next_to(brace, UP, buff=0.10)
        self.play(FadeOut(hdr), run_time=0.5)
        self.play(Create(chain),
                  *[Write(m) if i % 2 == 0 else FadeIn(m)
                    for i, m in enumerate(row1)],
                  Create(brace), Write(brace_lab), run_time=2.2)

        # s12.2 — the tokenizer cuts the sentence into six pieces
        self.cue("s12.2")
        chips = VGroup(*[token_pill(it["text"], str(it["id"]), color=WHITE,
                                    size=15) for it in TOKENS["items"]])
        chips.arrange(RIGHT, buff=0.22).move_to([0, -0.15, 0])
        self.play(*[FadeIn(c, shift=UP * 0.10) for c in chips], run_time=1.4)
        self.play(Indicate(chips, scale_factor=1.05, color=WHITE), run_time=0.9)

        # s12.3 — the embedding table turns each ID into a vector
        self.cue("s12.3")
        mat = grid(6, 5, cell=0.20, gap=0.04)
        mat.set_stroke(BLUE, opacity=0.28).set_fill(BLUE, 0.10)
        mat.move_to([-4.55, -1.75, 0])
        mat_br = matrix_brackets(mat, BLUE, pad=0.05, tick=0.11)
        mat_lab = DIN_T(f"E   {VOCAB_LABEL} × {D_MODEL:,}", size=TYPE["micro"],
                        color=BLUE)
        mat_lab.next_to(mat, DOWN, buff=0.18)
        vecs = VGroup()
        for i in range(4):
            vecs.add(VGroup(*[Rectangle(width=0.13, height=0.30,
                                        stroke_width=0, fill_color=BLUE,
                                        fill_opacity=0.80)
                              for _ in range(5)]))
        for v in vecs:
            v.arrange(RIGHT, buff=0.06)
        vecs.arrange(DOWN, buff=0.14).move_to([-1.85, -1.85, 0])
        self.play(FadeIn(mat), Create(mat_br), Write(mat_lab), run_time=1.1)
        self.play(*[ReplacementTransform(mat[i * 5:(i + 1) * 5].copy(),
                                         vecs[i]) for i in range(4)],
                  run_time=1.5)
        self.play(Indicate(vecs, scale_factor=1.06, color=BLUE), run_time=0.8)

        # s12.4 — the residual stream carries those vectors through the blocks
        self.cue("s12.4")
        band = RoundedRectangle(corner_radius=0.18, width=11.6, height=0.85,
                                stroke_color=MUTED, stroke_opacity=0.45,
                                stroke_width=1.2, fill_color=WHITE,
                                fill_opacity=0.03)
        band.move_to([0, -1.95, 0])
        band_lab = DIN_T("THE RESIDUAL STREAM", size=TYPE["micro"], color=MUTED)
        band_lab.next_to(band, UP, buff=0.14).align_to(band, LEFT).shift(
            RIGHT * 0.25)
        self.play(FadeOut(mat), FadeOut(mat_br), FadeOut(mat_lab), Create(band), Write(band_lab),
                  run_time=1.1)
        self.play(vecs.animate.arrange(RIGHT, buff=0.16).move_to(band),
                  run_time=1.0)
        self.play(flow(band, GOLD), run_time=0.8)

        # s12.5 — one block opens: RMSNorm, attention, then SwiGLU
        self.cue("s12.5")
        self.play(FadeOut(chips), run_time=0.5)
        blk = DashedVMobject(RoundedRectangle(corner_radius=0.14, width=8.4,
                                              height=2.15, stroke_color=GOLD,
                                              stroke_width=1.8),
                             num_dashes=34)
        blk.move_to([0, 0.25, 0])
        stages = VGroup()
        for name, col in (("RMSNorm", GOLD), ("attention", BLUE),
                          ("SwiGLU", VIOLET)):
            c = card(2.35, 1.15, edge=col)
            t = T(name, size=18, color=lighter(col, 0.25)).move_to(c)
            stages.add(VGroup(c, t))
        stages.arrange(RIGHT, buff=0.85).move_to([0, 0.25, 0])
        self.play(Create(blk), run_time=1.0)
        self.play(*[FadeIn(s, shift=UP * 0.12) for s in stages], run_time=1.5)
        self.play(Indicate(stages[0], scale_factor=1.08, color=GOLD),
                  run_time=0.8)
        self.play(Indicate(stages[1], scale_factor=1.08, color=BLUE),
                  run_time=0.8)
        self.play(Indicate(stages[2], scale_factor=1.08, color=VIOLET),
                  run_time=0.8)

        # s12.6 — rotary embeddings keep the order visible
        self.cue("s12.6")
        dial = Circle(radius=0.72, stroke_color=VIOLET, stroke_opacity=0.65,
                      stroke_width=2.2, fill_color=VIOLET,
                      fill_opacity=0.08).move_to([5.45, 0.25, 0])
        needle = Line(dial.get_center(),
                      dial.get_center() + RIGHT * 0.52 + UP * 0.32,
                      stroke_width=3.0, stroke_color=VIOLET)
        dial_lab = DIN_T("RoPE", size=TYPE["micro"], color=VIOLET)
        dial_lab.next_to(dial, DOWN, buff=0.18)
        self.play(FadeIn(dial), Write(dial_lab), run_time=0.9)
        self.play(GrowFromEdge(needle, LEFT), run_time=0.6)
        self.play(Rotate(needle, angle=-TAU / 4, about_point=dial.get_center()),
                  run_time=1.0)

        # s12.7 — GQA keeps the cache affordable, QK-norm keeps logits stable
        self.cue("s12.7")
        self.play(FadeOut(blk), FadeOut(stages), FadeOut(dial),
                  FadeOut(needle), FadeOut(dial_lab), run_time=0.7)
        gqa = pill(f"GQA   {N_HEADS} Q : {N_KV_HEADS} KV", color=GOLD,
                   size=17, stroke=GOLD)
        gqa.move_to([-2.75, 0.55, 0])
        qkn = pill("QK-norm", color=MINT, size=17, stroke=MINT)
        qkn.move_to([2.75, 0.55, 0])
        gqa_c = T("affordable cache", size=15, color=MUTED)
        gqa_c.next_to(gqa, DOWN, buff=0.22)
        qkn_c = T("stable logits", size=15, color=MUTED)
        qkn_c.next_to(qkn, DOWN, buff=0.22)
        self.play(FadeIn(gqa, scale=0.8), Write(gqa_c), run_time=1.1)
        self.play(FadeIn(qkn, scale=0.8), Write(qkn_c), run_time=1.1)
        self.play(Indicate(gqa, scale_factor=1.08, color=GOLD),
                  Indicate(qkn, scale_factor=1.08, color=MINT), run_time=1.1)

        # s12.8 — the unembedding table scores every candidate; sampling takes one
        self.cue("s12.8")
        self.play(FadeOut(gqa), FadeOut(gqa_c), FadeOut(qkn), FadeOut(qkn_c),
                  FadeOut(band), FadeOut(band_lab), FadeOut(vecs), run_time=0.6)
        pb = bars(["mat", "floor", "rug", "…"], [0.82, 0.07, 0.04, 0.07],
                  [ROSE, MUTED, MUTED, MUTED], width=3.0, size=17)
        pb.move_to([-1.65, 0.15, 0])
        chosen = token_pill("mat", color=ROSE)
        chosen.move_to([3.15, 0.15, 0])
        pick = T("sampling takes one", size=17, color=ROSE)
        pick.move_to([3.15, -0.95, 0])
        self.play(GrowFromEdge(pb, DOWN), run_time=1.2)
        self.play(ReplacementTransform(pb[0][2].copy(), chosen[0]),
                  ReplacementTransform(pb[0][0].copy(), chosen[1]),
                  run_time=1.0)
        self.play(Write(pick), run_time=0.8)
        self.play(Indicate(chosen, scale_factor=1.10, color=ROSE), run_time=0.7)

        # s12.9 — the parameter count, and one function
        self.cue("s12.9")
        self.play(FadeOut(row1), FadeOut(chain), FadeOut(brace),
                  FadeOut(brace_lab), FadeOut(pb), FadeOut(chosen),
                  FadeOut(pick), run_time=0.7)
        kick = DIN_T("THE COMPLETE ARCHITECTURE", size=12, color=GOLD)
        end1 = T("LLaMA-3-Lite", size=56, color=WHITE, weight="BOLD")
        rule = RoundedRectangle(corner_radius=0.015, width=1.4, height=0.03,
                                stroke_width=0, fill_color=GOLD,
                                fill_opacity=0.9)
        end2 = T(f"{PARAMS_LABEL} parameters, one function from text to a"
                 f" sampled token.", size=20, color=MUTED)
        nums = Text(f"d_model {D_MODEL:,}   ·   {N_HEADS} Q : {N_KV_HEADS} KV"
                    f"   ·   d_ff {D_FF:,}   ·   vocab {VOCAB_LABEL}   ·   "
                    f"θ = {THETA_LABEL}   ·   ε = {EPS_LABEL}",
                    font=DIN, font_size=pt(12), color=MUTED, weight="BOLD")
        footer = DIN_T("DECODER-ONLY   ·   RoPE   ·   GQA   ·   SWIGLU   ·   RMSNORM",
                       size=TYPE["micro"], color=DIM)
        end = VGroup(kick, end1, rule, end2, nums, footer).arrange(DOWN,
                                                                   buff=0.30)
        end.move_to([0, 0.15, 0])
        wm = self.watermark()
        self.play(Write(kick), Write(end1), Create(rule), Write(end2),
                  Write(nums), Write(footer), FadeIn(wm), run_time=1.8)
        self.play(Indicate(end1, scale_factor=1.04, color=GOLD), run_time=0.8)

        # CC BY 4.0: the music credit must stay on screen. Held, not faded.
        credits = VGroup(
            T('Music: "Almost in F" by Kevin MacLeod (incompetech.com)',
              size=TYPE["micro"], color=DIM),
            T("Licensed under Creative Commons: By Attribution 4.0",
              size=TYPE["micro"], color=DIM),
        ).arrange(DOWN, buff=0.12).move_to([0, -3.20, 0])
        self.play(FadeIn(credits), run_time=0.6)
        self.wait(0.9)
        self.fade_content(0.40)
        self.sync_end()


ALL_SECTIONS = [Section01, Section02, Section03, Section04, Section05,
                Section06, Section07, Section08, Section09, Section10,
                Section11, Section12]


def main(argv=None):
    """Render through the manim CLI, tolerating our own ``--check_layout``.

    manim's click group rejects unknown options, so the flag is stripped here
    and re-enters the scene as CHECK_LAYOUT=1 (the env var works with a plain
    ``manim render`` invocation too).
    """
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if "--check_layout" in args:
        args.remove("--check_layout")
        os.environ["CHECK_LAYOUT"] = "1"
    from manim.__main__ import main as manim_main
    sys.argv = ["manim", "render", str(Path(__file__).resolve()), *args]
    manim_main()


if __name__ == "__main__":
    main()

