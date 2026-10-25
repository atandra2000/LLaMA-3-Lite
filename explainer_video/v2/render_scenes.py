"""LLaMA-3-Lite architecture explainer — v3 "Aurora" design system.

Premium dark design: near-black background with ambient indigo/teal light
blooms and drifting orbs (built from VMobjects), glass panels and pills,
layered glow accents, bundled Inter display type + Barlow Semi Condensed
technical labels, fluid easing and data-flow flashes.

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
                   Axes, Arrow, Brace, Circle, Circumscribe, Create, CurvedArrow,
                   AnimationGroup, DashedVMobject, Dot, FadeIn, FadeOut, Flash,
                   GrowFromCenter, GrowFromEdge, Indicate,
                   LaggedStart, Line, MathTypst, ORIGIN, Rectangle,
                   ReplacementTransform, Rotate,
                   RoundedRectangle, Scene, ShowPassingFlash, Square, Text,
                   Transform, ValueTracker, VGroup, Wait, Write, always_redraw,
                   config, interpolate_color, linear)

# ------------------------------------------------------------ aurora tokens -
WHITE = "#F4F6FB"
MUTED = "#9AA3BC"
DIM = "#5E6680"
ACCENT = "#8FA0FF"       # iris — chrome & highlights
BLUE = "#4FA3FF"         # queries
GOLD = "#FFB845"         # keys / scale
MINT = "#2FDD9A"         # writes / outputs
VIOLET = "#A07BFF"       # RoPE
ROSE = "#FF6B95"         # output / future-mask
RED = "#FF4D6B"

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
        text = "\u2003".join("\u2009".join(w) for w in text.split())
    return Text(text, font=DIN, font_size=pt(size), color=color, weight="BOLD",
                line_spacing=0.9)


def TY(s, size=TYPE["formula"], color=WHITE, **kw):
    return MathTypst(s, font_size=pt(size), color=color, **kw)


def lighter(c, f=0.35):
    from manim import ManimColor
    return interpolate_color(ManimColor(c), ManimColor("#FFFFFF"), f)


# ---------------------------------------------------------- surfaces & glow -
def glass(fill_op=0.05, stroke_op=0.18, radius=0.14):
    return {"stroke_color": MUTED, "stroke_opacity": stroke_op,
            "stroke_width": 1.2, "fill_color": "#0D111A",
            "fill_opacity": 0.85, "corner_radius": radius}


def card(w, h, edge=None, shadow=True, fill_op=0.05, radius=0.14):
    g = VGroup()
    if shadow:
        g.add(RoundedRectangle(corner_radius=radius, width=w, height=h,
                               stroke_width=0, fill_color=BLACK,
                               fill_opacity=0.0001).shift(DOWN * 0.07 + RIGHT * 0.04))
    box = RoundedRectangle(width=w, height=h, **glass(fill_op, 0.18, radius))
    g.add(box)
    if edge:
        g.add(RoundedRectangle(corner_radius=0.012, width=w - 0.44, height=0.032,
                               stroke_width=0, fill_color=edge,
                               fill_opacity=0.85).move_to(box.get_top() + DOWN * 0.05))
    return g


def pill(label, color=WHITE, size=TYPE["body"], stroke=None, weight="MEDIUM"):
    txt = T(label, size=size, color=color, weight=weight)
    box = RoundedRectangle(corner_radius=(txt.height + 0.32) / 2,
                           width=txt.width + 0.92, height=txt.height + 0.32,
                           stroke_color=stroke or WHITE,
                           stroke_opacity=0.7 if stroke else 0.18,
                           stroke_width=1.4,
                           fill_color=WHITE, fill_opacity=0.05)
    return VGroup(box, txt.move_to(box))


def token_pill(word, id_label=None, color=WHITE, size=20):
    box = RoundedRectangle(corner_radius=0.27, width=len(word) * 0.155 + 0.48,
                           height=0.54, stroke_color=color, stroke_opacity=0.75,
                           stroke_width=1.6, fill_color=WHITE, fill_opacity=0.06)
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
    rings = VGroup()
    for wdt, op in [(7, 0.06), (4.5, 0.11), (2.6, 0.18)]:
        rings.add(mobj.copy().set_stroke(color, width=wdt, opacity=op)
                  .set_fill(opacity=0))
    return rings


def grid(rows, cols, cell=0.24, gap=0.035):
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(side_length=cell, stroke_color=WHITE,
                        stroke_opacity=0.10, stroke_width=1.0,
                        fill_color=WHITE, fill_opacity=0.03)
            sq.move_to(np.array([c * (cell + gap), -r * (cell + gap), 0]))
            cells.add(sq)
    return cells


def grad(mobj, c, to_white=0.4):
    return mobj.set_color_by_gradient(c, lighter(c, to_white))


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
        grad(val, col, 0.45 if p == pmax else 0.25)
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
        # NOTE: built from VMobjects — manim's cairo camera cannot composite
        # ImageMobjects (2-D zeros array pasted with mode="RGBA" raises).
        fw, fh = config.frame_width, config.frame_height
        base = Rectangle(width=fw + 0.1, height=fh + 0.1, stroke_width=0,
                         fill_color="#0B0E14", fill_opacity=1)

        def bloom(center, radius, color, max_op, steps=30):
            g = VGroup()
            for i in range(steps, 0, -1):
                f = i / steps
                op = 0.0001
                c = Circle(radius=radius * f, stroke_width=0,
                           fill_color=color, fill_opacity=op)
                c._base_op = op          # for the ambient breathe
                g.add(c.move_to(center))
            return g

        blooms = VGroup(
            bloom([-3.5, 2.0, 0], 4.4, "#2E3C90", 0.0001),
            bloom([5.0, -2.4, 0], 4.8, "#11536A", 0.0001),
            bloom([0.6, 0.2, 0], 5.6, "#241F52", 0.0001),
        )
        self._blooms = blooms
        self._bloom_home = blooms.get_center().copy()
        dots = VGroup()
        yy = -fh / 2 + 0.5
        while yy < fh / 2:
            xx = -fw / 2 + 0.45
            while xx < fw / 2:
                dots.add(Dot([xx, yy, 0], radius=0.008, color=WHITE)
                         .set_opacity(0.0001))
                xx += 1.05
            yy += 1.05
        orb_a = VGroup()
        for mult, op in [(1.0, 0.0001), (0.72, 0.0001), (0.48, 0.0001)]:
            orb_a.add(Circle(radius=1.35 * mult, stroke_width=0,
                             fill_color=self.ACCENT,
                             fill_opacity=op))
        orb_b = VGroup()
        for mult, op in [(1.0, 0.0001), (0.7, 0.0001)]:
            orb_b.add(Circle(radius=1.8 * mult, stroke_width=0,
                             fill_color="#2A3E6E", fill_opacity=op))
        base_a = np.array([-4.6, 2.4, 0])
        base_b = np.array([4.9, -2.1, 0])
        orb_a.move_to(base_a)
        orb_b.move_to(base_b)
        self._ambient = VGroup(base, blooms, dots, orb_a, orb_b)
        self.add(self._ambient)
        self._orb_updaters = [
            (orb_a, base_a, 0.42, 0.78),
            (orb_b, base_b, 0.51, 0.92),
        ]

    def _start_orb_drift(self):
        """Ambient pulse: the orbs drift and the blooms breathe, always.

        One group updater advances the ambient clock, so the background never
        holds a still frame — that is what keeps a mid-clause wait from reading
        as a freeze. Call once per scene, before the first cue.
        """
        def breathe(m, dt):
            t = self._ft            # the same clock the playhead uses
            for orb, base, speed, amp in self._orb_updaters:
                orb.move_to(base + np.array([amp * np.sin(speed * t),
                                             amp * 0.7 * np.cos(speed * 0.83 * t),
                                             0]))
            # The whole bloom field drifts and breathes. A static gradient with
            # a barely-changing alpha reads as a freeze to freezedetect; moving
            # the field is what keeps a mid-clause wait alive on screen.
            self._blooms.move_to(self._bloom_home + np.array(
                [0.34 * np.sin(0.21 * t), 0.22 * np.cos(0.17 * t), 0]))
            for i, bloom in enumerate(self._blooms):
                k = 0.42 + 0.58 * (0.5 + 0.5 * np.sin(0.42 * t + i * 1.9))
                for circle in bloom:
                    circle.set_fill(opacity=circle._base_op * k)
        self._ambient.add_updater(breathe)

    # chrome ---------------------------------------------------------------
    def _build_chrome(self):
        self._chrome = VGroup()
        tag = VGroup(
            Dot(radius=0.035, color=self.ACCENT),
            DIN_T(self.CHAPTER, size=13, color=MUTED))
        tag.arrange(RIGHT, buff=0.16).to_corner(UL, buff=0.42)
        tag.shift(DOWN * 0.06)
        fw = config.frame_width
        pw = fw - 0.9
        py = -3.85
        pr = 0.0175
        track = RoundedRectangle(corner_radius=pr, width=pw, height=0.035,
                                 stroke_width=0, fill_color=WHITE,
                                 fill_opacity=0.0001).move_to([0, py, 0])
        done_w = max(pw * STARTS[self.SEC] / TOTAL, 1e-3)
        cur_w = pw * self.budget / TOTAL
        done = RoundedRectangle(corner_radius=pr, width=done_w, height=0.035,
                                stroke_width=0, fill_color=WHITE,
                                fill_opacity=0.0001)
        done.move_to([-pw / 2 + done_w / 2, py, 0])
        cur = RoundedRectangle(corner_radius=pr, width=cur_w, height=0.035,
                               stroke_width=0, fill_color=self.ACCENT,
                               fill_opacity=0.0001)
        cur.move_to([-pw / 2 + done_w + cur_w / 2, py, 0])
        ticks = VGroup()
        for k in ORDER:   # one tick per chapter boundary
            x = -pw / 2 + pw * STARTS[k] / TOTAL
            ticks.add(Rectangle(width=0.014, height=0.10, stroke_width=0,
                                fill_color=WHITE, fill_opacity=0.0001)
                      .move_to([x, py, 0]))
        # Clause playhead: slides across this scene's slice of the track and
        # flares at every cue. It is the moving highlight for the spoken
        # clause, and it is what keeps a wait from reading as a still frame.
        head = RoundedRectangle(corner_radius=0.02, width=0.09, height=0.30,
                                stroke_width=0, fill_color=self.ACCENT,
                                fill_opacity=0.0001)
        halo = RoundedRectangle(corner_radius=0.06, width=0.34, height=0.46,
                                stroke_width=0, fill_color=self.ACCENT,
                                fill_opacity=0.0001)
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
            halo.set_fill(opacity=0.0001)
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
        self._start_orb_drift()
        self.cue("s01.1")
        self.section_header("ARCHITECTURE DEEP-DIVE", "LLaMA-3-Lite",
                            "A decoder-only transformer — the complete architecture",
                            run_time=1.3, ty=2.3, size=58)

        demo_c = np.array([0, 0.25, 0])
        chips = VGroup(token_pill("t₁"), token_pill("t₂"), token_pill("t₃"))
        chips.arrange(RIGHT, buff=0.32).move_to(demo_c + LEFT * 3.4)
        arrow = dim_arrow(chips.get_right() + RIGHT * 0.05, demo_c + LEFT * 0.35,
                          op=0.45, w=2.4, tip=0.17, buff=0.1)
        prob = bars(["mat", "floor", "sofa", "…"],
                    [0.72, 0.12, 0.09, 0.07],
                    [GOLD, MUTED, MUTED, MUTED], width=1.9, size=18)
        prob.move_to(demo_c + RIGHT * 2.9).shift(DOWN * 0.5)
        prob_title = DIN_T("NEXT-TOKEN PROBABILITIES", size=TYPE["micro"], color=MUTED)
        prob_title.next_to(prob, UP, buff=0.18).align_to(prob, LEFT)
        self.cue("s01.2")
        self.play(LaggedStart(
            Write(chips[0]), Write(chips[1]), Write(chips[2]),
            Create(arrow), GrowFromEdge(prob, DOWN),
            Write(prob_title),
            lag_ratio=0.14), run_time=2.6)
        cap = T("one function:  tokens → probabilities", size=18, color=MUTED)
        cap.move_to([0, -1.5, 0])
        self.play(Write(cap), run_time=0.8)

        specs = VGroup(*[pill(s, color=WHITE, size=16)
                         for s in spec_chip_labels()])
        specs.arrange(RIGHT, buff=0.30).move_to([0, -2.5, 0])
        self.cue("s01.3")
        self.play(LaggedStart(*[Write(s) for s in specs[:3]], lag_ratio=0.12),
                  run_time=1.2)
        self.cue("s01.4")
        self.play(LaggedStart(*[Write(s) for s in specs[3:]], lag_ratio=0.12),
                  run_time=0.9)

        self.cue("s01.5")
        self.fade_content(0.35)
        stack = VGroup()
        for i in range(N_LAYERS):
            stack.add(RoundedRectangle(corner_radius=0.06, width=1.9,
                                       height=0.155, stroke_color=WHITE,
                                       stroke_opacity=0.13, stroke_width=1.1,
                                       fill_color=WHITE, fill_opacity=0.045))
        stack.arrange(DOWN, buff=0.115).move_to([0, 0.1, 0])
        emb = pill(f"Embedding   {VOCAB_LABEL} → {D_MODEL:,}", color=BLUE, size=16)
        emb.next_to(stack, UP, buff=0.30)
        unemb = pill(f"Unembedding   {D_MODEL:,} → {VOCAB_LABEL}", color=ROSE,
                     size=16)
        unemb.next_to(stack, DOWN, buff=0.30)
        self.play(LaggedStart(Write(emb),
                              *[Create(s) for s in stack],
                              Write(unemb), lag_ratio=0.07),
                  run_time=1.3)
        ring = RoundedRectangle(corner_radius=0.10, width=2.24, height=0.52,
                                stroke_color=GOLD, stroke_width=2.4)
        ring.move_to(stack[0])   # block 1, the one the next scene opens
        halo = VGroup()
        lab = T("we'll open one block", size=18, color=GOLD)
        lab.next_to(ring, RIGHT, buff=0.4)
        self.play(Create(ring), Write(lab), run_time=0.85)
        rem = self.budget - self._t - 0.45
        if rem > 0:
            self.wait(rem)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 2 ========
class Section02(Section):
    SEC = "sec02_tokens"
    CHAPTER = "02 · EMBEDDINGS"
    ACCENT = BLUE

    def construct(self):
        self._start_orb_drift()
        self.cue("s02.1")
        hdr = self.section_header("FROM TEXT TO GEOMETRY", "Text becomes numbers",
                                  run_time=1.1)

        sentence = T(TOKENS["sentence"], size=28, color=WHITE)
        sentence.move_to([0, 1.55, 0])
        items = TOKENS["items"]
        words = [item["text"] for item in items]
        ids = [str(item["id"]) for item in items]
        cols = [BLUE, GOLD, MINT, VIOLET, BLUE, ROSE]
        chips = VGroup()
        for w, i, c in zip(words, ids, cols):
            chips.add(token_pill(w, id_label=i, color=c))
        chips.arrange(RIGHT, buff=0.30).move_to([0, 1.55, 0])
        word_ts = VGroup(*[T(w, size=28, color=WHITE) for w in words])
        word_ts.arrange(RIGHT, buff=0.38).move_to([0, 1.55, 0])
        self.cue("s02.2")
        self.play(Write(sentence), run_time=1.0)
        self.play(ReplacementTransform(sentence, word_ts), run_time=0.6)
        self.play(*[ReplacementTransform(word_ts[i], chips[i])
                    for i in range(6)], run_time=1.1)

        mat = grid(8, 6, cell=0.235, gap=0.035)
        mat.move_to([4.35, -0.1, 0])
        mat_label = TY(f'E in RR^({VOCAB} times {D_MODEL})', size=22, color=MUTED)
        mat_label.next_to(mat, DOWN, buff=0.24)
        self.cue("s02.3")
        self.play(FadeOut(hdr), Create(mat), run_time=1.0)
        self.play(LaggedStart(*[Indicate(mat[6 * i: 6 * i + 6],
                                         scale_factor=1.2, color=cols[i])
                                for i in range(6)], lag_ratio=0.16),
                  run_time=2.2)
        for cellm in mat[6:12]:
            cellm.set_fill(GOLD, opacity=0.4).set_stroke(GOLD, width=1.4,
                                                         opacity=0.9)
        self.cue("s02.4")
        self.play(Write(mat_label), run_time=0.7)

        vec_vals = ["0.42", "-1.13", "0.07", "0.88", "-0.52", "…"]
        vec_cells = VGroup()
        for v, c in zip(vec_vals, cols[:5] + [MUTED]):
            bx = RoundedRectangle(corner_radius=0.08, width=0.62, height=0.5,
                                  stroke_color=WHITE, stroke_opacity=0.14,
                                  stroke_width=1.1, fill_color=WHITE,
                                  fill_opacity=0.05)
            notch = Rectangle(width=0.055, height=0.34, stroke_width=0,
                              fill_color=c, fill_opacity=0.95)
            notch.move_to(bx.get_left() + RIGHT * 0.07)
            vec_cells.add(VGroup(bx, notch,
                                 Text(v, font=DIN, font_size=pt(14), color=WHITE,
                                      weight="BOLD").move_to(bx)))
        vec_cells.arrange(RIGHT, buff=0.10).move_to([-1.5, 0.1, 0])
        vec_label = TY('x = E[t_2]', size=22, color=GOLD)
        vec_label.next_to(vec_cells, UP, buff=0.18)
        self.cue("s02.5")
        self.play(ReplacementTransform(mat[6:12].copy(),
                                       VGroup(*[v[0] for v in vec_cells])),
                  LaggedStart(*[GrowFromCenter(VGroup(v[1], v[2]))
                                for v in vec_cells], lag_ratio=0.08),
                  Write(vec_label), run_time=2.0)

        space = Axes(x_range=[-1, 1, 1], y_range=[-1, 1, 1], x_length=2.6,
                     y_length=2.1, tips=False,
                     axis_config={"stroke_color": MUTED, "stroke_opacity": 0.35,
                                  "stroke_width": 1.2})
        space.move_to([-3.7, -1.80, 0])
        pts = {"cat": (-0.35, 0.15, BLUE), "kitten": (-0.05, 0.45, BLUE),
               "dog": (0.25, -0.05, BLUE), "car": (0.45, 0.55, ROSE),
               "truck": (0.7, 0.3, ROSE), "apple": (-0.5, -0.55, MINT)}
        dots = VGroup()
        for name, (x, y, c) in pts.items():
            d = Dot(space.c2p(x, y), radius=0.07, color=c)
            lab = T(name, size=13, color=lighter(c, 0.2)).next_to(d, DOWN, buff=0.06)
            dots.add(VGroup(d, lab))
        space_cap = DIN_T(f"{D_MODEL:,}-D MEANING SPACE — 2-D SKETCH", size=TYPE["micro"],
                          color=MUTED)
        space_cap.next_to(space, DOWN, buff=0.16)
        self.play(Create(space),
                  LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.14),
                  Write(space_cap), run_time=2.4)

        fx = TY('x_i = E[t_i]', size=27, color=WHITE).move_to([1.7, -1.85, 0])
        note = T("the model only sees vectors from here on", size=13,
                 color=MUTED).move_to([1.7, -2.62, 0])
        self.cue("s02.6")
        self.play(Write(fx), Write(note), run_time=1.2)
        self.cue("s02.7")
        self.play(Indicate(vec_cells, scale_factor=1.12, color=GOLD),
                  run_time=0.5)
        rem = self.budget - self._t - 0.45
        if rem > 0:
            self.wait(rem)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 3 ========
class Section03(Section):
    SEC = "sec03_residual"
    CHAPTER = "03 · RESIDUAL STREAM"
    ACCENT = MINT

    def construct(self):
        self._start_orb_drift()
        self.cue("s03.1")
        hdr = self.section_header("THE BACKBONE", "The residual stream", run_time=1.0)

        band = card(10.6, 1.05, fill_op=0.045, radius=0.2)
        band.move_to([0, 0.3, 0])
        band_box = band[1]
        flow_lines = VGroup()
        for dy in (-0.28, 0.0, 0.28):
            ln = Line(band_box.get_left() + RIGHT * 0.3,
                      band_box.get_right() + LEFT * 0.3,
                      stroke_color=WHITE, stroke_opacity=0.08,
                      stroke_width=1.0)
            ln.shift(UP * dy)
            flow_lines.add(ln)
        band_lab = DIN_T("THE RESIDUAL STREAM", size=TYPE["micro"], color=MUTED)
        band_lab.next_to(band, DOWN, buff=0.18).align_to(band, LEFT).shift(RIGHT * 0.15)
        toks = VGroup(token_pill("The", color=BLUE), token_pill("cat", color=GOLD),
                      token_pill("sat", color=MINT))
        toks.arrange(RIGHT, buff=0.55).move_to(band.get_center() + LEFT * 2.6)
        self.play(Create(band), Create(flow_lines),
                  LaggedStart(*[Write(t) for t in toks], lag_ratio=0.2),
                  Write(band_lab), run_time=1.9)

        att = card(3.0, 0.78, edge=BLUE)
        att.move_to([-2.6, 2.0, 0])
        att_lab = T("Attention sublayer", size=17, color=lighter(BLUE, 0.25))
        att_lab.move_to(att)
        ffn = card(3.3, 0.78, edge=VIOLET)
        ffn.move_to([2.6, -1.4, 0])
        ffn_lab = T("Feed-forward sublayer", size=17, color=lighter(VIOLET, 0.25))
        ffn_lab.move_to(ffn)
        read1 = dim_arrow(band_box.get_top() + LEFT * 2.6, att.get_bottom(),
                          op=0.35)
        write1 = data_arrow(att.get_right() + RIGHT * 0.3,
                            band_box.get_top() + RIGHT * 1.6, MINT, w=2.6)
        plus1 = VGroup(Circle(radius=0.18, stroke_color=MINT, stroke_width=1.5,
                              fill_color="#0B0E14", fill_opacity=1.0).move_to(write1.get_end()),
                       T("+", size=18, color=MINT, weight="BOLD").move_to(write1.get_end()))
        read2 = dim_arrow(band_box.get_bottom() + RIGHT * 2.6, ffn.get_top(),
                          op=0.35)
        write2 = data_arrow(ffn.get_left() + LEFT * 0.3,
                            band_box.get_bottom() + LEFT * 1.6, MINT, w=2.6)
        plus2 = VGroup(Circle(radius=0.18, stroke_color=MINT, stroke_width=1.5,
                              fill_color="#0B0E14", fill_opacity=1.0).move_to(write2.get_end()),
                       T("+", size=18, color=MINT, weight="BOLD").move_to(write2.get_end()))
        self.cue("s03.2")
        self.play(FadeOut(hdr), Create(att), Write(att_lab), Create(read1),
                  Create(write1), GrowFromCenter(plus1),
                  Create(ffn), Write(ffn_lab), Create(read2),
                  Create(write2), GrowFromCenter(plus2),
                  run_time=2.3)

        eq_card = card(4.4, 1.05, edge=MINT)
        eq_card.move_to([-2.4, -2.55, 0])
        eq = TY('x_(t+1) = x_t + f(x_t)', size=29, color=WHITE).move_to(eq_card)
        eq_cap = T("read  →  transform  →  add back", size=16, color=MUTED)
        eq_cap.next_to(eq_card, DOWN, buff=0.16)
        self.cue("s03.3")
        self.play(Create(eq_card), Write(eq), Write(eq_cap), run_time=1.5)

        x16 = pill(f"× {N_LAYERS} blocks", color=WHITE, size=16)
        # Off the band: at y = 0.3 the pill straddled the band's right edge.
        x16.move_to([5.5, 1.25, 0])
        self.cue("s03.4")
        self.play(Write(x16),
                  toks.animate.shift(RIGHT * 2.3), run_time=2.0)
        self.play(flow(flow_lines[1], MINT), run_time=0.6)
        rem = self.budget - self._t - 0.45
        if rem > 0:
            self.wait(rem)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 4 ========
class Section04(Section):
    SEC = "sec04_rmsnorm"
    CHAPTER = "04 · RMSNORM"
    ACCENT = GOLD

    def construct(self):
        self._start_orb_drift()
        self.cue("s04.1")
        hdr = self.section_header("SCALE CONTROL", "RMSNorm — regulating scale",
                                  run_time=1.1)

        y_row = 0.20
        vals = ["2.10", "-0.83", "1.47", "0.62", "-1.05", "0.38"]

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

        xcol = VGroup(*[num_cell(v, WHITE) for v in vals])
        xcol.arrange(DOWN, buff=0.10).move_to([-5.0, y_row, 0])
        xlab = TY(f'x in RR^{D_MODEL}', size=20, color=MUTED).next_to(xcol, UP, buff=0.16)
        xnote = DIN_T(f"SHOWING 6 OF {D_MODEL:,}", size=TYPE["micro"], color=DIM)
        xnote.next_to(xcol, DOWN, buff=0.14)
        self.cue("s04.2")
        self.play(FadeOut(hdr), Write(xlab),
                  LaggedStart(*[Write(v) for v in xcol], lag_ratio=0.1),
                  Write(xnote), run_time=1.6)

        sqcol = VGroup(*[num_cell(v, GOLD, lighter(GOLD, 0.15))
                         for v in ["4.41", "0.69", "2.16", "0.38", "1.10", "0.14"]])
        sqcol.arrange(DOWN, buff=0.10).move_to([-3.3, y_row, 0])
        sqlab = T("x²", size=17, color=GOLD, weight="BOLD").next_to(sqcol, UP, buff=0.16)
        sqarrows = VGroup(*[dim_arrow(xcol[i].get_right(), sqcol[i].get_left(),
                                      GOLD, 0.6, 1.6, 0.11, 0.04)
                            for i in range(6)])
        self.play(LaggedStart(*[Create(a) for a in sqarrows] +
                              [ReplacementTransform(xcol[i].copy(), sqcol[i])
                               for i in range(6)], lag_ratio=0.08),
                  Write(sqlab), run_time=1.8)

        mean_chip = pill("mean  1.48", color=GOLD, size=16)
        mean_chip.move_to([-1.4, y_row + 0.55, 0])
        sq2mean = data_arrow(sqcol.get_right(), mean_chip.get_left(), GOLD,
                             w=2.0)
        rms_chip = pill("RMS  1.22", color=WHITE, size=16, stroke=WHITE)
        # A 0.55 gap leaves room for the √ operator to sit on the shaft.
        rms_chip.next_to(mean_chip, DOWN, buff=0.55)
        mean2rms = dim_arrow(mean_chip.get_bottom(), rms_chip.get_top(),
                             op=0.4)
        root = operator_on_arrow("√", mean2rms, size=24)
        self.cue("s04.3")
        self.play(Create(sq2mean), Write(mean_chip), run_time=1.0)
        self.play(Create(mean2rms), Write(root),
                  ReplacementTransform(mean_chip, rms_chip), run_time=1.0)

        ncol = VGroup(*[num_cell(v, MINT, lighter(MINT, 0.2))
                        for v in ["1.72", "-0.68", "1.20", "0.51", "-0.86", "0.31"]])
        ncol.arrange(DOWN, buff=0.10).move_to([0.7, y_row, 0])
        nlab = TY('hat(x) = x / "RMS"(x)', size=21, color=MINT)
        nlab.next_to(ncol, UP, buff=0.16)
        rms2n = data_arrow(rms_chip.get_right(), ncol.get_left() + DOWN * 0.4,
                           MINT, w=2.0)
        gchip = pill("· γ  (learned)", color=MINT, size=15)
        gchip.next_to(ncol, RIGHT, buff=0.45)
        self.cue("s04.4")
        self.play(Create(rms2n),
                  LaggedStart(*[Write(v) for v in ncol], lag_ratio=0.08),
                  Write(nlab), run_time=1.8)
        self.play(Write(gchip), run_time=0.8)

        f_card = card(5.6, 1.15, edge=GOLD)
        f_card.move_to([0.1, -1.92, 0])
        f = TY('y = gamma dot x / sqrt(1/d sum_i x_i^2 + epsilon)',
               size=31, color=WHITE).move_to(f_card)
        # The LayerNorm / RMSNorm / ε chips sit right of the formula card and
        # clear of the bottom margin; ε used to fall off the frame.
        ln = pill("LayerNorm: subtract the mean", color=MUTED, size=14)
        ln.align_to([3.05, -1.55, 0], LEFT).set_y(-1.55)
        strike = Line(ln[0].get_left() + RIGHT * 0.2,
                      ln[0].get_right() - RIGHT * 0.2,
                      stroke_color=ROSE, stroke_width=2.6)
        rmsn = pill("RMSNorm: no mean subtraction", color=MINT, size=14)
        rmsn.next_to(ln, DOWN, buff=0.26)
        eps = pill(f"ε = {EPS_LABEL}", color=WHITE, size=14, stroke=WHITE)
        eps.next_to(rmsn, DOWN, buff=0.26)
        self.cue("s04.5")
        # Circumscribing the RMS chip ties the formula to the quantity it means;
        # the old target (mean_chip) no longer existed by this beat.
        self.play(Create(f_card), Write(f),
                  Circumscribe(rms_chip, color=GOLD, buff=0.12), run_time=1.4)
        self.play(Write(ln), run_time=0.5)
        self.play(Create(strike), run_time=0.3)
        self.play(Write(rmsn), Write(eps), run_time=0.6)

        geo = Axes(x_range=[-1.3, 1.3, 1], y_range=[-1.3, 1.3, 1],
                   x_length=GEO_SIZE, y_length=GEO_SIZE, tips=False,
                   axis_config={"stroke_color": WHITE, "stroke_opacity": 0.14,
                                "stroke_width": 1.0})
        geo.move_to([-5.0, -2.5, 0])
        circ = DashedVMobject(Circle(radius=GEO_SIZE / 2.6, color=WHITE,
                                    stroke_width=1.2)
                              .move_to(geo.get_center()), num_dashes=28)
        circ.set_stroke(opacity=0.25)
        big = dim_arrow(geo.c2p(0, 0), geo.c2p(0.62, 0.62), GOLD, 1.0, 3.2,
                        0.15, 0)
        big_halo = glow_ring(big, GOLD)
        small_target = Arrow(geo.c2p(0, 0), geo.c2p(0.36, 0.36),
                             stroke_color=MINT, fill_color=MINT,
                             stroke_width=3.2, tip_length=0.15, buff=0)
        geo_cap = DIN_T("SAME DIRECTION · UNIT SCALE", size=TYPE["micro"], color=MUTED)
        geo_cap.next_to(geo, DOWN, buff=0.10)
        self.cue("s04.6")
        self.play(Create(geo), Create(circ), FadeIn(big, big_halo), run_time=1.3)
        self.play(Transform(big, small_target),
                  Transform(big_halo, glow_ring(small_target, MINT)),
                  run_time=1.1)
        self.play(Write(geo_cap), run_time=0.5)

        rem = self.budget - self._t - 0.45
        if rem > 0:
            self.wait(rem)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 5 ========
class Section05(Section):
    SEC = "sec05_attention"
    CHAPTER = "05 · SELF-ATTENTION"
    ACCENT = BLUE

    def construct(self):
        self._start_orb_drift()
        self.cue("s05.1")
        hdr = self.section_header("THE CORE COMPUTATION",
                                  "Self-attention — the core computation",
                                  run_time=0.9, size=36)

        words = ["The", "cat", "sat", "on"]
        toks = VGroup(*[T(w, size=21, color=WHITE) for w in words]).arrange(RIGHT, buff=0.95)
        toks.move_to([0, -2.4, 0])
        self.play(LaggedStart(*[Write(t) for t in toks], lag_ratio=0.12),
                  run_time=1.0)
        q_ring = RoundedRectangle(corner_radius=0.10, width=toks[2].width + 0.36,
                                  height=toks[2].height + 0.22, stroke_color=BLUE,
                                  stroke_width=2.4, fill_color=BLUE,
                                  fill_opacity=0.15).move_to(toks[2])
        self.play(Create(q_ring), run_time=0.4)

        proj_y = -1.15
        qdot = VGroup(Circle(radius=0.16, stroke_color=BLUE, stroke_width=1.5,
                             fill_color=BLUE, fill_opacity=0.15).move_to([-1.9, proj_y, 0]),
                      Dot([-1.9, proj_y, 0], radius=0.08, color=BLUE))
        kdot = VGroup(Circle(radius=0.16, stroke_color=GOLD, stroke_width=1.5,
                             fill_color=GOLD, fill_opacity=0.15).move_to([0.4, proj_y, 0]),
                      Dot([0.4, proj_y, 0], radius=0.08, color=GOLD))
        vdot = VGroup(Circle(radius=0.16, stroke_color=MINT, stroke_width=1.5,
                             fill_color=MINT, fill_opacity=0.15).move_to([2.7, proj_y, 0]),
                      Dot([2.7, proj_y, 0], radius=0.08, color=MINT))
        # Above the dots, not below them: below, the q→q arrow shaft ran
        # through the k label on its way across. Anchor on the core ([-1]), not
        # the glow VGroup: its halo is ~1 unit wide and lifts the label into
        # the weight-bar caption band, which is what collided with wcap.
        qlab = TY('q = x W_Q', size=19, color=lighter(BLUE, 0.2)).next_to(qdot[-1], UP, buff=0.14)
        klab = TY('k = x W_K', size=19, color=lighter(GOLD, 0.2)).next_to(kdot[-1], UP, buff=0.14)
        vlab = TY('v = x W_V', size=19, color=lighter(MINT, 0.2)).next_to(vdot[-1], UP, buff=0.14)
        aq = data_arrow(toks[2].get_top(), qdot[-1].get_center(), BLUE)
        ak = data_arrow(toks[2].get_top(), kdot[-1].get_center(), GOLD)
        av = data_arrow(toks[2].get_top(), vdot[-1].get_center(), MINT)
        self.cue("s05.2")
        self.play(Create(aq), GrowFromCenter(qdot), Write(qlab),
                  Create(ak), GrowFromCenter(kdot), Write(klab),
                  Create(av), GrowFromCenter(vdot), Write(vlab),
                  run_time=2.2)
        kvs = VGroup()
        for i in [0, 1, 3]:
            kd = Dot(radius=0.08, color=GOLD).set_fill(GOLD, 0.9)
            kd.move_to(toks[i].get_center() + UP * 0.48 + LEFT * 0.16)
            vd = Dot(radius=0.08, color=MINT).set_fill(MINT, 0.9)
            vd.move_to(toks[i].get_center() + UP * 0.48 + RIGHT * 0.16)
            kvs.add(VGroup(kd, vd))
        self.play(LaggedStart(*[GrowFromCenter(kv) for kv in kvs], lag_ratio=0.2),
                  run_time=1.1)

        raw, scaled, _ = attention_demo()
        qrow = ATTN["query_row"] - 1

        mat = grid(4, 4, cell=0.48, gap=0.05)
        mat.move_to([-4.45, 0.35, 0])
        mb_w = 4 * 0.48 + 3 * 0.05 + 0.16
        mb_h = 4 * 0.48 + 3 * 0.05 + 0.16
        mc = mat.get_center()
        bl_x, br_x = mc[0] - mb_w / 2, mc[0] + mb_w / 2
        bt_y, bb_y = mc[1] + mb_h / 2, mc[1] - mb_h / 2
        b_tick = 0.16
        mat_brackets = VGroup(
            Line([bl_x + b_tick, bt_y, 0], [bl_x, bt_y, 0], stroke_width=1.8, stroke_color=MUTED),
            Line([bl_x, bt_y, 0], [bl_x, bb_y, 0], stroke_width=1.8, stroke_color=MUTED),
            Line([bl_x, bb_y, 0], [bl_x + b_tick, bb_y, 0], stroke_width=1.8, stroke_color=MUTED),
            Line([br_x - b_tick, bt_y, 0], [br_x, bt_y, 0], stroke_width=1.8, stroke_color=MUTED),
            Line([br_x, bt_y, 0], [br_x, bb_y, 0], stroke_width=1.8, stroke_color=MUTED),
            Line([br_x, bb_y, 0], [br_x - b_tick, bb_y, 0], stroke_width=1.8, stroke_color=MUTED),
        )
        matcap = TY('"score"_(i j) = q_i dot k_j', size=18, color=MUTED)
        matcap.next_to(mat, DOWN, buff=0.18)
        rowlabs = VGroup(*[T(f"q{sub(i + 1)}", size=12, color=MUTED,
                             weight="BOLD").next_to(mat[4 * i], LEFT, buff=0.15)
                           for i in range(4)])
        collabs = VGroup(*[T(f"k{sub(j + 1)}", size=12, color=MUTED,
                             weight="BOLD").next_to(mat[j], UP, buff=0.12)
                           for j in range(4)])
        # The weight bars below are row q₃, so mark that row as the one being read.
        rowlabs[qrow].set_color(GOLD)
        row_hl = RoundedRectangle(corner_radius=0.09,
                                  width=4 * 0.48 + 3 * 0.05 + 0.16, height=0.64,
                                  stroke_color=GOLD, stroke_width=1.8,
                                  fill_color=GOLD, fill_opacity=0.06)
        row_hl.move_to(VGroup(*mat[4 * qrow:4 * qrow + 4]).get_center())
        self.cue("s05.3")
        self.play(FadeOut(hdr), run_time=0.5)
        self.play(Create(mat), Create(mat_brackets), Create(row_hl),
                  FadeIn(rowlabs), FadeIn(collabs), Write(matcap), run_time=1.6)

        # 1. raw scores, straight from facts.json
        snums = VGroup()
        for r in range(4):
            for c in range(4):
                tmon = Text(f"{raw[r][c]:.1f}", font=DIN, font_size=pt(13),
                            color=WHITE, weight="BOLD")
                tmon.move_to(mat[4 * r + c])
                snums.add(tmon)
        self.play(LaggedStart(*[Write(m) for m in snums], lag_ratio=0.04),
                  run_time=2.0)

        # 2. the ÷√d_k strip, then the same cells rescaled in place
        scale = VGroup(
            T("÷ √d_k", size=20, color=WHITE, weight="BOLD"),
            TY(f"d_k = {HEAD_DIM}  arrow.r  times {ATTN['scale']:.3f}", size=13,
               color=MUTED),
        ).arrange(DOWN, buff=0.10)
        scale.move_to([-1.90, 0.45, 0])
        scaled_cap = TY('s_(i j) = q_i dot k_j / sqrt(d_k)', size=18, color=MUTED)
        scaled_cap.move_to(matcap)
        self.cue("s05.4")
        self.play(Write(scale), Transform(matcap, scaled_cap),
                  *[Transform(snums[4 * r + c],
                              Text(f"{scaled[r][c]:.2f}", font=DIN,
                                   font_size=pt(13), color=WHITE,
                                   weight="BOLD").move_to(mat[4 * r + c]))
                    for r in range(4) for c in range(4)], run_time=1.4)

        # 3. causal mask on the scaled scores, before the softmax
        hatches = VGroup()
        for r in range(4):
            for c in range(r + 1, 4):
                cell = mat[4 * r + c]
                cell.set_fill(ROSE, opacity=0.10)
                cell.set_stroke(ROSE, width=1.2, opacity=0.5)
                snums[4 * r + c].set_color("#8A6572")
                sqd = cell.get_center()
                hatches.add(Line(sqd + UL * 0.17, sqd + DR * 0.17,
                                 stroke_color=ROSE, stroke_width=1.1,
                                 stroke_opacity=0.35),
                            Line(sqd + UR * 0.17, sqd + DL * 0.17,
                                 stroke_color=ROSE, stroke_width=1.1,
                                 stroke_opacity=0.35))
        lock = T("future hidden", size=14, color=ROSE)
        lock.next_to(matcap, DOWN, buff=0.14).align_to(mat, RIGHT)
        self.cue("s05.5")
        self.play(LaggedStart(*[Create(h) for h in hatches], lag_ratio=0.04),
                  Write(lock), run_time=1.8)

        # 4. masked softmax → the weights the bars show; the masked key is 0.00
        wlabels, wprobs = attention_weight_bars()
        wbars = bars(wlabels, wprobs, [BLUE, GOLD, MINT, VIOLET],
                     width=1.9, size=14)
        wbars.move_to([2.45, 1.35, 0])
        # Below the bars, not above them: above, the caption and then the
        # closing banner both landed on the same band at y ≈ 2. One line keeps
        # the caption above the weighted-sum arrows that cross this area.
        wcap = TY('w_(3 j) = "softmax"(s_(3 j) + M)', size=17, color=MUTED)
        wcap.next_to(wbars, DOWN, buff=0.22)
        zero_note = T("masked key → exactly 0.00", size=13, color=ROSE)
        zero_note.next_to(wcap, DOWN, buff=0.10)
        self.cue("s05.6")
        self.play(GrowFromEdge(wbars, DOWN), Write(wcap), Write(zero_note),
                  run_time=2.0)

        obox = Arrow([4.55, 1.35, 0], [5.65, 1.35, 0], stroke_color=MINT, fill_color=MINT,
                     stroke_width=3.6, tip_length=0.18, buff=0)
        obox_halo = VGroup()
        otxt = T("o₃", size=24, color=MINT, weight="BOLD").next_to(obox, RIGHT, buff=0.14)
        olab = TY('o_i = sum_j w_(i j) v_j', size=17, color=MINT)
        olab.next_to(obox, DOWN, buff=0.24).align_to(obox, LEFT)
        wsum_arrows = VGroup()
        for i in range(4):
            src = wbars[i][1].get_right() + RIGHT * 0.10
            wsum_arrows.add(data_arrow(src, obox.get_start() + LEFT * 0.12, MINT,
                                       w=1.6, op=0.75))
        self.cue("s05.7")
        # The arrows and the box they land on arrive together: an arrow fan
        # pointing at empty space reads as a mistake for a whole clause.
        self.play(LaggedStart(*[Create(a) for a in wsum_arrows], lag_ratio=0.12),
                  Create(obox), FadeIn(obox_halo), Write(otxt), run_time=1.4)
        self.cue("s05.8")
        self.play(Write(olab), run_time=0.6)
        self.play(flow(wsum_arrows[1], MINT), run_time=0.9)

        hcols = [BLUE, GOLD, MINT, VIOLET] * (N_HEADS // 4)
        heads = VGroup(*[head_badge(f"h{i+1}", hcols[i], r=0.18)
                         for i in range(N_HEADS)])
        heads.arrange(RIGHT, buff=0.30).move_to([-0.7, -2.93, 0])
        hlab = pill(f"× {N_HEADS} heads", color=WHITE, size=13)
        hlab.next_to(heads, LEFT, buff=0.35)
        # Two lines under the heads: one line to the right ran into the frame
        # edge and put the W_O caption flush against it.
        hcap = T("independent projections → concat\n→ W_O", size=13,
                 color=MUTED)
        hcap.next_to(heads, DOWN, buff=0.05)
        self.cue("s05.9")
        self.play(LaggedStart(*[Write(h) for h in heads], lag_ratio=0.07),
                  Write(hlab), Write(hcap), run_time=1.8)

        banner = TY('"Attention"(Q, K, V) = "softmax"((Q K^T)/sqrt(d_k) + M) V',
                    size=26, color=WHITE)
        banner_card = banner
        banner_card.move_to([0, 2.45, 0])
        # The header has done its job and would collide with the banner.
        self.cue("s05.10")
        to_clear = [mat, mat_brackets, row_hl, snums, matcap, scaled_cap, hatches, lock,
                    scale, wbars, wcap, zero_note, obox, otxt, olab, wsum_arrows,
                    rowlabs, collabs, toks, q_ring, qdot, kdot, vdot, qlab, klab,
                    vlab, aq, ak, av, kvs]
        self.play(*[FadeOut(m) for m in to_clear], Write(banner), run_time=1.6)
        self.wait(1.0)
        self.fade_content(0.6)
        self.sync_end()


# =========================================================== scene 6 ========
class Section06(Section):
    SEC = "sec06_rope"
    CHAPTER = "06 · ROTARY POSITIONS"
    ACCENT = VIOLET

    def construct(self):
        self._start_orb_drift()
        self.cue("s06.1")
        hdr = self.section_header("POSITION, ROTATING", "Rotary positions — RoPE",
                                  run_time=1.0)

        row1 = VGroup(T("the", size=22, color=BLUE), T("cat", size=22, color=GOLD),
                      T("sat", size=22, color=MINT))
        row1.arrange(RIGHT, buff=0.35).move_to([-3.3, 1.7, 0])
        b1_l = T("[", size=26, color=MUTED).next_to(row1, LEFT, buff=0.15)
        b1_r = T("]", size=26, color=MUTED).next_to(row1, RIGHT, buff=0.15)
        seq1 = VGroup(b1_l, row1, b1_r)

        row2 = VGroup(T("sat", size=22, color=MINT), T("cat", size=22, color=GOLD),
                      T("the", size=22, color=BLUE))
        row2.arrange(RIGHT, buff=0.35).move_to([-3.3, 0.9, 0])
        b2_l = T("[", size=26, color=MUTED).next_to(row2, LEFT, buff=0.15)
        b2_r = T("]", size=26, color=MUTED).next_to(row2, RIGHT, buff=0.15)
        seq2 = VGroup(b2_l, row2, b2_r)

        eqt = TY('q_i dot k_j = "const"  ("order-blind")', size=16, color=MUTED)
        eqt.next_to(seq2, DOWN, buff=0.18)
        self.play(Write(seq1), Write(eqt), run_time=1.3)
        self.play(ReplacementTransform(seq1, seq2), run_time=0.9)

        cc = np.array([-3.3, -0.6, 0])
        circ = Circle(radius=0.95, stroke_color=WHITE, stroke_opacity=0.35,
                      stroke_width=1.6).move_to(cc)
        c_axes = VGroup(
            Line(cc + LEFT * 1.20, cc + RIGHT * 1.20, stroke_color=MUTED, stroke_opacity=0.45, stroke_width=1.4),
            Line(cc + DOWN * 1.20, cc + UP * 1.20, stroke_color=MUTED, stroke_opacity=0.45, stroke_width=1.4),
        )
        vec = Arrow(cc, cc + RIGHT * 0.95, stroke_color=VIOLET,
                    fill_color=VIOLET, stroke_width=3.6, tip_length=0.18, buff=0)
        mtrack = ValueTracker(0)
        mlab = always_redraw(
            lambda: TY(f"m = {mtrack.get_value():.0f}", size=20, color=VIOLET)
            .next_to(circ, DOWN, buff=0.20))
        self.cue("s06.2")
        self.play(FadeOut(hdr), FadeOut(seq2), FadeOut(eqt),
                  Create(circ), Create(c_axes), FadeIn(vec),
                  run_time=1.3)
        self.add(mlab)
        # m sweeps 0 → 7 in one continuous rotation at a constant rate; v2
        # stepped three times and left the arrow frozen at m = 3.
        self.cue("s06.3")
        self.play(Rotate(vec, angle=-DEMO_LAST_M * TAU / 8, about_point=cc),
                  mtrack.animate(rate_func=linear).set_value(DEMO_LAST_M),
                  run_time=2.6, rate_func=linear)
        mlab.clear_updaters()
        rot = TY('R(m theta) = mat(cos m theta, - sin m theta; sin m theta, cos m theta)',
                 size=23, color=WHITE).move_to([1.2, 0.4, 0])
        rotcap = T("each 2-D pair of q, k rotates by  m·θ", size=16, color=MUTED)
        rotcap.next_to(rot, UP, buff=0.25)
        rot2 = TY('e^(i m theta)', size=22, color=VIOLET)
        rot2.next_to(rot, DOWN, buff=0.25)
        self.cue("s06.4")
        self.play(Write(rotcap), Write(rot), Write(rot2), run_time=2.2)

        lads = VGroup()
        # One circle per frequency sample, rotating at its own constant rate —
        # the periods are facts, not a hand-typed "FAST / SLOW" caption.
        samples = FACTS["rope"]["samples"]
        last_pair = FACTS["rope"]["pairs"] - 1
        assert samples[-1]["pair"] == last_pair, "facts must cover the slowest pair"
        turns = [4 * TAU / 8, 1.2 * TAU / 8, 0.22 * TAU / 8]
        speeds = [(c, s, turn) for c, s, turn in
                  zip([BLUE, VIOLET, GOLD], samples, turns)]
        for i, (c, sample, ang) in enumerate(speeds):
            c_center = np.array([3.8 + 1.45 * (i - 1), 2.25, 0])
            cx = Circle(radius=0.46, stroke_color=WHITE, stroke_opacity=0.25,
                        stroke_width=1.3).move_to(c_center)
            ticks = VGroup(
                Line(c_center + LEFT * 0.46, c_center + RIGHT * 0.46, stroke_color=MUTED, stroke_opacity=0.25, stroke_width=1.0),
                Line(c_center + DOWN * 0.46, c_center + UP * 0.46, stroke_color=MUTED, stroke_opacity=0.25, stroke_width=1.0),
            )
            halo = glow_ring(cx, c)
            v = Arrow(cx.get_center(), cx.get_center() + RIGHT * 0.44,
                      stroke_color=c, fill_color=c, stroke_width=2.8,
                      tip_length=0.12, buff=0)
            lab = VGroup(
                DIN_T(f"PAIR {sample['pair']}", size=TYPE["micro"],
                      color=lighter(c, 0.2)),
                DIN_T(f"{period_label(sample['period_tokens'])} TOKENS",
                      size=TYPE["micro"], color=MUTED),
            ).arrange(DOWN, buff=0.05)
            lab.next_to(cx, DOWN, buff=0.12)
            lads.add(VGroup(cx, halo, v, lab, ticks))
        self.cue("s06.5")
        self.play(LaggedStart(*[AnimationGroup(Create(g[0]), FadeIn(g[1]),
                                               Create(g[2]), Write(g[3]), Create(g[4]))
                                for g in lads], lag_ratio=0.3),
                  run_time=1.5)
        # linear: a constant frequency is a constant angular velocity, so the
        # three pairs show their periods rather than an eased spin.
        self.cue("s06.6")
        self.play(*[Rotate(g[2], angle=ang, about_point=g[0].get_center())
                    for g, (_, _, ang) in zip(lads, speeds)],
                  run_time=2.6, rate_func=linear)
        fform = TY('theta_i = theta^(-2 i / d)', size=18, color=MUTED)
        fform.move_to([3.8, 1.05, 0])
        self.play(Write(fform), run_time=1.0)

        prop = TY('(R_m q)^T (R_n k) = g(m - n)', size=28, color=WHITE).move_to([0.6, -0.65, 0])
        pcap = T("depends only on the distance  m − n", size=16, color=VIOLET)
        pcap.next_to(prop, DOWN, buff=0.20)
        self.cue("s06.7")
        self.play(FadeOut(rot), FadeOut(rot2), FadeOut(rotcap),
                  Write(prop), run_time=1.4)
        self.cue("s06.8")
        self.play(Write(pcap), run_time=0.6)

        theta_chip = VGroup(
            T(f"θ = {THETA_LABEL}", size=16, color=GOLD, weight="BOLD"),
            T(f"—  slowest pairs stay smooth across {SEQ_LEN} tokens",
              size=15, color=MUTED),
        ).arrange(RIGHT, buff=0.20).move_to([0.6, -2.25, 0])
        self.cue("s06.9")
        self.play(Write(theta_chip), run_time=0.9)
        self.wait(1.5)
        self.fade_content(0.6)
        self.sync_end()


# =========================================================== scene 7 ========
class Section07(Section):
    SEC = "sec07_gqa"
    CHAPTER = "07 · GQA + QK-NORM"
    ACCENT = GOLD

    def construct(self):
        self._start_orb_drift()
        self.cue("s07.1")
        hdr = self.section_header("SHARING THE CACHE", "Grouped-query attention",
                                  run_time=1.0)

        mha_x = -3.4
        qrow = VGroup(*[head_badge(f"q{i+1}", BLUE, r=0.18)
                        for i in range(N_HEADS)])
        qrow.arrange(RIGHT, buff=0.20).move_to([mha_x, 1.5, 0])
        kvrow = VGroup(*[head_badge(f"kv{i+1}", GOLD, r=0.18)
                         for i in range(N_HEADS)])
        kvrow.arrange(RIGHT, buff=0.20).move_to([mha_x, -0.2, 0])
        lines = VGroup()
        for i in range(N_HEADS):
            for j in range(N_HEADS):
                lines.add(Line(qrow[i].get_bottom(), kvrow[j].get_top(),
                               stroke_color=WHITE, stroke_opacity=0.12,
                               stroke_width=0.8, buff=0.05))
        mlab = pill(f"multi-head attention   {N_HEADS} Q : {N_HEADS} KV",
                    color=MUTED, size=14)
        mlab.next_to(kvrow, DOWN, buff=0.4)
        self.cue("s07.2")
        self.play(FadeOut(hdr), run_time=0.5)
        self.play(LaggedStart(*[Write(h) for h in qrow], lag_ratio=0.06),
                  LaggedStart(*[Write(h) for h in kvrow], lag_ratio=0.06),
                  Create(lines), run_time=1.8)
        self.play(Write(mlab), run_time=0.6)

        per_group = N_HEADS // N_KV_HEADS
        kv_group = VGroup(*[head_badge(f"kv{i+1}", GOLD, r=0.18)
                            for i in range(N_KV_HEADS)])
        kv_group.arrange(RIGHT, buff=0.85).move_to([mha_x, -0.2, 0])
        arcs = VGroup()
        pair_cols = [BLUE, VIOLET]
        for qi in range(N_HEADS):
            ki = qi // per_group
            arcs.add(CurvedArrow(qrow[qi].get_bottom(), kv_group[ki].get_top(),
                                 stroke_color=pair_cols[qi % 2],
                                 fill_color=pair_cols[qi % 2],
                                 stroke_width=2.0, tip_length=0.11, angle=0.4,
                                 stroke_opacity=0.85, fill_opacity=0.85))
        glab = pill(f"grouped-query attention   {N_HEADS} Q : {N_KV_HEADS} KV",
                    color=WHITE, size=14, stroke=ACCENT)
        glab.move_to(mlab)
        share = T(f"each KV head shared by {per_group} query heads", size=12,
                  color=MUTED)
        share.next_to(glab, DOWN, buff=0.18)
        self.cue("s07.3")
        self.play(FadeOut(lines), run_time=0.3)
        self.play(ReplacementTransform(kvrow, kv_group),
                  LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.08),
                  run_time=1.1)
        self.play(ReplacementTransform(mlab, glab), Write(share), run_time=0.7)
        self.play(flow(arcs[1], VIOLET, time_width=0.5, run_time=0.5),
                  run_time=0.5)

        cache_x = 4.0

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

        n_mha, n_gqa = cache_bar_counts()
        cache_title = T("KV cache — per layer × past token", size=14,
                        color=MUTED).move_to([cache_x, 1.82, 0])
        mha_cache = cache_stack(n_mha, ROSE, f"MHA · {N_HEADS} KV heads")
        mha_cache[0].move_to([cache_x - 1.15, 0.35, 0])
        gqa_cache = cache_stack(n_gqa, MINT, f"GQA · {N_KV_HEADS} KV heads")
        gqa_cache[0].next_to(mha_cache[0], RIGHT, buff=1.45)
        gqa_cache[0].align_to(mha_cache[0], DOWN)
        for grp in (mha_cache, gqa_cache):
            grp[1].move_to([grp[0].get_center()[0], -1.15, 0])
        arrow_halve = data_arrow(mha_cache[0].get_right() + RIGHT * 0.1,
                                 gqa_cache[0].get_left() + LEFT * 0.1, MINT,
                                 w=2.4)
        halve_halo = glow_ring(arrow_halve, MINT)
        arrow_halve = VGroup(halve_halo, arrow_halve)
        halve_lab = T("÷ 2", size=18, color=MINT, weight="BOLD")
        halve_lab.next_to(arrow_halve, UP, buff=0.14)
        self.cue("s07.4")
        self.play(Write(cache_title), GrowFromEdge(mha_cache, UP), run_time=1.3)
        self.cue("s07.5")
        self.play(GrowFromEdge(gqa_cache, UP), FadeIn(arrow_halve),
                  Write(halve_lab), run_time=1.4)

        strip_y = -2.15
        qn = VGroup(T("q", size=20, color=BLUE, weight="BOLD"),
                    pill("RMSNorm", color=WHITE, size=13),
                    pill("RoPE", color=VIOLET, size=13))
        qn.arrange(RIGHT, buff=0.32).move_to([-2.3, strip_y, 0])
        kn = VGroup(T("k", size=20, color=GOLD, weight="BOLD"),
                    pill("RMSNorm", color=WHITE, size=13),
                    pill("RoPE", color=VIOLET, size=13))
        kn.arrange(RIGHT, buff=0.32).move_to([2.3, strip_y, 0])
        qk_lab = pill("QK-norm — bounds the scale of q and k", color=MINT,
                      size=14, stroke=MINT)
        qk_lab.move_to([0, strip_y - 0.70, 0])
        caveat = T("a Lite addition — LLaMA 3 itself has none", size=13,
                   color=GOLD)
        caveat.next_to(qk_lab, DOWN, buff=0.14)
        self.cue("s07.6")
        self.play(LaggedStart(Write(qn), Write(kn), lag_ratio=0.2), run_time=1.2)
        self.cue("s07.7")
        self.play(Write(qk_lab), run_time=0.8)
        self.cue("s07.8")
        self.play(Write(caveat), run_time=0.7)
        rem = self.budget - self._t - 0.45
        if rem > 0:
            self.wait(rem)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 8 ========
class Section08(Section):
    SEC = "sec08_swiglu"
    CHAPTER = "08 · SWIGLU"
    ACCENT = MINT

    def construct(self):
        self._start_orb_drift()
        self.cue("s08.1")
        hdr = self.section_header("THE GATED MLP", "SwiGLU — the gated feed-forward",
                                  run_time=1.0)

        x_box = pill(f"x   ({D_MODEL:,})", color=WHITE, size=17, stroke=WHITE)
        x_box.move_to([-5.4, 0.6, 0])
        gate = card(2.5, 0.72, edge=GOLD)
        gate.move_to([-2.7, 1.6, 0])
        gate_lab = T(f"W_gate  → {D_FF:,}", size=16,
                     color=lighter(GOLD, 0.2)).move_to(gate)
        up = card(2.5, 0.72, edge=BLUE)
        up.move_to([-2.7, -0.4, 0])
        up_lab = T(f"W_up  → {D_FF:,}", size=16,
                   color=lighter(BLUE, 0.2)).move_to(up)
        sil = pill("SiLU", color=GOLD, size=15, stroke=GOLD)
        sil.move_to([-0.6, 1.6, 0])
        mult = odot(MINT).move_to([0.9, 0.6, 0])
        mult_halo = glow_ring(mult[0], MINT)
        mult_lab = DIN_T("ELEMENT-WISE", size=TYPE["micro"], color=MINT)
        # Below-right of the ⊙: the 12 pt label no longer sits under the a4 arrow
        # or the W_down card (it overlapped both at the old 8 pt size).
        mult_lab.move_to(mult.get_center() + RIGHT * 0.55 + DOWN * 0.52)
        down = card(2.5, 0.72, edge=MINT)
        down.move_to([2.9, 0.6, 0])
        down_lab = T(f"W_down  → {D_MODEL:,}", size=16,
                     color=lighter(MINT, 0.2)).move_to(down)
        out = pill(f"out   ({D_MODEL:,})", color=WHITE, size=17, stroke=WHITE)
        out.move_to([5.5, 0.6, 0])
        a1 = dim_arrow(x_box.get_right(), gate.get_left())
        a2 = dim_arrow(x_box.get_right(), up.get_left())
        a3 = data_arrow(gate.get_right(), sil.get_left(), GOLD, w=2.0)
        a4 = data_arrow(sil.get_right(), mult.get_center() + UP * 0.32, GOLD,
                        w=2.0)
        a5 = data_arrow(up.get_right(), mult.get_center() + DOWN * 0.32, BLUE,
                        w=2.0)
        a6 = data_arrow(mult.get_right(), down.get_left(), MINT, w=2.2)
        a7 = dim_arrow(down.get_right(), out.get_left())
        per_tok = T("per token — no mixing across tokens here", size=13,
                    color=MUTED).move_to([2.9, -0.62, 0])
        self.cue("s08.2")
        self.play(FadeOut(hdr), Write(x_box),
                  LaggedStart(Write(gate), Write(gate_lab),
                              Create(a1), Write(up),
                              Write(up_lab), Create(a2),
                              lag_ratio=0.16),
                  run_time=2.2)
        self.cue("s08.3")
        self.play(Write(sil), Create(a3), run_time=0.9)
        self.cue("s08.4")
        self.play(Create(a4), Create(a5), run_time=0.8)

        plot_card = card(4.0, 2.5, edge=GOLD)
        plot_card.move_to([-1.5, -2.2, 0])
        ax = Axes(x_range=[-3.5, 3.5, 1], y_range=[-1.5, 4, 1], x_length=2.5,
                  y_length=1.85, tips=False,
                  axis_config={"stroke_color": WHITE, "stroke_opacity": 0.15,
                               "stroke_width": 1.2})
        ax.move_to(plot_card.get_center() + LEFT * 0.72)
        silu = ax.plot(lambda t: t / (1 + np.exp(-t)), color=GOLD,
                       stroke_width=3, x_range=[-3.5, 3.5])
        silu_halo = glow_ring(silu, GOLD)
        relu = DashedVMobject(ax.plot(lambda t: np.maximum(0, t),
                                      stroke_color=WHITE, stroke_width=1.5,
                                      stroke_opacity=0.35,
                                      x_range=[-3.5, 3.5]), num_dashes=40)
        sl = TY('"SiLU"(x) = x dot sigma(x)', size=17, color=GOLD)
        sl.next_to(ax, RIGHT, buff=0.3).align_to(ax, UP).shift(DOWN * 0.15)
        rl = DIN_T("RELU — DASHED", size=TYPE["micro"], color=MUTED).next_to(sl, DOWN, buff=0.2)
        self.cue("s08.5")
        self.play(Create(plot_card), Create(ax), run_time=0.9)
        self.play(Create(relu), run_time=0.9)
        self.play(Create(silu), FadeIn(silu_halo), run_time=1.5)
        self.play(Write(sl), Write(rl), run_time=0.7)

        silu_vals, up_vals, prod_vals = swiglu_demo()
        rows = VGroup()
        for r, (lab, vals, col) in enumerate(
                [("SiLU(gate)", silu_vals, GOLD),
                 ("up", up_vals, BLUE),
                 ("product", prod_vals, MINT)]):
            cells = VGroup()
            for v in vals:
                bx = RoundedRectangle(corner_radius=0.07, width=0.40,
                                      height=0.40, stroke_color=col,
                                      stroke_opacity=0.55, stroke_width=1.2,
                                      fill_color=col,
                                      fill_opacity=0.15 if r == 2 else 0.06)
                tx = Text(f"{v:.2f}", font=DIN, font_size=pt(12),
                          color=lighter(col, 0.2), weight="BOLD").move_to(bx)
                cells.add(VGroup(bx, tx))
            cells.arrange(RIGHT, buff=0.10).move_to([3.95, -1.14 - 0.46 * r, 0])
            rlab = T(lab, size=12, color=col, weight="MEDIUM")
            rlab.next_to(cells, LEFT, buff=0.18)
            rows.add(VGroup(rlab, cells))
        # The ⊙ is a product, not a switch: every feature survives, scaled.
        gcap = T("the gate scales each feature — nothing is switched off",
                 size=12, color=MUTED)
        gcap.next_to(rows, DOWN, buff=0.16)
        self.cue("s08.6")
        self.play(FadeIn(mult_halo), GrowFromCenter(mult), Write(mult_lab),
                  run_time=0.8)
        self.play(LaggedStart(*[Write(r) for r in rows], lag_ratio=0.30),
                  Write(gcap), run_time=1.6)

        fx_card = card(5.6, 0.85, edge=MINT)
        # Up to y = -3.1: at -3.3 the card crossed the safe-rect bottom.
        fx_card.move_to([3.45, -3.1, 0])
        fx = TY('"FFN"(x) = W_("down")("SiLU"(W_("gate") x) dot W_("up") x)',
                size=22, color=WHITE).move_to(fx_card)
        self.cue("s08.7")
        self.play(Create(a6), Write(down), Write(down_lab),
                  Create(a7), Write(out), Write(per_tok), run_time=0.7)
        self.play(Create(fx_card), Write(fx), run_time=1.0)
        rem = self.budget - self._t - 0.45
        if rem > 0:
            self.wait(rem)
        self.fade_content(0.40)
        self.sync_end()


# =========================================================== scene 9 ========
class Section09(Section):
    SEC = "sec09_output"
    CHAPTER = "09 · OUTPUT HEAD"
    ACCENT = ROSE

    def construct(self):
        self._start_orb_drift()
        self.cue("s09.1")
        hdr = self.section_header("CLOSING THE LOOP", "From vector to next token",
                                  run_time=1.0)

        stack = VGroup()
        for i in range(N_LAYERS):
            stack.add(RoundedRectangle(corner_radius=0.045, width=1.5,
                                       height=0.11, stroke_color=WHITE,
                                       stroke_opacity=0.13, stroke_width=1.0,
                                       fill_color=WHITE, fill_opacity=0.045))
        stack.arrange(DOWN, buff=0.085).move_to([4.35, 0.95, 0])
        fin = pill("final RMSNorm", color=GOLD, size=13)
        fin.next_to(stack, DOWN, buff=0.22)
        wu = pill(f"W_U   {D_MODEL:,} → {VOCAB_LABEL}", color=ROSE, size=13,
                  stroke=ROSE)
        wu.next_to(fin, DOWN, buff=0.22)
        slab = DIN_T(f"× {N_LAYERS} BLOCKS", size=TYPE["micro"], color=MUTED)
        slab.next_to(stack, RIGHT, buff=0.25)
        self.play(LaggedStart(*[Create(s) for s in stack], lag_ratio=0.06),
                  Write(fin), Write(slab), run_time=1.8)

        words = ["The", "cat", "sat", "on", "the"]
        ctx = VGroup(*[token_pill(w, color=BLUE) for w in words])
        ctx.arrange(RIGHT, buff=0.28).move_to([-3.4, 1.35, 0])
        cursor = RoundedRectangle(corner_radius=0.03, width=0.12, height=0.44,
                                  stroke_width=0, fill_color=ROSE,
                                  fill_opacity=1)
        cursor.next_to(ctx, RIGHT, buff=0.14)
        cursor_halo = glow_ring(cursor, ROSE)
        logits = VGroup(*[Text(v, font=DIN, font_size=pt(14), color=MUTED,
                               weight="BOLD")
                          for v in ["17.2", "15.8", "14.1", "13.6", "…"]])
        logits.arrange(RIGHT, buff=0.5).move_to([-3.4, 0.55, 0])
        logit_lab = T(f"logits — one score per vocabulary item ({VOCAB_LABEL})",
                      size=12, color=MUTED)
        logit_lab.next_to(logits, DOWN, buff=0.12)
        self.cue("s09.2")
        self.play(FadeOut(hdr), run_time=0.4)
        self.play(Write(wu),
                  LaggedStart(*[Write(c) for c in ctx], lag_ratio=0.12),
                  FadeIn(cursor, cursor_halo), run_time=1.3)
        self.play(LaggedStart(*[Write(l) for l in logits], lag_ratio=0.1),
                  Write(logit_lab),
                  Indicate(wu, color=ROSE), run_time=1.7)

        pb = bars(["mat", "floor", "rug", "…"],
                  [0.82, 0.07, 0.04, 0.07],
                  [ROSE, MUTED, MUTED, MUTED], width=2.2, size=16)
        pb.move_to([-3.2, -1.6, 0])
        softmax_lab = TY('p = "softmax"("logits")', size=18, color=MUTED)
        softmax_lab.next_to(pb, UP, buff=0.24)
        self.cue("s09.3")
        self.play(GrowFromEdge(pb, DOWN), Write(softmax_lab), run_time=1.9)

        # The new token is drawn from copies of the top bar: the probability
        # chart stays complete behind it (v2 consumed bar 0 and left three).
        mat_chip = token_pill("mat", color=ROSE)
        mat_chip.next_to(ctx, RIGHT, buff=0.55)
        self.cue("s09.4")
        self.play(ReplacementTransform(pb[0][2].copy(), mat_chip[0]),
                  ReplacementTransform(pb[0][0].copy(), mat_chip[1]),
                  FadeOut(cursor, cursor_halo), run_time=0.9)
        # Routed above the token row, stroke-only: the filled arc used to paint
        # a rose wedge over the logits. Only the arrowhead keeps a fill.
        loop = CurvedArrow(mat_chip.get_top(), ctx[0].get_top(),
                           stroke_color=ROSE, fill_color=ROSE,
                           stroke_width=2.2, tip_length=0.14, angle=0.64,
                           stroke_opacity=0.85, fill_opacity=0)
        loop.tip.set_fill(ROSE, 0.95)
        loop_halo = glow_ring(loop, ROSE)
        loop_cap = T("one forward pass per generated token", size=14, color=ROSE)
        loop_cap.move_to([-3.4, -2.95, 0])
        self.cue("s09.5")
        self.play(Create(loop), FadeIn(loop_halo), Write(loop_cap), run_time=1.3)

        self.fade_content(0.6)
        recap_title = T("the entire architecture", size=24, color=WHITE,
                        weight="BOLD").move_to([0, 2.2, 0])
        row1 = VGroup(pill("Embedding", color=BLUE, size=13),
                      T("→", size=15, color=DIM),
                      pill("RMSNorm · RoPE · GQA  +", color=GOLD, size=13),
                      T("→", size=15, color=DIM),
                      pill("RMSNorm · SwiGLU  +", color=VIOLET, size=13),
                      T("→", size=15, color=DIM),
                      pill("Norm", color=GOLD, size=13),
                      T("→", size=15, color=DIM),
                      pill("Unembed · softmax", color=ROSE, size=13))
        row1.arrange(RIGHT, buff=0.12).move_to([0, 0.95, 0])
        # The brace has to span the two repeated pills, not float over the row.
        brace = Brace(VGroup(row1[2], row1[4]), UP, buff=0.10)
        brace_lab = DIN_T(f"× {N_LAYERS} BLOCKS", size=TYPE["micro"], color=MUTED)
        brace_lab.next_to(brace, UP, buff=0.10)
        nums = Text(f"d_model {D_MODEL:,}   ·   {N_HEADS} Q : {N_KV_HEADS} KV heads"
                    f"   ·   d_ff {D_FF:,}   ·   vocab {VOCAB_LABEL}   ·   "
                    f"θ = {THETA_LABEL}   ·   ε = {EPS_LABEL}",
                    font=DIN, font_size=pt(12), color=MUTED, weight="BOLD")
        nums.move_to([0, 0.15, 0])
        chain = Line(row1[0].get_right(), row1[-1].get_left(),
                     stroke_color=WHITE, stroke_opacity=0.12, stroke_width=1.0)
        # chain first so it sits behind the pills instead of over their text.
        self.cue("s09.6")
        self.play(Write(recap_title),
                  LaggedStart(Create(chain),
                              *[Write(m) if i % 2 == 0 else FadeIn(m)
                                for i, m in enumerate(row1)], lag_ratio=0.08),
                  run_time=2.2)
        self.cue("s09.7")
        self.play(Create(brace), Write(brace_lab), Write(nums), run_time=1.2)
        self.play(flow(chain, ACCENT, time_width=0.5, run_time=0.8),
                  run_time=0.8)

        # end card — ambient stays, chrome and content leave
        self.fade_content(0.5)
        self.play(FadeOut(self._chrome), run_time=0.5)
        kick = DIN_T("THE COMPLETE ARCHITECTURE", size=12, color=GOLD)
        end1 = T("LLaMA-3-Lite", size=56, color=WHITE, weight="BOLD")
        end1.set_color_by_gradient("#FFFFFF", "#AAB6FF")
        rule = RoundedRectangle(corner_radius=0.015, width=1.4, height=0.03,
                                stroke_width=0, fill_color=GOLD,
                                fill_opacity=0.9)
        end2 = T(f"{PARAMS_LABEL} parameters — and nothing else.", size=20,
                 color=MUTED)
        footer = DIN_T("DECODER-ONLY   ·   RoPE   ·   GQA   ·   SWIGLU   ·   RMSNORM",
                       size=TYPE["micro"], color=DIM)
        end = VGroup(kick, end1, rule, end2, footer).arrange(DOWN, buff=0.34)
        end.move_to([0, 0.25, 0])
        wm = self.watermark()
        self.cue("s09.8")
        self.play(Write(kick), Write(end1),
                  Create(rule), Write(end2),
                  Write(footer), FadeIn(wm), run_time=1.5)
        rem = self.budget - self._t - 0.45
        if rem > 0:
            self.wait(rem)
        self.play(FadeOut(end), FadeOut(wm), run_time=0.40)
        self.sync_end()


ALL_SECTIONS = [Section01, Section02, Section03, Section04, Section05,
                Section06, Section07, Section08, Section09]


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

