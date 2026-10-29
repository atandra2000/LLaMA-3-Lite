"""Check a scene's beat clock before rendering: cue gaps vs the plays between.

The render log is where a LATE cue shows up, but a 1080p render is minutes per
section. This reads the same arithmetic the scene will run — every run_time
between two cue() calls has to fit the gap between those sentences — and prints
the overruns and the long holds in seconds.

    .venv/bin/python tools/check_beats.py Section03
    .venv/bin/python tools/check_beats.py            # every scene
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TIM = json.loads((HERE / "audio" / "timings.json").read_text())
LEAD, TAIL = TIM["lead"], TIM["tail"]
HOLD = 2.5          # a gap the plays do not fill; the render sits on a held frame


def number(node):
    """The literal value of a run_time expression, or None if it is not a literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def play_time(call: ast.Call) -> float:
    """How long Section.play will run this call for."""
    for kw in call.keywords:
        if kw.arg == "run_time":
            return number(kw.value) or 1.0
    if call.args:
        inner = call.args[0]
        if isinstance(inner, ast.Call):
            for kw in inner.keywords:
                if kw.arg == "run_time":
                    return number(kw.value) or 1.0
            # flow() defaults to 0.8s and passes it down as the animation time
            fn = inner.func
            if (isinstance(fn, ast.Name) and fn.id == "flow") or (
                    isinstance(fn, ast.Attribute) and fn.attr == "flow"):
                return 0.8
    return 1.0


def walk_time(body) -> list[tuple[str, float | None]]:
    """[(kind, seconds)] in source order, walking into if/for blocks."""
    out = []
    for node in body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            fn = call.func
            name = fn.attr if isinstance(fn, ast.Attribute) else ""
            if name == "cue":
                out.append(("cue", call.args[0].value
                            if call.args and isinstance(call.args[0], ast.Constant)
                            else None))
            elif name == "play":
                out.append(("play", play_time(call)))
            elif name == "wait":
                out.append(("wait", number(call.args[0]) if call.args else None))
            elif name == "fade_content":
                out.append(("fade", number(call.args[0]) if call.args else 0.6))
            elif name == "sync_end":
                out.append(("sync", 0.0))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            # `hdr = self.section_header(..., run_time=N)` spends N seconds
            fn = node.value.func
            name = fn.attr if isinstance(fn, ast.Attribute) else ""
            if name == "section_header":
                out.append(("play", play_time(node.value)))
        elif isinstance(node, (ast.If, ast.For, ast.While)):
            times = 1
            if isinstance(node, ast.For):
                times = int(loop_count(node.iter) or 1)
            for _ in range(times):
                out += walk_time(node.body)
            out += walk_time(node.orelse)
    return out


def loop_count(iter_node) -> int | None:
    """How many times a ``for`` body runs, when the count is readable.

    Scene code loops over a fixed tuple (``zip(ticks, (a, b, c, d))``) or a
    ``range(N)``. Walking the body once hides every repeat, which reports a
    long hold where there is none — and can hide a LATE cue where there is.
    """
    if isinstance(iter_node, ast.Call):
        fn = iter_node.func
        name = fn.id if isinstance(fn, ast.Name) else (
            fn.attr if isinstance(fn, ast.Attribute) else "")
        if name == "range" and iter_node.args:
            return number(iter_node.args[-1])
        if name == "zip" and iter_node.args:
            lengths = [loop_count(a) for a in iter_node.args]
            lengths = [n for n in lengths if n]
            return max(lengths) if lengths else None
    if isinstance(iter_node, (ast.Tuple, ast.List)):
        return len(iter_node.elts)
    return None


def scene_sec(tree, class_name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                    if stmt.targets[0].id == "SEC" and isinstance(stmt.value, ast.Constant):
                        return stmt.value.value
    return None


def check(class_name: str) -> int:
    src = (HERE / "render_scenes.py").read_text()
    tree = ast.parse(src)
    sec = scene_sec(tree, class_name)
    if sec is None:
        print(f"{class_name}: no SEC found")
        return 1
    if sec not in TIM["sections"]:
        print(f"{class_name}: SEC {sec!r} is not in timings.json")
        return 1

    construct = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "construct":
                    construct = stmt
    if construct is None:
        print(f"{class_name}: no construct()")
        return 1

    events = walk_time(construct.body)
    spans = TIM["sections"][sec]["sentences"]
    starts = {sid: LEAD + s["start"] for sid, s in spans.items()}
    budget = TIM["sections"][sec]["duration"] + LEAD + TAIL

    # Split into beats: one beat per cue, running until the next cue.
    beats: list[tuple[str | None, float]] = []
    cur_id, cur_sum = None, 0.0
    for kind, val in events:
        if kind == "cue":
            beats.append((cur_id, cur_sum))
            cur_id, cur_sum = val, 0.0
        elif kind in ("play", "wait", "fade"):
            if val is None:
                print(f"   note: a {kind} after {cur_id} has no literal run_time")
                val = 0.0
            cur_sum += val
    beats.append((cur_id, cur_sum))

    bad = 0
    print(f"{class_name}  SEC={sec}  budget={budget:.2f}s")
    for i, (sid, spent) in enumerate(beats):
        nxt = beats[i + 1][0] if i + 1 < len(beats) else None
        if sid is None:
            if spent > 1e-9:
                print(f"   before the first cue: {spent:.2f}s of plays")
            continue
        if sid not in starts:
            print(f"   {sid}: not a sentence of {sec}")
            bad += 1
            continue
        limit = (starts[nxt] - starts[sid]) if nxt in starts else (
            budget - starts[sid])
        delta = spent - limit
        note = ""
        if delta > 1 / 30:
            note = f"LATE by {delta:.2f}s"
            bad += 1
        elif limit - spent > HOLD:
            note = f"hold {limit - spent:.2f}s"
        print(f"   {sid:<8} plays {spent:5.2f}s / gap {limit:5.2f}s  {note}")
    print(f"   {'TOTAL':<8} {sum(s for _, s in beats):5.2f}s / budget {budget:5.2f}s")
    return bad


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    names = args or [
        n.name for n in ast.parse((HERE / "render_scenes.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name.startswith("Section")
        and n.name != "Section"]
    bad = sum(check(n) for n in names)
    if len(names) > 1:
        print(f"\n{bad} late beat(s) across {len(names)} scene(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
