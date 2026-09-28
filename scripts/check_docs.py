#!/usr/bin/env python3
"""Doc↔code alignment checker for LLaMA-3-Lite.

Resolves every `file.py:Symbol` / `file.py::Symbol` citation in the markdown
docs against the working tree, flags line-number citations (they rot), and
validates intra-repo markdown links including their `#heading` anchors.
`--coverage` additionally requires every public symbol in COVERAGE_MODULES
to be cited at least once.

This is the portfolio's reference gate: the other repos ported their
`check_docs.py` from here. `tests/test_doc_refs.py` is the pytest variant
and stays the single source of truth for anchor resolution.

Usage:
    python3 scripts/check_docs.py                      # resolve all anchors
    python3 scripts/check_docs.py --coverage --links
    python3 -m pytest tests/test_doc_refs.py           # same gate under pytest
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Imported doc modules may reference project packages (kernels, shared_data).
# `tests/` is here because the docs cite tests/*.py symbols, and those modules
# do `import conftest` — pytest puts this dir on the path, a bare CLI does not.
for _p in (ROOT, ROOT / "data", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DOC_PATHS = [
    ROOT / "docs",
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "SKILLS.md",
]
# Every public module-level symbol here must be cited in the docs.
COVERAGE_MODULES = [
    "model.py",
    "train.py",
    "config.py",
    "dataset.py",
    "benchmark_data.py",
    "kernels/rmsnorm_triton.py",
    "kernels/swiglu_triton.py",
    "kernels/cross_entropy_triton.py",
    "data/prepare_data.py",
    "data/shared_data/loader.py",
]
# `file.py:Symbol` or `file.py::Class.method` (backticked, or bare in prose)
ANCHOR_RE = re.compile(
    r"`?([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]*)"  # file.py:Symbol
    r"|`?([A-Za-z_][A-Za-z0-9_./-]*\.py)::([A-Za-z_][A-Za-z0-9_.]*)"  # file.py::Class.test
)
# Line-number anchors: file.py:123 or file.py L123 / (L123-140) / bare L123–456.
# "L2 norm", "L1 loss", etc. are math terms, not anchors — excluded.
LINE_ANCHOR_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_./-]*\.py)\s*[:L]\s*\d+"
    r"|\bL\d+(?:\s*[–-]\s*\d+)?\b(?!\s*(?:norm|regularization|reg|loss|penalty|distance|ball|error|regularizer)\b)"
)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Heading-anchor machinery: a link target may carry a `#fragment`, which must
# match a heading id in the target file. GitHub derives ids as: drop inline
# markup, lowercase, keep word chars and hyphens, spaces -> hyphens, and give
# a repeated heading the `-1`, `-2` suffix.
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
SETEXT_RE = re.compile(r"^(?:=+|-{2,})\s*$")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SLUG_STRIP_RE = re.compile(r"[^\w\s-]")

_MODULE_CACHE: dict[str, object | None] = {}
_SLUG_CACHE: dict[Path, set[str]] = {}


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for p in DOC_PATHS:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.exists():
            files.append(p)
    return files


def _load_module(rel_path: str):
    """Import a repo module by relative path; returns (module, error).

    By-path loading, not importlib.import_module — this repo's modules are
    flat (`model.py`, `train.py`), not an importable package.
    """
    if rel_path in _MODULE_CACHE:
        mod = _MODULE_CACHE[rel_path]
        return (mod, None) if mod is not None else (None, f"previous import failure: {rel_path}")
    path = ROOT / rel_path
    if not path.exists():
        _MODULE_CACHE[rel_path] = None
        return None, f"unknown file: {rel_path}"
    name = f"_docref_{path.stem}_{abs(hash(str(path))) % (10 ** 8)}"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        _MODULE_CACHE[rel_path] = None
        return None, f"import failed for {rel_path}: {type(exc).__name__}: {exc}"
    _MODULE_CACHE[rel_path] = module
    return module, None


def _has_instance_attr(cls, name: str) -> bool:
    """Instance attrs assigned via self.name = … are invisible to hasattr."""
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False
    return re.search(rf"self\.{re.escape(name)}\s*=", src) is not None


def collect_anchors() -> list[tuple[Path, str, str]]:
    """Return (doc_path, file.py, symbol) triples from all scanned docs."""
    anchors: list[tuple[Path, str, str]] = []
    for doc in _doc_files():
        for m in ANCHOR_RE.finditer(doc.read_text(encoding="utf-8")):
            anchors.append((doc, m.group(1) or m.group(3), m.group(2) or m.group(4)))
    return anchors


def resolve_anchor(rel_path: str, symbol: str) -> tuple[bool, str | None]:
    mod, err = _load_module(rel_path)
    if err:
        return False, err
    obj = mod
    for part in symbol.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        if isinstance(obj, type) and _has_instance_attr(obj, part):
            continue
        return False, f"{rel_path}:{symbol} — '{part}' not found"
    return True, None


def check_resolution() -> list[str]:
    failures = []
    for doc, rel_path, symbol in collect_anchors():
        if rel_path == "file.py":
            continue  # literal metasyntax placeholder in the writing contract
        ok, err = resolve_anchor(rel_path, symbol)
        if not ok:
            failures.append(f"{doc.relative_to(ROOT)}: {err}")
    return failures


def public_symbols(rel_path: str) -> list[str]:
    mod, err = _load_module(rel_path)
    if err:
        return []
    return sorted(
        n for n in dir(mod)
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", n)
        and not n.startswith("_")
        and callable(getattr(mod, n))
        and getattr(mod, n).__module__ == mod.__name__
    )


def check_coverage() -> list[str]:
    """Every public module-level symbol in COVERAGE_MODULES must be cited once."""
    cited = {f"{rel}:{sym.split('.')[0]}" for _, rel, sym in collect_anchors()}
    missing = []
    for rel_path in COVERAGE_MODULES:
        for sym in public_symbols(rel_path):
            if f"{rel_path}:{sym}" not in cited:
                missing.append(f"{rel_path}:{sym}")
    return missing


def check_line_anchors() -> list[str]:
    """Flag line-number citations (file.py:123, L15–32) — they rot after refactors."""
    hits = []
    for doc in _doc_files():
        for m in LINE_ANCHOR_RE.finditer(doc.read_text(encoding="utf-8")):
            hits.append(f"{doc.relative_to(ROOT)}: {m.group(0)!r}")
    return hits


def _slug(text: str) -> str:
    """GitHub heading id for a heading's text."""
    text = MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"[*~]", "", text).lower()
    # GitHub maps EACH space to its own hyphen — never collapse runs, or
    # "Part B — Config" would yield `part-b-config` instead of `part-b--config`.
    return SLUG_STRIP_RE.sub("", text).strip().replace(" ", "-")


def heading_slugs(path: Path) -> set[str]:
    """Every anchor id `path` exposes. Setext headings count, thematic
    breaks do not, and a repeated heading gets GitHub's `-1`, `-2` suffix.
    """
    if path in _SLUG_CACHE:
        return _SLUG_CACHE[path]
    lines = FENCE_RE.sub("", path.read_text(encoding="utf-8")).split("\n")
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            title = m.group(1)
        elif line.strip() and i + 1 < len(lines) and SETEXT_RE.match(lines[i + 1]):
            title = line.strip()
        else:
            continue
        base = _slug(title)
        if not base:
            continue
        n = counts.get(base, 0)
        counts[base] = n + 1
        slugs.add(base if n == 0 else f"{base}-{n}")
    _SLUG_CACHE[path] = slugs
    return slugs


def check_links() -> list[str]:
    """Validate intra-repo markdown links *and* their `#fragment` anchors;
    code fences are stripped first."""
    broken = []
    for doc in _doc_files():
        text = FENCE_RE.sub("", doc.read_text(encoding="utf-8"))
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            if not target or target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path_part, _, fragment = target.partition("#")
            if not path_part:
                continue
            candidates = [(doc.parent / path_part).resolve(), (ROOT / path_part).resolve()]
            resolved = next((c for c in candidates if c.exists()), None)
            if resolved is None:
                broken.append(f"{doc.relative_to(ROOT)}: broken link -> {target}")
            elif fragment and resolved.suffix == ".md":
                if fragment.lower() not in heading_slugs(resolved):
                    broken.append(f"{doc.relative_to(ROOT)}: dead anchor -> {target}")
    return broken


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Doc↔code alignment checker")
    ap.add_argument("--coverage", action="store_true", help="also enforce public-symbol coverage")
    ap.add_argument("--links", action="store_true", help="also validate intra-repo markdown links")
    args = ap.parse_args()

    anchors = collect_anchors()
    failures = check_resolution()
    print(f"[doc-refs] scanned {len(_doc_files())} docs, {len(anchors)} anchors")
    for f in failures:
        print(f"  FAIL {f}")
    print(f"[doc-refs] resolution: {'PASS' if not failures else f'{len(failures)} FAILURES'}")

    missing = check_coverage() if args.coverage else []
    if args.coverage:
        for m in missing:
            print(f"  UNCOVERED {m}")
        print(f"[doc-refs] coverage: {'PASS' if not missing else f'{len(missing)} UNCOVERED'}")

    line_hits = check_line_anchors()
    for h in line_hits:
        print(f"  LINE-ANCHOR {h}")
    print(f"[doc-refs] line anchors: {'PASS' if not line_hits else f'{len(line_hits)} FOUND'}")

    link_broken = check_links() if args.links else []
    if args.links:
        for b in link_broken:
            print(f"  BROKEN-LINK {b}")
        print(f"[doc-refs] links: {'PASS' if not link_broken else f'{len(link_broken)} BROKEN'}")

    return 1 if (failures or (args.coverage and missing) or line_hits
                 or (args.links and link_broken)) else 0


if __name__ == "__main__":
    sys.exit(main())
