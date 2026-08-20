# Docs Premium Re-theme — Design Spec

**Date:** 2026-08-17 · **Status:** Approved (design reviewed in-session)
**Scope:** `assets/style.css`, `scripts/build_docs_html.py`, regenerated `docs_html/`

## Goal

Elevate the static documentation portal to a "premium developer tool"
standard: minimalist, dark, code-centric — while preserving every word of
existing content and the Markdown-source build pipeline.

## Locked decisions (from brainstorming session)

1. **Theme:** "Dark Bench Notebook" — the existing manila-notebook identity
   translated to dark. Espresso-graphite surfaces, warm-bone ink, terracotta
   + olive accents retained.
2. **Hero animation:** "Breathing Field" — abstract llama silhouette rendered
   as a perpetually animated character field; shading glyphs (`.:░▒▓█`)
   cycle under two interfering waves; terracotta glints on wave crests.
   Static silhouette fallback under `prefers-reduced-motion`.
3. **Dark only:** theme toggle removed; light token blocks deleted.
4. **Type:** JetBrains Mono for everything (headings, prose, labels, code),
   weights 400/500/600. IBM Plex Serif dropped.
5. **Widgets:** interactive GQA wiring figure (hover a Q head → paired KV
   head + wire light up) and collapsible long code blocks (threshold
   14 lines; `expand ▾ · N lines` affordance).

## Architecture

`docs_html/` is a git-ignored build artifact. All changes land in the two
sources of truth; the portal is regenerated at the end.

- `assets/style.css` — full rewrite as the dark design system.
- `scripts/build_docs_html.py` — template changes (toggle removal, hero
  canvas, GQA widget, code-collapse). New JS kept inline per existing
  pattern but in plain (non-f-string) Python constants to avoid
  brace-escaping.

No new build or runtime dependencies.

## § 1 Design-system tokens

| Token | Value | Role |
|---|---|---|
| `--bg` | `#12100d` | page |
| `--bg-2` | `#161310` | card |
| `--bg-3` | `#1a1712` | inset plane |
| `--fg` | `#d8ccb4` | warm-bone ink |
| `--fg-faint` | `#5c554a` | muted ink |
| `--rule` | `#2c261c` | hairline |
| `--terracotta` | `#e07a3f` | active/anomaly accent |
| `--olive` | `#9a9440` | structural accent |

Components: hairline-bordered cards, corner-tick details, quiet hover lifts
(border-color + 2px translate), no heavy shadows. All motion gated behind
`prefers-reduced-motion`.

## § 2 Landing page

- Decorative GQA hero SVG → `<pre id="asciiHero">` Breathing Field at ~3×
  brainstorm-preview resolution.
- Interactive GQA wiring figure becomes its own terminal-styled widget below
  the DATASHEET spec grid; same 8Q→4KV topology.
- Title, subtitle, coordinate strip, spec grid: **verbatim preserved**.

## § 3 Doc pages

- Theme toggle button + `toggleTheme()` JS removed.
- Fenced code blocks > 14 lines: collapsed by default, expand in place.
- Sidebar, breadcrumb, TOC, prev/next, KaTeX, highlight.js unchanged in
  structure.

## § 4 Content integrity

No Markdown source file is modified. All portal text preserved verbatim.
Only presentational markup changes (SVG → canvas; toggle removed).

## § 5 Verification

1. `python3 scripts/build_docs_html.py`
2. Local serve + visual check: landing animation, GQA widget, one doc page
   (collapse, dark chrome, mobile sidebar).
3. Reduced-motion fallback check.
4. `python3 -m pytest tests/` and `python3 tests/test_doc_refs.py` green.
5. Every local HTML link target in the portal resolves.
