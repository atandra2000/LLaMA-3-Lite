# Docs Premium Re-theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-theme the generated documentation portal to a dark, premium, code-centric "Dark Bench Notebook" design with a Breathing-Field ASCII hero animation, an interactive GQA figure, and collapsible long code blocks — without altering any documentation text.

**Architecture:** `docs_html/` is a git-ignored artifact. All changes land in the two sources of truth: `assets/style.css` (design system) and `scripts/build_docs_html.py` (HTML templates + inline JS). The stylesheet is token-driven, so the re-theme is a token-layer swap (dark palette into `:root`), deletion of all light-only / theme-toggle rules, and additive styles for the three new components. The portal is regenerated last.

**Tech Stack:** Python stdlib generator, CSS custom properties, vanilla JS (no new dependencies).

**Spec:** `docs/superpowers/specs/2026-08-17-docs-premium-retheme-design.md`

## Global Constraints

- No Markdown source file is modified; all portal text stays verbatim.
- No new build or runtime dependencies.
- `docs_html/` stays fully static and locally navigable.
- All animation respects `prefers-reduced-motion`.

---

### Task 1: Token swap + font + theme-toggle removal in style.css

**Files:**
- Modify: `assets/style.css` (token block §1, dark block §1, type §3, body rules §2)

- [ ] Replace the light `:root` palette with the Dark Bench Notebook tokens from the spec (`--bg #12100d`, `--bg-2 #161310`, `--bg-3 #1a1712`, `--fg #d8ccb4`, `--fg-faint #5c554a`, `--rule #2c261c`, terracotta `#e07a3f`, olive `#9a9440`), keeping existing token *names* (`--paper`, `--ink`, …) so component rules keep working.
- [ ] Delete the `[data-theme="dark"]` override block and every `[data-theme="dark"] …` selector elsewhere.
- [ ] Switch `--font-prose` to JetBrains Mono (same stack as `--font-mono`); remove serif usage.
- [ ] Fix hardcoded light colors (body::before grid rgba, mobile sidebar shadow).
- [ ] Delete `.theme-toggle` rules; drop it from the shared mono selector list.

### Task 2: Generator chrome — fonts, theme toggle removal

**Files:**
- Modify: `scripts/build_docs_html.py` (`generate_html_page`, `generate_index_portal`)

- [ ] Swap both Google-Fonts links to JetBrains Mono (400/500/600 + italic).
- [ ] Remove `data-theme` attribute handling, the theme-toggle button, and `toggleTheme()` / saved-theme JS from both templates.
- [ ] Keep highlight.js pinned to `github-dark` (no runtime swap).

### Task 3: Breathing Field hero animation (index)

**Files:**
- Modify: `scripts/build_docs_html.py` (`generate_index_portal`, new plain-string JS constant)
- Modify: `assets/style.css` (new `.ascii-hero` styles)

- [ ] Replace the hero SVG block with `<pre class="ascii-hero" id="asciiHero" aria-hidden="true"></pre>`; title/subtitle/coords/spec-sheet untouched.
- [ ] Add plain-string JS constant `HERO_ANIMATION_JS`: llama-mask silhouette, wave-interference shading over `.:░▒▓█`, terracotta crest glints, ~12 fps; static silhouette when `prefers-reduced-motion: reduce`.
- [ ] Add CSS for `.ascii-hero` (size, colors g0–g5 + terracotta glint, mobile scaling).

### Task 4: Interactive GQA wiring widget (index)

**Files:**
- Modify: `scripts/build_docs_html.py` (`generate_index_portal`, new JS constant)
- Modify: `assets/style.css` (new `.gqa-widget` styles)

- [ ] Insert the widget after the spec sheet: 8 Q rows, each wired to KV head `floor(i/2)`; hover/click/focus on a Q row lights the paired KV node and both wires.
- [ ] Build the DOM from a JS data array (plain-string constant) so markup stays terse; keep the "FIG. A1" coordinate-stamp motif.
- [ ] Keyboard accessible (buttons), reduced-motion safe.

### Task 5: Collapsible long code blocks (doc pages)

**Files:**
- Modify: `scripts/build_docs_html.py` (code-block restoration in `parse_markdown_to_html`, page JS)
- Modify: `assets/style.css` (`.code-wrapper.collapsed` + expand button)

- [ ] Fenced blocks with > 14 lines get class `collapsed` and an `expand ▾ · N lines` button in the code header.
- [ ] CSS: collapsed `pre` max-height with fade-out gradient; expanded state restores full height.
- [ ] JS `toggleCode(btn)` toggles the state and the label.

### Task 6: Regenerate, verify, visual QA

**Files:**
- Generate: `docs_html/**`

- [ ] Run `python3 scripts/build_docs_html.py`.
- [ ] Verify every local HTML link target in `docs_html/` resolves.
- [ ] Serve locally and visually check: landing animation, GQA widget, one doc page (collapse, chrome, mobile sidebar), reduced-motion fallback.
- [ ] Run `python3 -m pytest tests/` and `python3 tests/test_doc_refs.py` — both green.
