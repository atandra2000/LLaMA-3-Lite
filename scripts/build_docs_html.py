#!/usr/bin/env python3
"""
LLaMA-3-Lite Documentation Generator
Converts project markdown files into a responsive, beautifully-styled HTML documentation portal
with full LaTeX math (KaTeX) and syntax highlighting support.
Output directory: docs_html/ (ignored by git).

Design system: the "dark bench notebook" — espresso-graphite paper, warm-bone
ink, terracotta + olive marks, one mono voice. Shares its lineage with the
sibling DeepSeek-v3-Lite, Mamba-3-Lite, and GPT-OSS-Lite portals. See `assets/style.css`.
"""

import os
import re
import html
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_DIR / "docs_html"

DOC_FILES = [
    # (relative_path_from_root, category, display_title)
    ("README.md", "Core", "Project Overview (README)"),
    ("AGENTS.md", "Core", "AGENTS & System Architecture"),
    ("SKILLS.md", "Core", "Skills Reference"),
    ("docs/README.md", "Core", "Documentation Index"),
    ("docs/training.md", "Core", "Training, Memory Stack & Data Pipeline"),
    ("docs/AUDIT.md", "Core", "Docs & Codebase Audit"),

    # Concepts
    ("docs/concepts/architecture-components.md", "Concepts", "Architecture Components — Norm, FFN, Loss"),
    ("docs/concepts/attention-and-positional.md", "Concepts", "Attention & Positional Encoding (GQA, RoPE)"),
    ("docs/concepts/data-and-kernels.md", "Concepts", "Data Pipeline & Triton Kernels"),
    ("docs/concepts/training-and-memory.md", "Concepts", "Training, Memory & Numerical Stability"),

    # Guides
    ("docs/guides/quickstart.md", "Guides", "Quickstart — From Zero to a Running Loop"),
    ("docs/guides/learning-paths.md", "Guides", "Learning Paths — How to Read the Docs"),
    ("docs/guides/troubleshooting.md", "Guides", "Troubleshooting — FAQ"),
    ("docs/guides/glossary.md", "Guides", "Glossary — Notation, Acronyms, File Layout"),

    # References
    ("docs/references/model-reference.md", "References", "Model, RoPE & Config Reference"),
    ("docs/references/data-reference.md", "References", "Data, Tokenizer & Kernels Reference"),
    ("docs/references/training-reference.md", "References", "Training & Test Reference"),
    ("docs/references/workspace-data.md", "References", "Workspace Shared Data Pipeline"),
]

# Premium-polish assets: mono-only font link, boot overlay, and shared portal.js
FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400'
             '&display=swap" rel="stylesheet">')

BOOT_OVERLAY_HTML = (
    '<div id="boot-overlay" aria-hidden="true">'
    '<div class="boot-inner">'
    '<div class="boot-wordmark">LLAMA-3-LITE</div>'
    '<div class="boot-line">loading weights '
    '<span class="boot-bar">[░░░░░░░░░░░░] 0%</span>'
    '</div></div></div>'
)

BOOT_SCRIPT = """<script>
(function () {
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduced) document.documentElement.classList.add('booting');
    document.addEventListener('DOMContentLoaded', function () {
        setTimeout(function () {
            document.documentElement.classList.remove('booting');
            var ov = document.getElementById('boot-overlay');
            if (ov && ov.parentNode) ov.parentNode.removeChild(ov);
        }, reduced ? 0 : 350);
    });
})();
</script>"""

# Shared <head> for every generated page.
HEAD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Fonts — secondary mono voice for headings/numerics, JetBrains for body. -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    {font_link}
    {boot_script}
{extra_head}
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
</head>
"""

DOC_EXTRA_HEAD = """    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>"""


def slugify(text: str) -> str:
    """Generate clean HTML id for headings matching GitHub anchor conventions."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s', '-', text)
    return text.strip('-') or "heading"


@lru_cache(maxsize=1)
def github_base_url() -> str:
    """Derive the GitHub blob base (https://github.com/<owner>/<repo>/blob/<branch>)."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
        out = out.replace("git@github.com:", "https://github.com/").removesuffix(".git")
        if not out.startswith("https://github.com/"):
            return ""
    except Exception:
        return ""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, check=True, cwd=WORKSPACE_DIR,
        ).stdout.strip()
    except Exception:
        return ""
    return f"{out}/blob/{branch}" if branch else ""


def fix_md_links(content: str, src_rel_path: str) -> str:
    """Rewrite relative markdown links for the HTML build."""
    repo_base = github_base_url()
    src_dir = WORKSPACE_DIR / Path(src_rel_path).parent

    def link_replacer(match):
        label = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return f"[{label}]({url})"
        path_part, _, anchor = url.partition("#")
        if path_part.endswith(".md"):
            if path_part.startswith("/"):
                repo_rel = Path(path_part[1:])
            else:
                cand_src = src_dir / path_part
                cand_root = WORKSPACE_DIR / path_part
                if cand_src.exists():
                    repo_rel = cand_src.resolve().relative_to(WORKSPACE_DIR)
                elif cand_root.exists():
                    repo_rel = cand_root
                else:
                    repo_rel = Path(src_rel_path).parent / path_part
            rel = os.path.relpath(WORKSPACE_DIR / repo_rel, src_dir).replace(os.sep, '/')
            target = rel[:-3] + ".html"
            if anchor:
                target += "#" + anchor
            return f"[{label}]({target})"
        if repo_base and not path_part.startswith("/"):
            try:
                repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
                return f"[{label}]({repo_base}/{repo_rel})"
            except ValueError:
                pass
        return f"[{label}]({url})"

    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)


def parse_markdown_to_html(md_text: str, src_rel_path: str) -> tuple[str, list[dict]]:
    """Statically convert markdown to rich HTML structure with full LaTeX & Math protection.

    Returns (html_content, toc_items).
    """
    md_text = fix_md_links(md_text, src_rel_path)

    # STEP 1: Protect Code Blocks & Inline Code
    code_blocks = []
    def store_code_block(m):
        code_blocks.append(m.group(0))
        return f"\n\n___CODEBLOCK_{len(code_blocks)-1}___\n\n"

    md_text = re.sub(r'```[\s\S]*?```', store_code_block, md_text)

    inline_codes = []
    def store_inline_code(m):
        inline_codes.append(m.group(0))
        return f"___INLINECODE_{len(inline_codes)-1}___"

    md_text = re.sub(r'`[^`\n]+`', store_inline_code, md_text)

    # STEP 2: Protect LaTeX Math Blocks & Inline Math
    display_maths = []
    def store_display_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        display_maths.append(f'<div class="math-block">$$\n{safe_math}\n$$</div>')
        return f"\n\n___DISPLAYMATH_{len(display_maths)-1}___\n\n"

    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', store_display_math, md_text)
    md_text = re.sub(r'\\\[([\s\S]+?)\\\]', store_display_math, md_text)

    inline_maths = []
    def store_inline_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        inline_maths.append(f'<span class="math-inline">${safe_math}$</span>')
        return f"___INLINEMATH_{len(inline_maths)-1}___"

    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', store_inline_math, md_text)
    md_text = re.sub(r'\\\(([\s\S]+?)\\\)', store_inline_math, md_text)

    # STEP 3: Parse Document Structure Line by Line
    toc = []
    seen_slugs = {}
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    table_headers = []
    table_rows = []

    list_stack = []
    h1_seen = False

    in_blockquote = False
    blockquote_type = "normal"
    blockquote_lines = []

    def close_li():
        nonlocal list_stack
        if list_stack and list_stack[-1]['li_open']:
            html_lines.append("</li>")
            list_stack[-1]['li_open'] = False

    def flush_list():
        nonlocal list_stack
        while list_stack:
            close_li()
            html_lines.append(f"</{list_stack[-1]['tag']}>")
            list_stack.pop()

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_type, blockquote_lines
        if in_blockquote:
            content = "<br>".join(blockquote_lines)
            if blockquote_type != "normal":
                title = blockquote_type.upper()
                icon = {"NOTE": "ℹ️", "TIP": "💡", "IMPORTANT": "📌", "WARNING": "⚠️", "CAUTION": "🚨"}.get(title, "ℹ️")
                html_lines.append(
                    f'<div class="callout callout-{blockquote_type.lower()}">'
                    f'<div class="callout-header"><span class="callout-icon">{icon}</span><span class="callout-title">{title}</span></div>'
                    f'<div class="callout-body">{content}</div>'
                    f'</div>'
                )
            else:
                html_lines.append(f'<blockquote>{content}</blockquote>')
            in_blockquote = False
            blockquote_type = "normal"
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if in_table:
            th_html = "".join(f"<th>{h}</th>" for h in table_headers)
            tr_html = ""
            for row in table_rows:
                td_html = "".join(f"<td>{c}</td>" for c in row)
                tr_html += f"<tr>{td_html}</tr>"
            html_lines.append(
                f'<div class="table-container"><table class="doc-table">'
                f'<thead><tr>{th_html}</tr></thead>'
                f'<tbody>{tr_html}</tbody>'
                f'</table></div>'
            )
            in_table = False
            table_headers = []
            table_rows = []

    def escape_preserving_entities(text: str) -> str:
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        return text.replace('<', '&lt;').replace('>', '&gt;')

    def render_inline_formatting(text: str) -> str:
        text = escape_preserving_entities(text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("___DISPLAYMATH_") or stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if not stripped:
            flush_table()
            flush_list()
            flush_blockquote()
            i += 1
            continue

        # Blockquote or Callout
        if stripped.startswith(">"):
            flush_table()
            flush_list()
            bq_content = stripped.lstrip(">").strip()
            callout_match = re.match(r'^\[\!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]', bq_content, re.IGNORECASE)
            if callout_match:
                in_blockquote = True
                blockquote_type = callout_match.group(1).upper()
                remaining = bq_content[callout_match.end():].strip()
                if remaining:
                    blockquote_lines.append(render_inline_formatting(remaining))
            else:
                if not in_blockquote:
                    in_blockquote = True
                    blockquote_type = "normal"
                if bq_content:
                    blockquote_lines.append(render_inline_formatting(bq_content))
            i += 1
            continue

        # Horizontal Rule
        if re.match(r'^(---|\*\*\*|___)\s*$', stripped):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append("<hr class='doc-hr'>")
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()

            clean_title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text_raw)
            clean_title = re.sub(r'`([^`]+)`', r'\1', clean_title)
            clean_title = re.sub(r'___INLINECODE_\d+___', '', clean_title)
            clean_title = re.sub(r'___INLINEMATH_\d+___', '', clean_title)
            raw_slug = slugify(clean_title)
            if raw_slug in seen_slugs:
                seen_slugs[raw_slug] += 1
                heading_id = f"{raw_slug}-{seen_slugs[raw_slug]}"
            else:
                seen_slugs[raw_slug] = 0
                heading_id = raw_slug
            rendered_heading = render_inline_formatting(heading_text_raw)

            if level in (2, 3):
                toc.append({'level': level, 'title': clean_title, 'id': heading_id})

            if level == 1 and not h1_seen:
                h1_seen = True
                html_lines.append(f'<span class="doc-anchor" id="{heading_id}"></span>')
            else:
                html_lines.append(
                    f'<h{level} id="{heading_id}" class="heading-anchor">'
                    f'{rendered_heading}'
                    f'<a href="#{heading_id}" class="anchor-link" aria-label="Link to section">#</a>'
                    f'</h{level}>'
                )
            i += 1
            continue

        # Markdown Table Detection
        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?---', lines[i + 1].strip()):
            flush_list()
            flush_blockquote()
            in_table = True
            headers_raw = [c.strip() for c in line.strip().strip("|").split("|")]
            table_headers = [render_inline_formatting(h) for h in headers_raw]
            i += 2

            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        # Lists
        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            tag = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            indent = len(line) - len(line.lstrip(' '))

            while list_stack and indent < list_stack[-1]['indent']:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if list_stack and list_stack[-1]['indent'] == indent and list_stack[-1]['tag'] != tag:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            if not list_stack or list_stack[-1]['indent'] != indent:
                list_stack.append({'indent': indent, 'tag': tag, 'li_open': False})
                html_lines.append(f'<{tag} class="doc-list">')
            else:
                close_li()

            task_match = re.match(r'^\[([ xX])\]\s+(.+)$', item_text)
            if task_match:
                checked = 'checked' if task_match.group(1).lower() == 'x' else ''
                item_content = render_inline_formatting(task_match.group(2))
                html_lines.append(f'<li class="task-item"><input type="checkbox" disabled {checked}> {item_content}')
            else:
                html_lines.append(f'<li>{render_inline_formatting(item_text)}')
            list_stack[-1]['li_open'] = True
            i += 1
            continue

        # Continuation of an open list item
        if list_stack and list_stack[-1]['li_open'] and line[:1] in (' ', '\t'):
            html_lines.append("<br> " + render_inline_formatting(stripped))
            i += 1
            continue

        # Standard Paragraph
        flush_table()
        flush_list()
        flush_blockquote()
        html_lines.append(f'<p>{render_inline_formatting(stripped)}</p>')
        i += 1

    flush_table()
    flush_list()
    flush_blockquote()

    full_html = "\n".join(html_lines)

    # STEP 4: Restore Protected Tokens
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    for idx, raw_code in enumerate(inline_codes):
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.lstrip("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)

        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'

        n_lines = len(lines_b) - 2
        collapsed = n_lines > 14
        wrapper_cls = 'code-wrapper collapsed' if collapsed else 'code-wrapper'
        expand_btn = (
            f'<button class="expand-btn" data-label="expand ▾ · {n_lines} lines" '
            f'onclick="toggleCode(this)">expand ▾ · {n_lines} lines</button>'
            if collapsed else ''
        )

        block_html = (
            f'<div class="{wrapper_cls}" data-lines="{n_lines}">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<span class="code-actions">{expand_btn}'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></span>'
            f'</div>'
            f'<pre><code{lang_attr}>{escaped_content}</code></pre>'
            f'</div>'
        )
        full_html = full_html.replace(f"___CODEBLOCK_{idx}___", block_html)

    return full_html, toc


def compute_rel_prefix(target_rel_path: str) -> str:
    """Calculate relative path back to root docs_html directory."""
    parts = Path(target_rel_path).parts
    if len(parts) <= 1:
        return "./"
    return "../" * (len(parts) - 1)


def build_sidebar_html(current_rel_path: str, rel_prefix: str) -> str:
    """Build the navigation sidebar HTML matching sibling layout."""
    sidebar_sections = {"Core": [], "Concepts": [], "Guides": [], "References": []}

    for rel_path, category, display_title in DOC_FILES:
        target_html_rel = rel_path.replace(".md", ".html")
        href = rel_prefix + target_html_rel
        is_active = (rel_path == current_rel_path) or (rel_path.replace(".md", ".html") == current_rel_path)
        active_cls = "active" if is_active else ""
        sidebar_sections[category].append(
            f'<li class="nav-item"><a href="{href}" class="nav-link {active_cls}" title="{display_title}"><span class="nav-link-text">{display_title}</span></a></li>'
        )

    html_out = ['<div class="sidebar-search"><input type="text" id="navSearch" placeholder="Search docs..." onkeyup="filterNav()"></div>']

    for cat_name, items in sidebar_sections.items():
        if items:
            html_out.append('<div class="nav-group">')
            html_out.append(f'<div class="nav-group-title">{cat_name}</div>')
            html_out.append(f'<ul class="nav-list">{"".join(items)}</ul>')
            html_out.append('</div>')

    return "\n".join(html_out)


def build_toc_html(toc_items: list[dict]) -> str:
    """Build the right sidebar table of contents."""
    if not toc_items:
        return '<div class="toc-empty">No section headings</div>'

    toc_links = []
    for item in toc_items:
        indent_cls = "toc-h3" if item['level'] == 3 else "toc-h2"
        toc_links.append(f'<li class="{indent_cls}"><a href="#{item["id"]}" class="toc-link">{item["title"]}</a></li>')

    return f'<ul class="toc-list">{"".join(toc_links)}</ul>'


def generate_html_page(rel_path: str, category: str, display_title: str):
    """Generate single HTML file for a markdown document."""
    src_file = WORKSPACE_DIR / rel_path
    if not src_file.exists():
        print(f"Warning: {src_file} does not exist, skipping.")
        return

    md_text = src_file.read_text(encoding="utf-8")
    word_count = len(re.findall(r'\b\w+\b', md_text))
    reading_time = max(1, round(word_count / 220))

    html_body, toc_items = parse_markdown_to_html(md_text, rel_path)
    rel_prefix = compute_rel_prefix(rel_path)
    sidebar_html = build_sidebar_html(rel_path, rel_prefix)
    toc_html = build_toc_html(toc_items)

    curr_idx = -1
    for idx, (p, _, _) in enumerate(DOC_FILES):
        if p == rel_path:
            curr_idx = idx
            break

    prev_doc = DOC_FILES[curr_idx - 1] if curr_idx > 0 else None
    next_doc = DOC_FILES[curr_idx + 1] if 0 <= curr_idx < len(DOC_FILES) - 1 else None

    prev_html = ""
    if prev_doc:
        prev_href = rel_prefix + prev_doc[0].replace(".md", ".html")
        prev_html = f'<a href="{prev_href}" class="nav-card prev-card"><span class="card-label">← Previous</span><span class="card-title">{prev_doc[2]}</span></a>'
    next_html = ""
    if next_doc:
        next_href = rel_prefix + next_doc[0].replace(".md", ".html")
        next_html = f'<a href="{next_href}" class="nav-card next-card"><span class="card-label">Next →</span><span class="card-title">{next_doc[2]}</span></a>'

    page_html = HEAD_TEMPLATE.format(
        rel_prefix=rel_prefix,
        title=f"{display_title} | LLaMA-3-Lite Documentation",
        extra_head=DOC_EXTRA_HEAD,
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body>
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="{rel_prefix}index.html" class="brand-logo">
                <span class="brand-name">LLaMA-3-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="{rel_prefix}index.html" class="header-link">Portal</a>
            <a href="{rel_prefix}README.html" class="header-link">README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Content Area -->
        <main class="main-content">
            <div class="content-container">
                <div class="breadcrumb">
                    <a href="{rel_prefix}index.html">Docs</a> &gt; <span>{category}</span> &gt; <span class="current">{display_title}</span>
                </div>

                <div class="doc-header">
                    <h1 class="doc-title">{display_title}</h1>
                    <div class="doc-meta">
                        <span class="meta-item"><span class="meta-mark">&sect;</span> {rel_path}</span>
                        <span class="meta-item"><span class="meta-mark">&para;</span> {word_count:,} words</span>
                        <span class="meta-item"><span class="meta-mark">&tau;</span> ~{reading_time} min read</span>
                    </div>
                </div>

                <article class="markdown-body" id="articleBody">
                    {html_body}
                </article>

                <div class="doc-footer-nav">
                    {prev_html}
                    {next_html}
                </div>
            </div>
        </main>

        <!-- Right Sidebar Table of Contents -->
        <aside class="toc-sidebar">
            <div class="toc-inner">
                <div class="toc-title">On This Page</div>
                {toc_html}
            </div>
        </aside>
    </div>

    <!-- Scripts -->
    <script defer src="{rel_prefix}assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / rel_path.replace(".md", ".html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page_html, encoding="utf-8")


def generate_index_portal():
    """Generate interactive index.html home portal."""
    sidebar_html = build_sidebar_html("index.html", "./")

    categories = {
        ("CORE", "Core Architecture"): [
            ("README.html", "README", "Project Overview", "513.8M-param pure-PyTorch LLaMA-3 reproduction with an 8-technique memory stack (78% VRAM reduction)."),
            ("AGENTS.html", "AGENTS", "System Architecture", "Codebase contracts, hard rules, Triton kernel carve-out, and memory optimization rules."),
            ("SKILLS.html", "SKILLS", "Skills Map", "Specialized developer workflows, memory engineering tools, and agent competencies."),
            ("docs/README.html", "DOCS", "Documentation Index", "A map of the concepts, guides, and API references in this portal."),
            ("docs/training.html", "CORE", "Training Pipeline", "Corpus mix, 8-technique memory stack, chunked cross-entropy, and training loop."),
            ("docs/AUDIT.html", "CORE", "Docs & Code Audit", "Verified architectural constants, symbol resolution, and codebase invariants."),
        ],
        ("CONCEPTS", "Architecture & Concepts"): [
            ("docs/concepts/architecture-components.html", "C1", "Architecture Components", "RMSNorm pre-norm, SwiGLU FFN (4096d), chunked cross-entropy, and z-loss formulation."),
            ("docs/concepts/attention-and-positional.html", "C2", "Attention & RoPE", "Grouped-Query Attention (8Q/4KV) and RoPE θ=500K for long-context extrapolation up to 8192."),
            ("docs/concepts/data-and-kernels.html", "C3", "Data & Triton Kernels", "Disk-backed uint32 token cache, async prefetching, and sanctioned Triton kernel opt-ins."),
            ("docs/concepts/training-and-memory.html", "C4", "Training & Memory", "The 8-technique memory stack (92 GB → 20 GB), gradient checkpointing, and numerical stability."),
        ],
        ("GUIDES", "Guides & Playbooks"): [
            ("docs/guides/quickstart.html", "G0", "Quickstart", "From zero to a running training loop — installation, verification, synthetic smoke runs."),
            ("docs/guides/learning-paths.html", "G1", "Learning Paths", "How to read the docs based on engineering role (Systems, Researcher, Developer)."),
            ("docs/guides/troubleshooting.html", "G2", "Troubleshooting", "Common failure modes, CUDA OOM mitigation, NaN prevention, and loss divergence."),
            ("docs/guides/glossary.html", "G3", "Glossary", "Notation, acronyms, tensor shapes, and workspace file layout reference."),
        ],
        ("REFS", "API References"): [
            ("docs/references/model-reference.html", "R1", "Model Reference", "LLaMA3Transformer, Block, GQA, SwiGLU, RMSNorm, and RoPE APIs and tensor signatures."),
            ("docs/references/data-reference.html", "R2", "Data Reference", "Dataset, Tokenizer, Sharding, and custom Triton kernel contracts."),
            ("docs/references/training-reference.html", "R3", "Training Reference", "Train loop, optimizer, cosine LR scheduler, and checkpoint manager."),
            ("docs/references/workspace-data.html", "R4", "Workspace Data", "Universal 8B-token data pipeline (shared_data), sharding, and exact dedup."),
        ],
    }

    portal_cards_html = ""
    for (cat_tag, cat_title), items in categories.items():
        cards = ""
        for href, tag, title, desc in items:
            cards += f"""
            <a href="{href}" class="portal-card">
                <span class="card-tag">{tag}</span>
                <div class="card-body">
                    <h3 class="card-heading">{title}</h3>
                    <p class="card-desc">{desc}</p>
                </div>
            </a>
            """
        portal_cards_html += f"""
        <section class="portal-section">
            <header class="portal-section-head">
                <span class="portal-section-mark">&sect; {cat_tag.lower()}</span>
                <h2 class="portal-section-title">{cat_title}</h2>
                <span class="portal-section-meta">{len(items)} entries</span>
            </header>
            <div class="portal-grid">{cards}</div>
        </section>
        """

    index_html = HEAD_TEMPLATE.format(
        title="LLaMA-3-Lite Documentation Portal",
        extra_head="",
        rel_prefix="./",
        font_link=FONT_LINK,
        boot_script=BOOT_SCRIPT,
    ) + f"""<body class="index-portal">
    {BOOT_OVERLAY_HTML}
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="index.html" class="brand-logo">
                <span class="brand-name">LLaMA-3-Lite</span>
                <span class="brand-badge">Documentation</span>
            </a>
        </div>
        <div class="header-right">
            <a href="README.html" class="header-link">GitHub README</a>
        </div>
    </header>

    <div class="app-layout">
        <!-- Left Sidebar Navigation -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-inner">
                {sidebar_html}
            </div>
        </aside>

        <!-- Main Portal Content -->
        <main class="main-content">
            <div class="content-container">
                <div class="hero-banner">
                    <div class="hero-margin-ticks" aria-hidden="true"></div>

                    <div class="hero-coords" aria-hidden="true">
                        <span class="coord">FIG &middot; A0</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">PARAM 513.8M</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">GQA 8Q&middot;4KV</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">SWIGLU 4096</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">RMSNORM 1024</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">ROPE &theta;=500K</span>
                        <span class="coord-sep">/</span>
                        <span class="coord">78% VRAM CUT</span>
                    </div>
                    <h1 class="hero-title">LLaMA<span class="hero-title-em">-3</span><span class="hero-title-em-accent">-Lite</span><span class="hero-title sr-only" data-title="LLAMA-3-LITE"> — documentation portal</span></h1>
                    <p class="hero-subtitle">From-scratch PyTorch reproduction of LLaMA-3 at ~513.8M params — Grouped-Query Attention (8Q/4KV), RoPE &theta;=500K, fused SwiGLU, and the 8-technique memory stack achieving 78% peak VRAM reduction (92 GB &rarr; 20 GB). Pure PyTorch with three sanctioned Triton opt-in kernels. Read it like a field notebook: a name, a wiring sketch, then the measurements.</p>

                    <div class="llama-telemetry-ribbon" aria-label="Key LLaMA-3 Architectural Metrics">
                        <div class="telemetry-card terra">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> CHUNKED CE</div>
                            <div class="tc-val">99.4% <span class="unit">LOGIT CUT</span></div>
                            <div class="tc-desc">Logits 50.3 GB BF16 &rarr; 0.3 GB via 256-row streaming slices + z-loss</div>
                        </div>
                        <div class="telemetry-card olive">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> MEMORY STACK</div>
                            <div class="tc-val">78% <span class="unit">VRAM SAVED</span></div>
                            <div class="tc-desc">92 GB &rarr; 20 GB peak activation &amp; gradient cut allowing batch 96 on 1&times; A100</div>
                        </div>
                        <div class="telemetry-card gold">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> GQA 8Q / 4KV</div>
                            <div class="tc-val">2.0&times; <span class="unit">KV CACHE CUT</span></div>
                            <div class="tc-desc">4 KV heads shared across 8 Query heads; halving KV projection params &amp; cache</div>
                        </div>
                        <div class="telemetry-card ink">
                            <div class="tc-badge"><span class="tc-badge-dot"></span> ROPE &theta;=500K</div>
                            <div class="tc-val">50&times; <span class="unit">BASE EXP</span></div>
                            <div class="tc-desc">&theta;=500,000 high-frequency rotary carrier enables 8K+ context extrapolation</div>
                        </div>
                    </div>

                    <div class="hero-figure" id="hero-decode" data-title="LLAMA-3-LITE" data-sub="513.8M &middot; GQA 8Q/4KV &middot; SwiGLU 4096 &middot; RoPE &theta;=500K &middot; 78% VRAM Cut" role="region" aria-label="Figure A0: Grouped-Query Attention (8Q/4KV) and RoPE &theta;=500K Phase-Space Dynamics">
                        <div class="hero-figure-header">
                            <div class="fig-badge">
                                <span class="fig-dot" aria-hidden="true"></span>
                                <span class="fig-tag">FIG. A0</span>
                                <span class="fig-sep" aria-hidden="true">/</span>
                                <span class="fig-title">GQA 8Q / 4KV ATTENTION &amp; ROPE &theta;=500K</span>
                                <span class="fig-dim">8Q / 4KV &middot; 128 DIMS &middot; &theta;=500,000</span>
                            </div>
                            <div class="fig-controls">
                                <button class="fig-btn active" data-mode="gqa" type="button" title="Grouped-Query Attention (8 Queries to 4 KV Heads)">GQA &middot; ATTENTION</button>
                                <button class="fig-btn" data-mode="rope" type="button" title="RoPE &theta;=500K Rotary Frequency Phasors">RoPE &middot; PHASORS</button>
                                <button class="fig-btn" data-mode="chunkce" type="button" title="Chunked Cross-Entropy 256-Row Slices">CHUNKED &middot; CE</button>
                                <button class="fig-btn" data-mode="memory" type="button" title="8-Technique Memory Stack Allocation">MEMORY &middot; STACK</button>
                                <button class="fig-btn fig-btn-icon" id="heroSpeedBtn" type="button" title="Simulation Speed" aria-label="Speed: 1x">1&times;</button>
                                <button class="fig-btn fig-btn-icon" id="heroPauseBtn" type="button" title="Pause / Resume" aria-label="Pause simulation">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="fig-canvas-wrap">
                            <canvas id="heroStateCanvas" class="hero-canvas" aria-hidden="true"></canvas>
                            <div class="fig-overlay-hud" aria-hidden="true">
                                <div class="hud-corner top-left">
                                    <span class="hud-lbl">SIMULATION MODE</span>
                                    <span class="hud-val" id="hudModeLabel">GQA 8Q / 4KV ATTENTION MANIFOLD</span>
                                </div>
                                <div class="hud-corner top-right">
                                    <span class="hud-lbl">KV CACHE REDUCTION</span>
                                    <span class="hud-val" id="hudKVCut">2.0&times; <span class="unit">(128 &rarr; 64 MiB/seq)</span></span>
                                </div>
                                <div class="hud-corner bottom-left">
                                    <span class="hud-lbl">HEAD ROUTING</span>
                                    <span class="hud-val" id="hudGQARouting">8 QUERIES <span class="num">(2:1)</span> &rarr; 4 KV HEADS</span>
                                </div>
                                <div class="hud-corner bottom-right">
                                    <span class="hud-lbl">BASE FREQUENCY</span>
                                    <span class="hud-val" id="hudRopeFreq">&theta; = 500,000 <span class="unit">RAD/POS</span></span>
                                </div>
                                <div class="hud-probe" id="heroProbeHUD" style="opacity: 0;">
                                    <div class="probe-card">
                                        <span class="probe-tag" id="probeTag">PROBE &middot; HEAD #00</span>
                                        <span class="probe-val" id="probeCoords">Query Head Q0 &rarr; KV Head KV0 [128 dims]</span>
                                        <span class="probe-sub" id="probeDecay">Attention Activity: 84.2%</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="hero-figure-footer">
                            <div class="fig-formula" id="heroFormulaBar">
                                <span class="formula-sym">head<sub class="f-sub">i</sub></span>
                                <span class="formula-op">=</span>
                                <span class="formula-term term-a">Softmax(Q<sub class="f-sub">i</sub> K<sub class="f-sub">&lfloor;i/2&rfloor;</sub><sup class="f-sub">&top;</sup> / &radic;d<sub class="f-sub">k</sub>)</span>
                                <span class="formula-dot">&middot;</span>
                                <span class="formula-term term-b">V<sub class="f-sub">&lfloor;i/2&rfloor;</sub></span>
                                <span class="formula-op">&middot;</span>
                                <span class="formula-sym">RoPE(x<sub class="f-sub">m</sub>, m, &theta;)</span>
                                <span class="formula-op">=</span>
                                <span class="formula-term">R<sub class="f-sub">&theta;,m</sub><sup class="f-sub">d</sup> x<sub class="f-sub">m</sub></span>
                            </div>
                            <div class="fig-legend">
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">8 Query Heads (Q<sub class="f-sub">0..7</sub>)</span></span>
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">4 Shared KV Heads (KV<sub class="f-sub">0..3</sub>)</span></span>
                                <span class="legend-item"><span class="legend-swatch gold"></span> <span class="legend-label">RoPE &theta;=500K Phasors</span></span>
                                <span class="legend-hint">CLICK: SHOCKWAVE &middot; HOVER: PROBE HUD</span>
                            </div>
                        </div>
                    </div>

                    <div class="spec-sheet">
                        <div class="spec-sheet-rule" aria-hidden="true">
                            <span class="spec-rule-key">DATASHEET</span>
                            <span class="spec-rule-meta">rev 0.1 &middot; chk bf16 &middot; 8.25B tok / A100-80</span>
                        </div>
                        <dl class="spec-grid">
                            <div class="spec-cell"><dt class="spec-key">01 &middot; params</dt><dd class="spec-val">513.8<span class="unit">M</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">02 &middot; layers</dt><dd class="spec-val">16<span class="unit"> blocks</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">03 &middot; attention</dt><dd class="spec-val">8Q / 4KV<span class="unit"> GQA</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">04 &middot; hidden</dt><dd class="spec-val">1024<span class="unit"> d_model</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">05 &middot; ffn</dt><dd class="spec-val">4096<span class="unit"> SwiGLU</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">06 &middot; vocab</dt><dd class="spec-val">128<span class="unit"> K</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">07 &middot; ctx &amp; rope</dt><dd class="spec-val">2048<span class="unit"> &theta;=500K</span></dd></div>
                            <div class="spec-cell"><dt class="spec-key">08 &middot; memory cut</dt><dd class="spec-val">78%<span class="unit"> 92&rarr;20GB</span></dd></div>
                        </dl>
                    </div>

                    <div class="mechanism-section" aria-label="Interactive LLaMA-3 Core Mechanisms">
                        <div class="mechanism-section-head">
                            <span class="mechanism-section-title">&sect; LLAMA-3 CORE MECHANISMS &middot; INTERACTIVE BENCHMARK LABS</span>
                        </div>
                        <div class="mechanism-grid">
                            <!-- Card 1: GQA 8Q/4KV Compression & Interactive Head Router -->
                            <div class="mechanism-card card-gqa" id="mchCardGqa">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">01 &middot; GQA COMPRESSION</span>
                                    <span class="mch-title">8Q / 4KV Cache Footprint Analyzer</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Compare KV cache footprint between standard Multi-Head Attention (8 Q / 8 KV) and Grouped-Query Attention (8 Q / 4 KV) across context lengths with interactive head routing.</p>
                                    <div class="mch-control-row">
                                        <span>Context Length:</span>
                                        <strong id="gqaContextLabel">32,768 tokens</strong>
                                    </div>
                                    <input type="range" id="gqaContextSlider" class="mch-slider" min="2048" max="131072" step="2048" value="32768" aria-label="Context length in tokens">
                                    <canvas id="gqaRouterCanvas" class="mch-canvas" width="280" height="120" aria-hidden="true"></canvas>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Standard MHA (8 KV &times; 128d &times; 16L):</span>
                                            <span id="statMhaVram">2.00 GB</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>LLaMA-3 GQA (4 KV &times; 128d &times; 16L):</span>
                                            <span id="statGqaVram">1.00 GB</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>KV Cache Saved:</span>
                                            <span id="statGqaSaved">1.00 GB (2.00&times; cut &middot; 50% savings)</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="gqaRouteBtn">&#9654; Cycle Query Head Routing (Q0..Q7 &rarr; KV0..KV3)</button>
                                </div>
                            </div>

                            <!-- Card 2: Chunked Cross-Entropy 256-Token Logit Slicer & SRAM Simulator -->
                            <div class="mechanism-card card-chunkce" id="mchCardChunkCe">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">02 &middot; CHUNKED CE</span>
                                    <span class="mch-title">256-Token Slicer &amp; SRAM Simulator</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Avoid materializing the full 196,608 &times; 128,000 logit tensor (50.3 GB BF16). Slicing into 256-token chunks keeps SRAM tile under 0.20 GB with streaming z-loss regularization.</p>
                                    <div class="mch-control-row">
                                        <span>Slice Chunk Size:</span>
                                        <strong id="chunkSizeLabel">256 Tokens</strong>
                                    </div>
                                    <input type="range" id="chunkSizeSlider" class="mch-slider" min="64" max="512" step="64" value="256" aria-label="Chunk size in tokens">
                                    <div class="chunk-pipeline-box" id="chunkStreamDisplay">
                                        <div class="chunk-stage">
                                            <span class="chunk-badge scan">FULL TENSOR</span>
                                            <span>196k &times; 128k = 50.3 GB BF16 (OOM Crash)</span>
                                        </div>
                                        <div class="chunk-stage">
                                            <span class="chunk-badge chunk">SRAM TILE</span>
                                            <span id="chunkTileText">256 &times; 128k = 65.5 MB SRAM Resident Tile</span>
                                        </div>
                                        <div class="chunk-stage">
                                            <span class="chunk-badge state">Z-LOSS REG</span>
                                            <span>L = L_ce + 1e-4 &middot; log&sup2;(Z) &middot; Backprop Stream</span>
                                        </div>
                                    </div>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Peak Logit Memory:</span>
                                            <span id="chunkPeakMem">65.5 MB SRAM (vs 50.3 GB Full)</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>Logit Memory Reduction:</span>
                                            <span>99.4% VRAM Cut</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>Batch Capacity on 1&times; A100-80GB:</span>
                                            <span>Batch 96 with 4&times; Headroom</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="chunkStreamBtn">&#9654; Stream 256-Token Slice &amp; Verify Gradients</button>
                                </div>
                            </div>

                            <!-- Card 3: RoPE θ=500K Extrapolation Explorer & Phase Stability Lab -->
                            <div class="mechanism-card card-rope" id="mchCardRope">
                                <div class="mechanism-card-head">
                                    <span class="mch-tag">03 &middot; ROPE &theta;=500K</span>
                                    <span class="mch-title">Long-Context Extrapolation Lab</span>
                                </div>
                                <div class="mechanism-card-body">
                                    <p class="mch-explainer">Compare base frequency &theta;=500,000 (LLaMA-3) vs standard &theta;=10,000. Ultra-high base frequency prevents phase wrapping and preserves high-frequency relative position discrimination at 8K+ context.</p>
                                    <div class="mch-control-row">
                                        <span>Sequence Position:</span>
                                        <strong id="ropePosLabel">Position: 2048 / 8192</strong>
                                    </div>
                                    <input type="range" id="ropePosSlider" class="mch-slider" min="0" max="8192" step="256" value="2048" aria-label="Sequence position">
                                    <canvas id="ropeExtrapCanvas" class="mch-canvas" width="280" height="120" aria-hidden="true"></canvas>
                                    <div class="mch-stat-box">
                                        <div class="msb-row">
                                            <span>Base Frequency &theta;:</span>
                                            <span>500,000 rad/pos (50&times; Expansion)</span>
                                        </div>
                                        <div class="msb-row">
                                            <span>Phase Stability at Pos 8192:</span>
                                            <span id="ropePhaseStatus">STABLE (No aliasing)</span>
                                        </div>
                                        <div class="msb-row highlight">
                                            <span>Context Extension:</span>
                                            <span>Zero Fine-Tuning Extrapolation to 8K+</span>
                                        </div>
                                    </div>
                                    <button type="button" class="mch-action-btn" id="ropeTestBtn">&#9654; Rotate Sequence Coordinates (Step +1024)</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="pass-widget" id="passWidget" role="region" aria-label="Figure A1: Complete 16-Layer Training Step Pipeline — Forward Activations, Autograd Backward Pass, Gradient Checkpointing Re-computation, and AdamW Step">
                        <div class="pass-widget-header">
                            <div class="pass-badge">
                                <span class="pass-dot" aria-hidden="true"></span>
                                <span class="pass-tag">FIG. A1</span>
                                <span class="pass-sep" aria-hidden="true">/</span>
                                <span class="pass-title">FULL TRAINING STEP &amp; MEMORY PIPELINE</span>
                                <span class="pass-dim">16 LAYERS &middot; FORWARD &middot; AUTOGRAD &middot; ADAMW</span>
                            </div>
                            <div class="pass-controls">
                                <button class="pass-btn active" data-phase="cycle" type="button" title="Full Forward / Backward / AdamW Cycle">AUTO CYCLE</button>
                                <button class="pass-btn" data-phase="forward" type="button" title="Inspect Forward Activation Flow">FORWARD</button>
                                <button class="pass-btn" data-phase="backward" type="button" title="Inspect Backward Gradient Flow">BACKWARD</button>
                                <button class="pass-btn pass-btn-icon" id="passPauseBtn" type="button" title="Pause / Resume Pipeline" aria-label="Pause pipeline">&#10074;&#10074;</button>
                            </div>
                        </div>

                        <div class="pass-canvas-wrap">
                            <canvas id="passDiagramCanvas" class="pass-canvas" aria-hidden="true"></canvas>
                            <div class="pass-overlay-hud" aria-hidden="true">
                                <div class="pass-hud-chip top-left">
                                    <span class="ph-lbl">CURRENT PHASE</span>
                                    <span class="ph-val" id="phCurrentPhase">AUTO CYCLE (FWD &rarr; BWD &rarr; ADAMW)</span>
                                </div>
                                <div class="pass-hud-chip top-right">
                                    <span class="ph-lbl">MEMORY CONTRACT</span>
                                    <span class="ph-val" id="phStepMetrics">8-TECHNIQUE STACK &middot; 78% VRAM SAVED (92&rarr;20GB)</span>
                                </div>
                                <div class="pass-stage-tooltip" id="passStageTooltip" style="opacity: 0;">
                                    <div class="st-card">
                                        <span class="st-tag" id="stTag">STAGE 03: GQA + ROPE</span>
                                        <span class="st-op" id="stOp">Attention(Q_8, K_4, V_4) &middot; RoPE &theta;=500K</span>
                                        <span class="st-shape" id="stShape">Tensor: Q[B, 8, 2048, 128], KV[B, 4, 2048, 128]</span>
                                        <span class="st-desc" id="stDesc">2&times; KV cache compression; RoPE &theta;=500K long context</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="pass-widget-footer">
                            <div class="pass-status-ticker">
                                <span class="ticker-beacon" id="passTickerBeacon">&bull;</span>
                                <span class="ticker-text" id="passTickerText">FORWARD &middot; Tokens x<sub class="f-sub">t</sub> &rarr; Untied Embed &rarr; 16&times; (RMSNorm &middot; GQA &middot; SwiGLU) &rarr; Chunked CE Loss (256 Slices) &rarr; AdamW</span>
                            </div>
                            <div class="pass-legend">
                                <span class="legend-item"><span class="legend-swatch terra"></span> <span class="legend-label">Forward Activations</span></span>
                                <span class="legend-item"><span class="legend-swatch olive"></span> <span class="legend-label">Backward Gradients (&part;&ell;/&part;&theta;)</span></span>
                                <span class="legend-item"><span class="legend-swatch gold"></span> <span class="legend-label">AdamW Update (&Delta;&theta;)</span></span>
                                <span class="legend-hint">CLICK STAGE: INSPECT &middot; HOVER: TENSOR SHAPES</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="portal-content">
                    {portal_cards_html}
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    <script defer src="assets/portal.js"></script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / "index.html"
    out_file.write_text(index_html, encoding="utf-8")


def generate_assets():
    """Copy assets/style.css and assets/portal.js into the docs build."""
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    src_css = WORKSPACE_DIR / "assets" / "style.css"
    shutil.copyfile(src_css, assets_dir / "style.css")
    src_js = WORKSPACE_DIR / "assets" / "portal.js"
    if src_js.exists():
        shutil.copyfile(src_js, assets_dir / "portal.js")


def main():
    print("Building LLaMA-3-Lite HTML Documentation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_assets()

    for rel_path, category, display_title in DOC_FILES:
        print(f"Generating: {rel_path} -> docs_html/{rel_path.replace('.md', '.html')}")
        generate_html_page(rel_path, category, display_title)

    generate_index_portal()
    print("\nDocumentation build complete!")
    print(f"HTML Portal location: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
