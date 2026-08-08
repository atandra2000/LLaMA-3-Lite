#!/usr/bin/env python3
"""
LLaMA-3-Lite Documentation Generator
Converts project markdown files into a responsive, beautifully-styled HTML documentation portal
with full LaTeX math (KaTeX) and syntax highlighting support.
Output directory: docs_html/ (ignored by git).
"""

import os
import re
import html
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


def slugify(text: str) -> str:
    """Generate clean HTML id for headings.

    Matches the anchor convention the docs were authored against (GitHub-style):
    lowercase, keep word chars + underscore + hyphen, drop everything else,
    and turn each space into a single hyphen (no run collapsing). An em-dash
    surrounded by spaces thus yields ``--`` (e.g. ``Data Flow — Training`` ->
    ``data-flow--training``), and code identifiers keep their underscores
    (``## train_step`` -> ``train_step``).
    """
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
    """Rewrite relative markdown links for the HTML build.

    - ``.md`` links (with optional ``#anchor``) -> ``.html`` twin.
    - non-``.md`` repo-relative links (``configs/…``, ``training/…``) -> GitHub
      blob URL (resolved against the source file's repo-relative dir), since the
      file isn't shipped inside ``docs_html/``.
    """
    repo_base = github_base_url()
    src_dir = Path(src_rel_path).parent

    def link_replacer(match):
        label = match.group(1)
        url = match.group(2)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return f"[{label}]({url})"
        path_part, _, anchor = url.partition("#")
        if path_part.endswith(".md"):
            target = path_part[:-3] + ".html"
            if anchor:
                target += "#" + anchor
            return f"[{label}]({target})"
        if repo_base and not path_part.startswith("/"):
            repo_rel = (src_dir / path_part).resolve().relative_to(WORKSPACE_DIR)
            return f"[{label}]({repo_base}/{repo_rel})"
        return f"[{label}]({url})"

    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)


def parse_markdown_to_html(md_text: str, src_rel_path: str) -> tuple[str, list[dict]]:
    """
    Statically convert markdown to rich HTML structure with full LaTeX & Math protection.
    Returns (html_content, toc_items).
    """
    md_text = fix_md_links(md_text, src_rel_path)
    
    # -------------------------------------------------------------
    # STEP 1: Protect Code Blocks & Inline Code
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # STEP 2: Protect LaTeX Math Blocks & Inline Math
    # -------------------------------------------------------------
    display_maths = []
    def store_display_math(m):
        inner = m.group(1).strip()
        # Escape minimal HTML entities (< and >) so browser doesn't interpret them as tags,
        # but keep all backslashes and LaTeX symbols untouched!
        safe_math = html.escape(inner, quote=False)
        display_maths.append(f'<div class="math-block">$$\n{safe_math}\n$$</div>')
        return f"\n\n___DISPLAYMATH_{len(display_maths)-1}___\n\n"
    
    # Match $$ ... $$ (display math)
    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', store_display_math, md_text)
    # Also match \[ ... \]
    md_text = re.sub(r'\\\[([\s\S]+?)\\\]', store_display_math, md_text)
    
    inline_maths = []
    def store_inline_math(m):
        inner = m.group(1).strip()
        safe_math = html.escape(inner, quote=False)
        inline_maths.append(f'<span class="math-inline">${safe_math}$</span>')
        return f"___INLINEMATH_{len(inline_maths)-1}___"
    
    # Match $ ... $ (inline math)
    md_text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', store_inline_math, md_text)
    # Also match \( ... \)
    md_text = re.sub(r'\\\(([\s\S]+?)\\\)', store_inline_math, md_text)

    # -------------------------------------------------------------
    # STEP 3: Parse Document Structure Line by Line
    # -------------------------------------------------------------
    toc = []
    lines = md_text.splitlines()
    html_lines = []
    
    in_table = False
    table_headers = []
    table_rows = []
    
    # Stack of open list contexts; each entry keeps its own <li> open until the
    # next item or a flush, so nested lists and wrapped items render correctly.
    list_stack = []  # [{'indent': int, 'tag': str, 'li_open': bool}]
    h1_seen = False  # the first H1 duplicates the page's doc-title; suppressed

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
        """Escape `<`, `>` and stray `&`, but leave valid HTML entities intact
        (``&nbsp;``, ``&rarr;``, ``&#8230;``, ...) so source entities survive."""
        text = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', text)
        return text.replace('<', '&lt;').replace('>', '&gt;')

    def render_inline_formatting(text: str) -> str:
        # Escape html chars for safety (except placeholders), preserving entities
        text = escape_preserving_entities(text)
        
        # Bold **text**
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        # Italic *text*
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        # Strikethrough ~~text~~
        text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)
        # Links [text](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" class="doc-link">\1</a>', text)

        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Placeholders for display math or code blocks on their own line
        if stripped.startswith("___DISPLAYMATH_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        if stripped.startswith("___CODEBLOCK_"):
            flush_table()
            flush_list()
            flush_blockquote()
            html_lines.append(stripped)
            i += 1
            continue

        # Empty line
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

        # Headings (# to ######)
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            flush_table()
            flush_list()
            flush_blockquote()
            level = len(heading_match.group(1))
            heading_text_raw = heading_match.group(2).strip()
            
            # Clean heading text for id/slug
            clean_title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', heading_text_raw)
            clean_title = re.sub(r'`([^`]+)`', r'\1', clean_title)
            clean_title = re.sub(r'___INLINECODE_\d+___', '', clean_title)
            clean_title = re.sub(r'___INLINEMATH_\d+___', '', clean_title)
            heading_id = slugify(clean_title)
            
            rendered_heading = render_inline_formatting(heading_text_raw)
            
            if level in (2, 3):
                toc.append({
                    'level': level,
                    'title': clean_title,
                    'id': heading_id
                })

            # The first H1 duplicates the page's doc-title; suppress the visible
            # heading but keep its id so deep links (e.g. #llama-3-lite) resolve.
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
            i += 2  # skip separator line
            
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells_raw = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table_rows.append([render_inline_formatting(c) for c in cells_raw])
                i += 1
            flush_table()
            continue

        # Lists (unordered - or *, ordered 1.) — nested by leading indentation
        ul_match = re.match(r'^[\*\-]\s+(.+)$', stripped)
        ol_match = re.match(r'^\d+\.\s+(.+)$', stripped)
        if ul_match or ol_match:
            flush_table()
            flush_blockquote()
            tag = 'ul' if ul_match else 'ol'
            item_text = (ul_match or ol_match).group(1).strip()
            indent = len(line) - len(line.lstrip(' '))

            # Close lists nested deeper than this item's indent
            while list_stack and indent < list_stack[-1]['indent']:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            # Same indent but a different list type → close the old list
            if list_stack and list_stack[-1]['indent'] == indent and list_stack[-1]['tag'] != tag:
                close_li()
                html_lines.append(f"</{list_stack[-1]['tag']}>")
                list_stack.pop()
            # Open a new list when none exists at this indent
            if not list_stack or list_stack[-1]['indent'] != indent:
                list_stack.append({'indent': indent, 'tag': tag, 'li_open': False})
                html_lines.append(f'<{tag} class="doc-list">')
            else:
                close_li()  # next item in the same list

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

        # Continuation of an open list item (indented prose that belongs to it)
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

    # -------------------------------------------------------------
    # STEP 4: Restore Protected Tokens
    # -------------------------------------------------------------
    # Restore inline math
    for idx, math_html in enumerate(inline_maths):
        full_html = full_html.replace(f"___INLINEMATH_{idx}___", math_html)

    # Restore inline code
    for idx, raw_code in enumerate(inline_codes):
        # Extract content between `...`
        code_content = raw_code[1:-1]
        code_html = f'<code class="inline-code">{html.escape(code_content)}</code>'
        full_html = full_html.replace(f"___INLINECODE_{idx}___", code_html)

    # Restore display math
    for idx, math_html in enumerate(display_maths):
        full_html = full_html.replace(f"___DISPLAYMATH_{idx}___", math_html)

    # Restore code blocks
    for idx, raw_block in enumerate(code_blocks):
        lines_b = raw_block.splitlines()
        first_line = lines_b[0].strip()
        code_lang = first_line.lstrip("```").strip().lower()
        code_content = "\n".join(lines_b[1:-1])
        escaped_content = html.escape(code_content)
        
        lang_attr = f' class="language-{code_lang}"' if code_lang else ''
        data_lang = code_lang if code_lang else 'code'
        
        block_html = (
            f'<div class="code-wrapper">'
            f'<div class="code-header">'
            f'<span class="code-lang">{data_lang}</span>'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button>'
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
    """Build the navigation sidebar HTML."""
    sidebar_sections = {
        "Core": [],
        "Concepts": [],
        "Guides": [],
        "References": []
    }
    
    for rel_path, category, display_title in DOC_FILES:
        target_html_rel = rel_path.replace(".md", ".html")
        href = rel_prefix + target_html_rel
        is_active = (rel_path == current_rel_path)
        active_cls = "active" if is_active else ""
        sidebar_sections[category].append(
            f'<li class="nav-item"><a href="{href}" class="nav-link {active_cls}">{display_title}</a></li>'
        )
        
    html_out = ['<div class="sidebar-search"><input type="text" id="navSearch" placeholder="Search docs..." onkeyup="filterNav()"></div>']
    
    for cat_name, items in sidebar_sections.items():
        if items:
            html_out.append(f'<div class="nav-group">')
            html_out.append(f'<div class="nav-group-title">{cat_name}</div>')
            html_out.append(f'<ul class="nav-list">{"".join(items)}</ul>')
            html_out.append(f'</div>')
            
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
    
    word_count = len(md_text.split())
    reading_time = max(1, round(word_count / 200))
    
    html_body, toc_items = parse_markdown_to_html(md_text, rel_path)
    
    rel_prefix = compute_rel_prefix(rel_path)
    sidebar_html = build_sidebar_html(rel_path, rel_prefix)
    toc_html = build_toc_html(toc_items)
    
    current_idx = next((i for i, df in enumerate(DOC_FILES) if df[0] == rel_path), 0)
    prev_doc = DOC_FILES[current_idx - 1] if current_idx > 0 else None
    next_doc = DOC_FILES[current_idx + 1] if current_idx < len(DOC_FILES) - 1 else None
    
    prev_html = ""
    if prev_doc:
        prev_href = rel_prefix + prev_doc[0].replace(".md", ".html")
        prev_html = f'<a href="{prev_href}" class="nav-card prev-card"><span class="card-label">← Previous</span><span class="card-title">{prev_doc[2]}</span></a>'
        
    next_html = ""
    if next_doc:
        next_href = rel_prefix + next_doc[0].replace(".md", ".html")
        next_html = f'<a href="{next_href}" class="nav-card next-card"><span class="card-label">Next →</span><span class="card-title">{next_doc[2]}</span></a>'
        
    page_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{display_title} | LLaMA-3-Lite Documentation</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" id="highlight-theme">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <!-- KaTeX for LaTeX Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="{rel_prefix}assets/style.css">
</head>
<body>
    <!-- Top Header -->
    <header class="site-header">
        <div class="header-left">
            <button class="mobile-toggle" onclick="toggleSidebar()" aria-label="Toggle Sidebar">☰</button>
            <a href="{rel_prefix}index.html" class="brand-logo">
                <span class="brand-name">LLaMA-3-Lite</span>
                <span class="brand-badge">Docs</span>
            </a>
        </div>
        <div class="header-right">
            <a href="{rel_prefix}index.html" class="header-link">Portal</a>
            <a href="{rel_prefix}README.html" class="header-link">README</a>
            <button class="theme-toggle" onclick="toggleTheme()" id="themeToggleBtn" aria-label="Toggle Theme">Dark</button>
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
                        <span class="meta-item">📁 {rel_path}</span>
                        <span class="meta-item">📝 {word_count:,} words</span>
                        <span class="meta-item">⏱️ ~{reading_time} min read</span>
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
    <script>
        // Copy Code Functionality
        function copyCode(btn) {{
            const wrapper = btn.closest('.code-wrapper');
            const code = wrapper.querySelector('code').innerText;
            navigator.clipboard.writeText(code).then(() => {{
                btn.innerText = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => {{
                    btn.innerText = 'Copy';
                    btn.classList.remove('copied');
                }}, 2000);
            }});
        }}

        // Theme Toggle
        function toggleTheme() {{
            const htmlEl = document.documentElement;
            const themeBtn = document.getElementById('themeToggleBtn');
            const hlTheme = document.getElementById('highlight-theme');
            
            if (htmlEl.getAttribute('data-theme') === 'dark') {{
                htmlEl.setAttribute('data-theme', 'light');
                themeBtn.innerText = 'Light';
                hlTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
                localStorage.setItem('theme', 'light');
            }} else {{
                htmlEl.setAttribute('data-theme', 'dark');
                themeBtn.innerText = 'Dark';
                hlTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css';
                localStorage.setItem('theme', 'dark');
            }}
        }}

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {{
            document.documentElement.setAttribute('data-theme', 'light');
            document.getElementById('themeToggleBtn').innerText = 'Light';
            document.getElementById('highlight-theme').href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
        }}

        function toggleSidebar() {{
            document.getElementById('sidebar').classList.toggle('open');
        }}

        function filterNav() {{
            const query = document.getElementById('navSearch').value.toLowerCase();
            const items = document.querySelectorAll('.nav-item');
            items.forEach(item => {{
                const text = item.innerText.toLowerCase();
                item.style.display = text.includes(query) ? 'block' : 'none';
            }});
        }}

        // Initialize Highlight.js & KaTeX
        document.addEventListener("DOMContentLoaded", function() {{
            if (window.hljs) {{
                hljs.highlightAll();
            }}
            if (window.renderMathInElement) {{
                renderMathInElement(document.body, {{
                    delimiters: [
                        {{left: '$$', right: '$$', display: true}},
                        {{left: '\\\\[', right: '\\\\]', display: true}},
                        {{left: '\\\\(', right: '\\\\)', display: false}},
                        {{left: '$', right: '$', display: false}}
                    ],
                    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
                    throwOnError: false
                }});
            }}
        }});
    </script>
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
        "Core Architecture": [
            ("README.html", "Project Overview", "515M-param LLaMA-3-class decoder in raw PyTorch with the 8-technique memory stack."),
            ("AGENTS.html", "AGENTS & System Architecture", "Codebase contracts, hard rules, GPU discipline, and file map."),
            ("SKILLS.html", "Skills Map", "Specialized agent workflows, scripts, and domain competencies."),
            ("docs/training.html", "Training Pipeline", "BF16 training loop, chunked cross-entropy, checkpointing & numerical stability."),
            ("docs/AUDIT.html", "Docs & Codebase Audit", "Findings table of docs↔code alignment and from-scratch codebase explanation.")
        ],
        "Architecture & Concepts": [
            ("docs/concepts/architecture-components.html", "Architecture Components", "RMSNorm + QK-norm, fused SwiGLU FFN, and loss design."),
            ("docs/concepts/attention-and-positional.html", "Attention & Positional", "GQA 8Q/4KV, causal mask, RoPE θ=500K, and the residual stream."),
            ("docs/concepts/data-and-kernels.html", "Data Pipeline & Triton", "Mixture, packing, dedup, disk-backed uint32 cache & Triton kernels."),
            ("docs/concepts/training-and-memory.html", "Training & Memory", "AdamW schedule, gradient checkpointing, BF16, chunked CE & z-loss.")
        ],
        "Guides & Playbooks": [
            ("docs/guides/quickstart.html", "Quickstart", "From zero to a running training loop — install, smoke test, full run."),
            ("docs/guides/learning-paths.html", "Learning Paths", "Beginner / intermediate / expert routes through the documentation."),
            ("docs/guides/troubleshooting.html", "Troubleshooting", "FAQ: CUDA OOM at batch 96, missing token cache, shared_data SystemExit."),
            ("docs/guides/glossary.html", "Glossary", "Notation, acronyms, config keys, and file layout.")
        ],
        "API References": [
            ("docs/references/model-reference.html", "Model, RoPE & Config", "Model classes, RoPE, config dataclass — shapes and wiring."),
            ("docs/references/data-reference.html", "Data & Kernels", "Data loader, tokenizer, and Triton kernel reference."),
            ("docs/references/training-reference.html", "Training & Tests", "Training loop, test strategy, fixtures, and CI."),
            ("docs/references/workspace-data.html", "Shared Data Pipeline", "The LLM/shared_data 8B-token pipeline — resolution, layout, use.")
        ]
    }
    
    portal_cards_html = ""
    for cat_title, items in categories.items():
        cards = ""
        for href, title, desc in items:
            cards += f"""
            <a href="{href}" class="portal-card">
                <h3 class="card-heading">{title}</h3>
                <p class="card-desc">{desc}</p>
            </a>
            """
        portal_cards_html += f"""
        <section class="portal-section">
            <h2 class="portal-category-title">{cat_title}</h2>
            <div class="portal-grid">{cards}</div>
        </section>
        """

    index_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaMA-3-Lite Documentation Portal</title>
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- CSS Stylesheet -->
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
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
            <a href="README.html" class="header-link">README</a>
            <button class="theme-toggle" onclick="toggleTheme()" id="themeToggleBtn" aria-label="Toggle Theme">Dark</button>
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
                    <div class="hero-accent-line" aria-hidden="true"></div>
                    <h1 class="hero-title">LLaMA-3-Lite</h1>
                    <p class="hero-subtitle">From-scratch PyTorch implementation of a LLaMA-3-class decoder — GQA, RoPE, fused SwiGLU, gradient checkpointing, chunked cross-entropy, disk-backed token cache — with a ~78% peak-memory cut.</p>
                    <div class="spec-row">
                        <div class="spec-datum"><span class="spec-num">515M</span><span class="spec-label">Parameters</span></div>
                        <div class="spec-datum"><span class="spec-num">16</span><span class="spec-label">Decoder blocks</span></div>
                        <div class="spec-datum"><span class="spec-num">8Q·4KV</span><span class="spec-label">GQA heads</span></div>
                        <div class="spec-datum"><span class="spec-num">~20 GB</span><span class="spec-label">Peak memory</span></div>
                    </div>
                </div>

                <div class="portal-content">
                    {portal_cards_html}
                </div>
            </div>
        </main>
    </div>

    <!-- Scripts -->
    <script>
        function toggleTheme() {{
            const htmlEl = document.documentElement;
            const themeBtn = document.getElementById('themeToggleBtn');
            if (htmlEl.getAttribute('data-theme') === 'dark') {{
                htmlEl.setAttribute('data-theme', 'light');
                themeBtn.innerText = 'Light';
                localStorage.setItem('theme', 'light');
            }} else {{
                htmlEl.setAttribute('data-theme', 'dark');
                themeBtn.innerText = 'Dark';
                localStorage.setItem('theme', 'dark');
            }}
        }}

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {{
            document.documentElement.setAttribute('data-theme', 'light');
            document.getElementById('themeToggleBtn').innerText = 'Light';
        }}

        function toggleSidebar() {{
            document.getElementById('sidebar').classList.toggle('open');
        }}

        function filterNav() {{
            const query = document.getElementById('navSearch').value.toLowerCase();
            const items = document.querySelectorAll('.nav-item');
            items.forEach(item => {{
                const text = item.innerText.toLowerCase();
                item.style.display = text.includes(query) ? 'block' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""

    out_file = OUTPUT_DIR / "index.html"
    out_file.write_text(index_html, encoding="utf-8")


def generate_css():
    """Create docs_html/assets/style.css with modern design system."""
    assets_dir = OUTPUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    css_content = """
/* LLaMA-3-Lite Documentation — Minimal Professional Design System
   Single accent, clean typography, generous whitespace.
   ──────────────────────────────────────────────────────────────────────── */
:root {
    --bg-main: #0c0e14;
    --bg-surface: #151820;
    --bg-surface-hover: #1c2030;
    --border-color: #252a38;
    --text-primary: #e2e8f0;
    --text-secondary: #8893a7;
    --text-muted: #555f73;
    --accent: #6b8aff;
    --accent-hover: #8aa4ff;
    --accent-alpha: rgba(107, 138, 255, 0.10);
    --accent-strong: rgba(107, 138, 255, 0.18);
    --code-bg: #12151d;
    --header-bg: rgba(12, 14, 20, 0.90);
    --callout-note-bg: rgba(107, 138, 255, 0.06);
    --callout-note-border: #6b8aff;
    --callout-tip-bg: rgba(107, 138, 255, 0.06);
    --callout-tip-border: #6b8aff;
    --callout-warn-bg: rgba(245, 180, 50, 0.06);
    --callout-warn-border: #f5b432;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'IBM Plex Mono', 'Fira Code', 'Consolas', monospace;
}

[data-theme="light"] {
    --bg-main: #f8f9fc;
    --bg-surface: #ffffff;
    --bg-surface-hover: #f1f3f8;
    --border-color: #dfe3ec;
    --text-primary: #1a1f2e;
    --text-secondary: #5c6478;
    --text-muted: #929bb0;
    --accent: #4a6cf7;
    --accent-hover: #3b5de6;
    --accent-alpha: rgba(74, 108, 247, 0.08);
    --accent-strong: rgba(74, 108, 247, 0.14);
    --code-bg: #f1f3f8;
    --header-bg: rgba(248, 249, 252, 0.92);
    --callout-note-bg: rgba(74, 108, 247, 0.05);
    --callout-note-border: #4a6cf7;
    --callout-tip-bg: rgba(74, 108, 247, 0.05);
    --callout-tip-border: #4a6cf7;
    --callout-warn-bg: rgba(180, 120, 10, 0.05);
    --callout-warn-border: #b4780a;
}

/* ── Reset ────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: var(--font-sans);
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.65;
    -webkit-font-smoothing: antialiased;
}

:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

/* ── Site Header ──────────────────────────────────────────────────────── */
.site-header {
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 1.5rem;
    height: 48px;
    background: var(--header-bg);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border-color);
}

.brand-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    text-decoration: none;
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: -0.01em;
}

.brand-badge {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    padding: 2px 7px;
    border-radius: 4px;
    background: var(--accent-alpha);
    color: var(--accent);
    font-weight: 500;
    letter-spacing: 0.02em;
}

.header-right { display: flex; align-items: center; gap: 0.85rem; }

.header-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.82rem;
    font-weight: 500;
    transition: color 0.15s;
}
.header-link:hover { color: var(--accent); }

.theme-toggle, .mobile-toggle {
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.78rem;
    font-family: var(--font-sans);
    font-weight: 500;
    transition: all 0.15s;
}
.theme-toggle:hover, .mobile-toggle:hover { background: var(--bg-surface-hover); color: var(--text-primary); }

.mobile-toggle { display: none; }

/* ── Layout Grid ──────────────────────────────────────────────────────── */
.app-layout {
    display: grid;
    grid-template-columns: 250px 1fr 210px;
    max-width: 1480px;
    margin: 0 auto;
    min-height: calc(100vh - 48px);
}

/* ── Sidebar Navigation ───────────────────────────────────────────────── */
.sidebar {
    border-right: 1px solid var(--border-color);
    background: var(--bg-main);
    position: sticky;
    top: 48px;
    height: calc(100vh - 48px);
    overflow-y: auto;
}

.sidebar-inner { padding: 1.15rem 0.85rem; }

.sidebar-search input {
    width: 100%;
    padding: 6px 10px;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 0.78rem;
    margin-bottom: 1.15rem;
    outline: none;
}
.sidebar-search input:focus { border-color: var(--accent); }

.nav-group { margin-bottom: 1.2rem; }

.nav-group-title {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--text-muted);
    margin-bottom: 0.45rem;
    padding-left: 8px;
}

.nav-list { list-style: none; }
.nav-item { margin-bottom: 1px; }

.nav-link {
    display: block;
    padding: 5px 10px;
    border-radius: 4px;
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 400;
    transition: all 0.12s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.nav-link:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
}
.nav-link.active {
    background: var(--accent-alpha);
    color: var(--accent);
    font-weight: 500;
}

/* ── Main Content Area ────────────────────────────────────────────────── */
.main-content {
    padding: 2.5rem 3rem;
    overflow-x: hidden;
}

.content-container { max-width: 780px; margin: 0 auto; }

.breadcrumb {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 0.85rem;
}
.breadcrumb a { color: var(--text-secondary); text-decoration: none; }
.breadcrumb a:hover { color: var(--accent); }
.breadcrumb .current { color: var(--text-primary); font-weight: 500; }

.doc-header {
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 1.15rem;
    margin-bottom: 1.75rem;
}

.doc-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.25;
    margin-bottom: 0.55rem;
}

.doc-meta {
    display: flex;
    gap: 1rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
}

/* ── Typography ───────────────────────────────────────────────────────── */
.markdown-body p {
    margin-bottom: 1.15rem;
    font-size: 0.95rem;
    color: var(--text-primary);
}

.markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
    color: var(--text-primary);
    font-weight: 600;
    line-height: 1.3;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    scroll-margin-top: 68px;
    position: relative;
}

.heading-anchor .anchor-link {
    opacity: 0;
    margin-left: 0.4rem;
    color: var(--text-muted);
    text-decoration: none;
    font-weight: 400;
    transition: opacity 0.15s;
}
.heading-anchor:hover .anchor-link { opacity: 1; }

.doc-anchor { display: block; height: 0; }

.markdown-body h2 {
    font-size: 1.3rem;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.3rem;
}
.markdown-body h3 { font-size: 1.1rem; font-weight: 600; }
.markdown-body h4 { font-size: 1rem; }

.doc-link { color: var(--accent); text-decoration: none; font-weight: 500; }
.doc-link:hover { text-decoration: underline; }

.inline-code {
    background: var(--code-bg);
    border: 1px solid var(--border-color);
    padding: 1px 5px;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 0.84em;
    color: var(--text-primary);
}

/* ── Code Blocks ──────────────────────────────────────────────────────── */
.code-wrapper {
    background: var(--code-bg);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    margin: 1.25rem 0;
    overflow: hidden;
}

.code-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 12px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-color);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 0.02em;
}

.copy-btn {
    background: transparent;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    padding: 2px 7px;
    border-radius: 3px;
    cursor: pointer;
    font-family: var(--font-sans);
    font-size: 0.68rem;
    transition: all 0.15s;
}
.copy-btn:hover { background: var(--bg-surface-hover); color: var(--text-primary); }
.copy-btn.copied { border-color: var(--accent); color: var(--accent); }

.code-wrapper pre {
    margin: 0;
    padding: 0.85rem 1rem;
    overflow-x: auto;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    line-height: 1.6;
}

/* ── Math ─────────────────────────────────────────────────────────────── */
.math-block {
    overflow-x: auto;
    margin: 1.25rem 0;
    padding: 0.8rem 1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    text-align: center;
}

.math-inline { font-size: 1.02em; padding: 0 2px; }

/* ── Tables ───────────────────────────────────────────────────────────── */
.table-container { overflow-x: auto; margin: 1.25rem 0; }

.doc-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    text-align: left;
}
.doc-table th {
    font-family: var(--font-mono);
    background: var(--bg-surface);
    padding: 7px 12px;
    font-weight: 500;
    font-size: 0.75rem;
    letter-spacing: 0.01em;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-secondary);
}
.doc-table td {
    padding: 7px 12px;
    border-bottom: 1px solid var(--border-color);
}
.doc-table tr:hover { background: var(--bg-surface-hover); }

/* ── Callouts / Blockquotes ───────────────────────────────────────────── */
.callout {
    padding: 0.8rem 1.1rem;
    border-left: 3px solid;
    border-radius: 0 6px 6px 0;
    margin: 1.25rem 0;
}
.callout-note { background: var(--callout-note-bg); border-color: var(--callout-note-border); }
.callout-tip  { background: var(--callout-tip-bg);  border-color: var(--callout-tip-border); }
.callout-warn, .callout-warning { background: var(--callout-warn-bg); border-color: var(--callout-warn-border); }
.callout-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 0.82rem;
    margin-bottom: 0.3rem;
}

blockquote {
    border-left: 3px solid var(--border-color);
    padding: 0.35rem 0.9rem;
    color: var(--text-secondary);
    margin: 1rem 0;
    font-style: italic;
}

/* ── Lists ────────────────────────────────────────────────────────────── */
.doc-list { padding-left: 1.4rem; margin-bottom: 1rem; }
.doc-list li { margin-bottom: 0.3rem; }

/* ── Footer Page Nav ──────────────────────────────────────────────────── */
.doc-footer-nav {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-color);
}

.nav-card {
    display: flex;
    flex-direction: column;
    padding: 0.8rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    text-decoration: none;
    transition: all 0.15s;
}
.nav-card:hover {
    border-color: var(--accent);
    background: var(--accent-alpha);
}
.next-card { text-align: right; }
.card-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.card-title {
    font-size: 0.85rem;
    color: var(--text-primary);
    font-weight: 500;
    margin-top: 3px;
}

/* ── TOC Sidebar ──────────────────────────────────────────────────────── */
.toc-sidebar {
    border-left: 1px solid var(--border-color);
    padding: 1.25rem 0.85rem;
    position: sticky;
    top: 48px;
    height: calc(100vh - 48px);
    overflow-y: auto;
}

.toc-title {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--text-muted);
    margin-bottom: 0.65rem;
}

.toc-list { list-style: none; }
.toc-list li { margin-bottom: 4px; }
.toc-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.75rem;
    transition: color 0.15s;
}
.toc-link:hover { color: var(--accent); }
.toc-h3 { padding-left: 10px; }

/* ── Hero / Index Portal ──────────────────────────────────────────────── */
.hero-banner {
    padding: 2rem 0 1.75rem;
    margin-bottom: 2.25rem;
    border-bottom: 1px solid var(--border-color);
    position: relative;
}
.hero-accent-line {
    position: absolute;
    top: 0; left: 0;
    width: 48px; height: 2px;
    background: var(--accent);
    border-radius: 1px;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin-bottom: 0.35rem;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    max-width: 640px;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* Spec readout row */
.spec-row {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}
.spec-datum { text-align: left; }
.spec-num {
    display: block;
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.01em;
}
.spec-label {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}

/* ── Portal Cards ─────────────────────────────────────────────────────── */
.portal-section { margin-bottom: 2.25rem; }

.portal-category-title {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--border-color);
    letter-spacing: -0.01em;
}

.portal-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 0.75rem;
}

.portal-card {
    display: block;
    padding: 1rem 1.1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    text-decoration: none;
    transition: all 0.15s;
    border-left: 2px solid transparent;
}
.portal-card:hover {
    border-color: var(--border-color);
    border-left-color: var(--accent);
    background: var(--bg-surface-hover);
}

.card-heading {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
    letter-spacing: -0.005em;
}
.card-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.45;
}

/* ── Responsive ───────────────────────────────────────────────────────── */
@media (max-width: 1100px) {
    .app-layout { grid-template-columns: 240px 1fr; }
    .toc-sidebar { display: none; }
}

@media (max-width: 768px) {
    .app-layout { grid-template-columns: 1fr; }
    .sidebar {
        display: none;
        position: fixed;
        left: 0;
        top: 48px;
        width: 260px;
        z-index: 99;
        background: var(--bg-main);
    }
    .sidebar.open { display: block; }
    .mobile-toggle { display: block; }
    .main-content { padding: 1.5rem 1.25rem; }
    .spec-row { gap: 1.25rem; }
    .spec-num { font-size: 0.95rem; }
    .hero-title { font-size: 1.5rem; }
}

@media print {
    .site-header, .sidebar, .toc-sidebar, .doc-footer-nav { display: none; }
    .app-layout { grid-template-columns: 1fr; }
    body { background: #fff; color: #000; }
}

"""
    (assets_dir / "style.css").write_text(css_content, encoding="utf-8")


def main():
    print("Building LLaMA-3-Lite HTML Documentation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generate_css()
    
    for rel_path, category, display_title in DOC_FILES:
        print(f"Generating: {rel_path} -> docs_html/{rel_path.replace('.md', '.html')}")
        generate_html_page(rel_path, category, display_title)
        
    generate_index_portal()
    print("\nDocumentation build complete!")
    print(f"HTML Portal location: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
