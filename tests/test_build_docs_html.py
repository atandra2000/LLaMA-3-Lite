"""Build-output contract tests for the docs_html portal in LLaMA-3-Lite.

Runs the generator once (module scope) and asserts the premium-polish
wiring: portal.js asset, boot overlay, hero canvas instrument, mono-only fonts,
pass diagram, and interactive mechanism widgets. Markdown sources are never
modified by the build.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "scripts" / "build_docs_html.py"
OUT = ROOT / "docs_html"


@pytest.fixture(scope="module")
def built():
    subprocess.run([sys.executable, str(BUILD)], check=True, cwd=ROOT)
    return OUT


def read(rel: str) -> str:
    return (OUT / rel).read_text(encoding="utf-8")


def test_portal_js_and_css_assets_copied(built):
    assert (OUT / "assets" / "portal.js").is_file()
    assert (OUT / "assets" / "style.css").is_file()


def test_doc_page_boot_wiring(built):
    html = read("README.html")
    assert 'id="boot-overlay"' in html
    assert "booting" in html
    assert '<script defer src="./assets/portal.js"></script>' in html


def test_nested_page_rel_prefix(built):
    html = read("docs/concepts/architecture-components.html")
    assert 'src="../../assets/portal.js"' in html
    assert 'href="../../assets/style.css"' in html


def test_font_link_mono_only(built):
    html = read("README.html")
    assert "IBM+Plex+Mono" in html
    assert "JetBrains+Mono" in html
    assert "Serif" not in html


def test_index_hero_figure_stage(built):
    html = read("index.html")
    assert 'id="hero-decode"' in html
    assert 'id="heroStateCanvas"' in html
    assert 'data-title="LLAMA-3-LITE"' in html
    assert "hero-title sr-only" in html
    assert "llama-telemetry-ribbon" in html


def test_index_pass_widget_container(built):
    html = read("index.html")
    assert 'id="passWidget"' in html
    assert 'id="passDiagramCanvas"' in html


def test_index_mechanism_cards(built):
    html = read("index.html")
    assert 'id="mchCardGqa"' in html
    assert 'id="mchCardChunkCe"' in html
    assert 'id="mchCardRope"' in html
    assert 'id="gqaRouterCanvas"' in html
    assert 'id="ropeExtrapCanvas"' in html


def test_dark_theme_only(built):
    html = read("README.html")
    assert "toggleTheme" not in html
    assert "theme-toggle" not in html
    css = (OUT / "assets" / "style.css").read_text(encoding="utf-8")
    assert '[data-theme="light"]' not in css


def test_readme_badges_render_as_images(built):
    """README badge markdown became <img>, not literal `![alt](url)` text."""
    html = read("README.html")
    assert html.count('class="md-img"') >= 4
    assert "![Python 3.10+](https://" not in html


def test_readme_raw_html_not_escaped(built):
    """`<div align="center">` etc. render as markup, not printed source."""
    html = read("README.html")
    assert "&lt;div align=&quot;center&quot;&gt;" not in html
    assert "&lt;br&gt;" not in html
    assert "<div align=\"center\">" in html


def test_readme_figure_image_resolved(built):
    """The atlas preview resolves to a renderable raw URL, not a repo path.

    The wrapping anchor may legitimately point at a GitHub blob page; the
    invariant is on the <img src>, which must be a raw-content URL.
    """
    html = read("README.html")
    assert 'src="docs/diagrams/atlas/' not in html
    srcs = re.findall(r'<img[^>]+src="([^"]*visual-check[^"]*)"', html)
    assert srcs, "atlas preview image missing"
    for src in srcs:
        assert src.startswith("https://raw.githubusercontent.com/")
        assert "/blob/" not in src


def test_readme_details_disclosure(built):
    html = read("README.html")
    assert "<details>" in html
    assert "<summary>" in html
    assert "&lt;details&gt;" not in html


def test_raw_html_sanitizer_drops_script_and_style():
    """Raw markdown HTML is filtered: no script, no inline style, no handlers."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_docs_html as b

    out = b.sanitize_raw_html(
        '<div align="center"><script>alert(1)</script>'
        '<img src="x.png" onerror="alert(2)" style="position:fixed">'
        '<a href="p.html" onclick="alert(3)">x</a></div>',
        "README.md",
    )
    assert "script" not in out
    assert "onerror" not in out
    assert "onclick" not in out
    assert "style=" not in out
    assert 'align="center"' in out
    assert "<img" in out


def test_prose_with_angle_brackets_stays_escaped(built):
    """`a < b and c > d` is prose, not an HTML island."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_docs_html as b

    assert not b.looks_like_raw_html_block("if x < 3 and y > 2:")
    assert not b.looks_like_raw_html_block("use <T> as a type parameter")
    assert b.looks_like_raw_html_block('<div align="center">')
    assert b.looks_like_raw_html_block("<details>")
    assert b.looks_like_raw_html_block("<br>")


def test_no_escaped_html_leaks_any_page(built):
    """No generated page prints raw HTML source at the reader."""
    import re

    leak = re.compile(r"&lt;/?(?:div|details|summary|img|br)\b", re.IGNORECASE)
    offenders = [p.relative_to(OUT).as_posix() for p in OUT.rglob("*.html")
                 if leak.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"escaped HTML leaked into: {offenders}"


def test_markdown_image_inside_link(built):
    """`[![alt](src)](href)` keeps the link wrapper around the image."""
    html = read("README.html")
    assert re.search(r'<a href="https://www\.python\.org[^"]*"[^>]*>\s*<img class="md-img"',
                     html)


def test_css_styles_passthrough_content(built):
    css = (OUT / "assets" / "style.css").read_text(encoding="utf-8")
    assert ".markdown-body img" in css
    assert ".markdown-body details" in css
    assert ".markdown-body summary" in css


def test_publish_branch_is_not_the_checked_out_branch(built):
    """Asset/blob links target the published branch, not a local worktree branch.

    Built from a feature branch or worktree, `git branch --show-current` names a
    ref that does not exist on GitHub, so every generated image and source link
    404s in the deployed site.
    """
    import build_docs_html as b

    base = b.github_base_url()
    if not base:
        pytest.skip("no GitHub remote configured")
    assert f"/blob/{b.DEFAULT_PUBLISH_BRANCH}" in base, base

    html = read("README.html")
    for src in re.findall(r'<img[^>]+src="(https://raw\.githubusercontent\.com/[^"]+)"', html):
        assert f"/{b.DEFAULT_PUBLISH_BRANCH}/" in src, src
        assert "/blob/" not in src


def test_publish_branch_env_override(monkeypatch):
    import build_docs_html as b

    monkeypatch.setenv("DOCS_PUBLISH_BRANCH", "release")
    base = b.github_base_url.__wrapped__()
    assert base.endswith("/blob/release"), base
