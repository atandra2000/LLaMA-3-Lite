"""Build-output contract tests for the docs_html portal in LLaMA-3-Lite.

Runs the generator once (module scope) and asserts the premium-polish
wiring: portal.js asset, boot overlay, hero canvas instrument, mono-only fonts,
pass diagram, and interactive mechanism widgets. Markdown sources are never
modified by the build.
"""

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
