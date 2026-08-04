#!/usr/bin/env python3
"""Generate docs/CODE_MAP.md from every `<module>.py:<symbol>` citation in the docs.

The CI checker (tests/test_doc_refs.py) verifies the citations resolve; this
script aggregates them into the human-readable symbol ↔ doc map.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_GLOBS = ["docs/**/*.md", "*.md", "data/*.md"]
SKIP_FILES = {"docs/docs_expansion_plan.md", "docs/CODE_MAP.md", "docs/README.md"}

SYMBOL_RE = re.compile(
    r"`?([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]*)"
    r"|`?([A-Za-z_][A-Za-z0-9_./-]*\.py)::([A-Za-z_][A-Za-z0-9_.]*)"
)

OUT = """# CODE_MAP — symbol ↔ doc ↔ test map

> **Generated** by `scripts/generate_code_map.py` from the docs' own
> `<module>.py:<symbol>` citations. `tests/test_doc_refs.py` (CI) verifies
> every citation resolves; run this script after doc changes and commit the
> diff. Line-number anchors are banned.

{body}
"""


def _doc_files():
    found = []
    for pattern in DOC_GLOBS:
        for p in ROOT.glob(pattern):
            rel = str(p.relative_to(ROOT))
            if p.is_file() and p.suffix == ".md" and rel not in SKIP_FILES:
                found.append(p)
    return sorted(set(found))


def main() -> int:
    by_module: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in _doc_files():
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for m in SYMBOL_RE.finditer(text):
            mod, sym = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
            by_module[mod][sym].append(str(rel))

    rows = []
    for mod in sorted(by_module):
        rows.append(f"| `{mod}` | {', '.join(f'`{s}`' for s in sorted(by_module[mod]))} |")
    table = "\n".join(rows)
    body = f"## Modules and the symbols the docs cite\n\n| Module | Cited symbols |\n|--------|---------------|\n{table}\n"

    (ROOT / "docs" / "CODE_MAP.md").write_text(OUT.format(body=body), encoding="utf-8")
    print(f"wrote docs/CODE_MAP.md ({len(by_module)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
