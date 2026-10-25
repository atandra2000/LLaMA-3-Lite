"""Lint ``narration.yaml`` — sentence length, cue ids, and shape.

The rule the plan enforces is **max 22 words per sentence** (section 5, the
attention walk-through, was rewritten to 12–18 and is capped at 18). Sentence
length is what lets Phase 2 synthesise one clip per cue and cut captions at
≤ 17 cps, so it is checked here rather than eyeballed later.

Usage:
    python tools/lint_script.py            # hard failures only
    python tools/lint_script.py --strict   # warnings fail too
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
NARRATION = V2 / "narration.yaml"

MAX_WORDS = 22
STRICT_SECTION = 5
STRICT_MAX_WORDS = 18
SOFT_MAX_WORDS = 18      # outside section 5: a rhythm hint, not a failure
SOFT_MIN_WORDS = 5       # very short clips fragment the voice
ID_RE = re.compile(r"^s(\d{2})\.(\d+)$")
TERMINAL = ".!?"
REQUIRED_SECTION_KEYS = ("id", "number", "chapter", "title", "screen", "sentences")
REQUIRED_SENTENCE_KEYS = ("id", "text", "visual")


def words(text: str) -> list[str]:
    """Whitespace tokens that carry a spoken word (standalone dashes do not count)."""
    return [w for w in text.split() if any(c.isalnum() for c in w)]


def word_count(text: str) -> int:
    return len(words(text))


def load(path: Path = NARRATION) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def lint(doc: dict) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []

    empty = {"sections": 0, "sentences": 0, "mean": 0.0, "max": 0}
    if not isinstance(doc, dict) or "sections" not in doc:
        return ["narration.yaml: missing 'sections'"], warnings, empty
    for key in ("version", "title", "voice", "rate", "timing"):
        if key not in doc:
            errors.append(f"top level: missing '{key}'")

    sections = doc["sections"]
    seen_ids: set[str] = set()
    seen_text: Counter[str] = Counter()
    sentence_total = 0
    lengths: list[int] = []

    for lineno, section in enumerate(sections, start=1):
        where = f"section {lineno} ({section.get('id', '?')})"
        for key in REQUIRED_SECTION_KEYS:
            if key not in section:
                errors.append(f"{where}: missing '{key}'")
        if section.get("number") != lineno:
            errors.append(f"{where}: number {section.get('number')!r} != position {lineno}")
        if section["id"] in seen_ids:
            errors.append(f"{where}: duplicate section id")
        seen_ids.add(section["id"])

        expected_prefix = f"s{lineno:02d}"
        for n, sentence in enumerate(section.get("sentences", []), start=1):
            sid = sentence.get("id", "?")
            tag = f"{where} · {sid}"
            for key in REQUIRED_SENTENCE_KEYS:
                if not sentence.get(key):
                    errors.append(f"{tag}: missing/empty '{key}'")
            match = ID_RE.match(str(sid))
            if not match:
                errors.append(f"{tag}: id must look like s{lineno:02d}.{n}")
            else:
                if int(match.group(1)) != lineno:
                    errors.append(f"{tag}: id prefix does not match section {lineno}")
                if int(match.group(2)) != n:
                    errors.append(f"{tag}: sentence number should be {n}")

            text = str(sentence.get("text", "")).strip()
            count = word_count(text)
            sentence_total += 1
            lengths.append(count)
            if count > MAX_WORDS:
                errors.append(f"{tag}: {count} words (max {MAX_WORDS})")
            if lineno == STRICT_SECTION and count > STRICT_MAX_WORDS:
                errors.append(f"{tag}: {count} words — section {STRICT_SECTION} "
                              f"is capped at {STRICT_MAX_WORDS}")
            if count > SOFT_MAX_WORDS and lineno != STRICT_SECTION:
                warnings.append(f"{tag}: {count} words — above the {SOFT_MAX_WORDS}-word "
                                f"rhythm target")
            if count < SOFT_MIN_WORDS:
                warnings.append(f"{tag}: only {count} words — may sound clipped")
            if text != str(sentence.get("text", "")).strip() or "  " in text:
                errors.append(f"{tag}: stray whitespace")
            if text and text[-1] not in TERMINAL:
                errors.append(f"{tag}: does not end with {TERMINAL!r}")
            seen_text[text.lower()] += 1

    for text, count in seen_text.items():
        if count > 1:
            errors.append(f"repeated sentence ×{count}: {text[:60]!r}")

    if doc.get("timing"):
        for key in ("lead", "tail", "sentence_gap"):
            if key not in doc["timing"]:
                errors.append(f"timing: missing '{key}'")

    stats = {"sections": len(sections), "sentences": sentence_total,
             "mean": sum(lengths) / len(lengths) if lengths else 0.0,
             "max": max(lengths) if lengths else 0}
    return errors, warnings, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=NARRATION)
    parser.add_argument("--strict", action="store_true",
                        help="treat rhythm warnings as failures")
    args = parser.parse_args(argv)

    doc = load(args.file)
    errors, warnings, stats = lint(doc)
    try:
        shown = args.file.relative_to(V2)
    except ValueError:
        shown = args.file

    print(f"{shown} · {stats['sections']} sections · {stats['sentences']} sentences · "
          f"mean {stats['mean']:.1f} words · max {stats['max']}")
    for warning in warnings:
        print(f"  warn {warning}")
    for error in errors:
        print(f"  FAIL {error}")

    if errors or (args.strict and warnings):
        print(f"\nSCRIPT LINT FAILED — {len(errors)} errors, {len(warnings)} warnings")
        return 1
    print(f"\nSCRIPT LINT PASSED — {stats['sentences']} sentences, "
          f"{len(warnings)} warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
