#!/bin/bash
# Phase 1b gate: 480p render of every scene with the safe-rect guard on.
cd "$(dirname "$0")/.." || exit 1
mkdir -p /tmp/p1b
for s in Section01 Section02 Section03 Section04 Section05 Section06 Section07 Section08 Section09; do
  echo "=== $s ==="
  .venv/bin/python render_scenes.py "$s" --check_layout -ql --format=mp4 > "/tmp/p1b/$s.log" 2>&1
  echo "$s exit=$?"
done
echo "=== ALL SCENES DONE ==="
