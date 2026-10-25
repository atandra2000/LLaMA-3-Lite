"""Measure the freeze gate on a rendered video.

R1 said 69.9% of the v2 master was a still frame. The Phase 2 gate is
frozen-share ≤ 35% and no hold longer than 3.5 s, both read from ffmpeg's
freezedetect. Prints the numbers and exits non-zero when the gate fails.

    .venv/bin/python tools/check_freeze.py path/to/video.mp4
"""
import re
import subprocess
import sys
from pathlib import Path

MIN_DUR = 1.0            # a "hold" is a freeze of at least this long
MAX_HOLD = 3.5           # gate: no single hold may exceed this
MAX_FROZEN_SHARE = 0.35  # gate: frozen time / total time


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def measure(path: Path, noise: str = "0.001"):
    """(holds, total_seconds) — holds are the freeze spans freezedetect reports."""
    total = probe_duration(path)
    r = subprocess.run(
        ["ffmpeg", "-i", str(path), "-vf", f"freezedetect=n={noise}:d={MIN_DUR}",
         "-f", "null", "-"],
        capture_output=True, text=True)
    blob = r.stderr
    starts = [float(m) for m in re.findall(r"freeze_start: ([0-9.]+)", blob)]
    ends = [float(m) for m in re.findall(r"freeze_end: ([0-9.]+)", blob)]
    # A freeze still open at end-of-stream has no freeze_end.
    while len(ends) < len(starts):
        ends.append(total)
    holds = [e - s for s, e in zip(starts, ends) if e > s]
    return holds, total


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    holds, total = measure(path)
    frozen = sum(holds)
    share = frozen / total if total else 0.0
    longest = max(holds) if holds else 0.0
    print(f"file            {path.name}")
    print(f"duration        {total:.2f}s")
    print(f"holds ≥{MIN_DUR}s     {len(holds)}")
    print(f"frozen share    {share * 100:.1f}%  (gate ≤ {MAX_FROZEN_SHARE * 100:.0f}%)")
    print(f"longest hold    {longest:.2f}s  (gate ≤ {MAX_HOLD}s)")
    bad = []
    if share > MAX_FROZEN_SHARE:
        bad.append(f"frozen share {share * 100:.1f}%")
    if longest > MAX_HOLD:
        bad.append(f"longest hold {longest:.2f}s")
    if bad:
        print("FAIL: " + ", ".join(bad))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
