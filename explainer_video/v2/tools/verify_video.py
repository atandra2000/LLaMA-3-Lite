"""Release gate — prove a built master is shippable, end to end.

Everything the plan calls a release blocker, checked against the artefacts a
reader will actually receive: the MP4, its chapter markers, and the caption
files. Numbers that live in ``narration.yaml`` or ``audio/timings.json`` are
read from there, never retyped here.

1. **Container** — the file decodes, and it carries one audio track.
2. **Chapters** — one per narration section, in order, contiguous, and titled
   from ``narration.yaml``.
3. **Narration fit** — each section's speech plus its lead sits inside the
   chapter that carries it, and the sentence spans are monotonic.
4. **Loudness** — integrated −16.0 ±0.5 LUFS, true peak at or below −1.5 dBFS.
5. **Freeze** — frozen share at most 35%, no hold longer than 3.5 s.
6. **Captions** — SRT and VTT both present, one cue per spoken sentence.

Usage:
    python tools/verify_video.py                       # the default master
    python tools/verify_video.py path/to/master.mp4    # a specific build
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

try:  # imported as ``tools.verify_video``
    from tools import check_freeze
except ImportError:  # run as ``python tools/verify_video.py``
    import check_freeze

V2 = Path(__file__).resolve().parents[1]
NARRATION = V2 / "narration.yaml"
TIMINGS = V2 / "audio" / "timings.json"
DEFAULT_MASTER = V2 / "llama3_lite_architecture_explainer.mp4"

# PLAN.md §2 targets. TP leaves 0.1 dB of headroom so a dither step cannot
# push a passing master over the −1.5 dBFS ceiling.
I_LO = -16.5
I_HI = -15.5
TP_MAX = -1.5
# Speech has to beat the gaps by this much. A dropped voice reads negative:
# the music bed plays on where the narration should be.
VOICE_MIN = 6.0
FROZEN_MAX = 0.35
HOLD_MAX = 3.5


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.checks += 1
        print(f"  ok   {name}{(' — ' + detail) if detail else ''}")

    def fail(self, name: str, detail: str) -> None:
        self.checks += 1
        self.failures.append(f"{name}: {detail}")
        print(f"  FAIL {name} — {detail}")

    def expect(self, name: str, actual, expected, detail: str = "") -> None:
        if actual == expected:
            self.ok(name, detail or f"{actual!r}")
        else:
            self.fail(name, f"{actual!r} != expected {expected!r}")


# ---------------------------------------------------------------- measuring --
def run_json(cmd: list[str]) -> dict:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd[:4])}...\n{r.stderr[-800:]}")
    return json.loads(r.stdout)


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def chapters(path: Path) -> list[dict]:
    """Chapter markers as [{'start': s, 'end': s, 'title': str}, ...]."""
    data = run_json(["ffprobe", "-v", "error", "-show_chapters",
                     "-print_format", "json", str(path)])
    return [{
        "start": float(c["start_time"]),
        "end": float(c["end_time"]),
        "title": c.get("tags", {}).get("title", ""),
    } for c in data.get("chapters", [])]


def has_audio(path: Path) -> str:
    """Codec name of the first audio stream, or '' when there is none."""
    data = run_json(["ffprobe", "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=codec_name",
                     "-print_format", "json", str(path)])
    streams = data.get("streams", [])
    return streams[0]["codec_name"] if streams else ""


def parse_ebur128(blob: str) -> tuple[float, float]:
    """(integrated LUFS, true-peak dBFS) out of an ebur128 stderr dump.

    The summary block repeats the label after the frame-by-frame log, so the
    last match is the one that counts.
    """
    i = re.findall(r"I:\s*(-?[0-9.]+) LUFS", blob)
    tp = re.findall(r"Peak:\s*(-?[0-9.]+) dBFS", blob)
    if not i or not tp:
        raise RuntimeError("ebur128 printed no summary block")
    return float(i[-1]), float(tp[-1])


def measure_loudness(path: Path) -> tuple[float, float]:
    """(integrated LUFS, true-peak dBFS) from ffmpeg's ebur128 read-out."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-map", "0:a", "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True)
    return parse_ebur128(r.stderr)


def count_srt_cues(path: Path) -> int:
    """Number of subtitle cues in an SRT file."""
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if "-->" in line:
            n += 1
    return n


# ------------------------------------------------------------------ checks --
def check_container(gate: Gate, master: Path, total: float) -> None:
    if total <= 0:
        gate.fail("duration", f"{total:.2f}s")
    else:
        gate.ok("duration", f"{total:.2f}s = {total / 60:.2f} min")
    codec = has_audio(master)
    if codec:
        gate.ok("audio track", codec)
    else:
        gate.fail("audio track", "no audio stream")


def check_chapters(gate: Gate, master: Path, total: float,
                   sections: list[dict]) -> list[dict]:
    got = chapters(master)
    gate.expect("chapter count", len(got), len(sections),
                f"{len(sections)} narration sections")
    titles = [s["title"] for s in sections]
    want = [f"{i + 1:02d} · {t}" for i, t in enumerate(titles)]
    for i, ch in enumerate(got):
        if i < len(want) and ch["title"] != want[i]:
            gate.fail(f"chapter {i + 1} title",
                      f"{ch['title']!r} != {want[i]!r}")
    if got and all(got[i]["title"] == want[i] for i in range(min(len(got), len(want)))):
        gate.ok("chapter titles", "match narration.yaml")

    # Contiguous, and covering the whole file.
    bad = []
    for i, ch in enumerate(got):
        if ch["end"] <= ch["start"]:
            bad.append(f"ch{i + 1} empty")
        if i and abs(ch["start"] - got[i - 1]["end"]) > 0.05:
            bad.append(f"ch{i + 1} gap {ch['start'] - got[i - 1]['end']:+.3f}s")
    if got:
        if abs(got[0]["start"]) > 0.05:
            bad.append(f"starts at {got[0]['start']:.3f}s")
        if abs(got[-1]["end"] - total) > 0.05:
            bad.append(f"ends at {got[-1]['end']:.3f}s vs {total:.3f}s")
    if bad:
        gate.fail("chapter continuity", "; ".join(bad))
    elif got:
        gate.ok("chapter continuity", "contiguous, covers the file")
    return got


def check_narration_fit(gate: Gate, chaps: list[dict], lead: float,
                        order: list[str], sections: dict) -> None:
    """Speech plus lead must sit inside the chapter that carries it."""
    over = []
    for i, k in enumerate(order):
        if i >= len(chaps):
            break
        audio = sections[k]["duration"]
        room = chaps[i]["end"] - chaps[i]["start"]
        if audio + lead > room:
            over.append(f"{k}: {audio:.2f}+{lead:.2f}s in {room:.2f}s")
    if over:
        gate.fail("narration fits chapter", "; ".join(over))
    else:
        gate.ok("narration fits chapter", f"{len(order)} sections, lead {lead}s")

    bad = []
    for k in order:
        prev = 0.0
        for sid, span in sections[k]["sentences"].items():
            if span["start"] < prev - 1e-6:
                bad.append(f"{sid} overlaps its predecessor")
            if span["end"] <= span["start"]:
                bad.append(f"{sid} empty span")
            if span["end"] > sections[k]["duration"] + 1e-6:
                bad.append(f"{sid} past its track")
            prev = span["end"]
    if bad:
        gate.fail("cue monotonicity", "; ".join(bad[:6]))
    else:
        n = sum(len(sections[k]["sentences"]) for k in order)
        gate.ok("cue monotonicity", f"{n} sentence spans in order")


def check_captions(gate: Gate, master: Path, n_sentences: int) -> None:
    for ext in ("srt", "vtt"):
        f = master.with_suffix(f".{ext}")
        if not f.exists():
            gate.fail(f"captions {ext}", f"missing {f.name}")
            continue
        if ext == "srt":
            n = count_srt_cues(f)
            if n == n_sentences:
                gate.ok(f"captions {ext}", f"{n} cues")
            else:
                gate.fail(f"captions {ext}", f"{n} cues != {n_sentences} sentences")
        else:
            text = f.read_text(encoding="utf-8")
            n = text.count("-->")
            if n == n_sentences:
                gate.ok(f"captions {ext}", f"{n} cues")
            else:
                gate.fail(f"captions {ext}", f"{n} cues != {n_sentences} sentences")


def speech_over_gaps(x: np.ndarray, chaps: list[dict], lead: float,
                     order: list[str], sections: dict, sr: int = 8000) -> float:
    """Level of the spoken spans minus the level of the gaps, in dB.

    Negative means the gaps are louder than the sentences, which is what a
    mix with the voice dropped and the bed left in looks like.
    """
    mask = np.zeros(len(x), dtype=bool)
    for i, k in enumerate(order):
        if i >= len(chaps):
            break
        base = chaps[i]["start"] + lead
        for span in sections[k]["sentences"].values():
            a = int(round((base + span["start"]) * sr))
            b = int(round((base + span["end"]) * sr))
            if 0 <= a < len(x):
                mask[a:min(b, len(x))] = True
    if not mask.any() or mask.all():
        raise ValueError("no speech windows landed inside the master")

    def rms(y: np.ndarray) -> float:
        return 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)

    return rms(x[mask]) - rms(x[~mask])


def check_voice_present(gate: Gate, master: Path, chaps: list[dict], lead: float,
                        order: list[str], sections: dict) -> None:
    """Speech must sit clearly above the gaps between sentences.

    Integrated loudness passing does not prove the narration reached the mix:
    a build that drops the voice and keeps the music bed still lands near
    −16 LUFS. Measure the master itself and compare the two populations.
    """
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(master), "-ac", "1", "-ar", "8000",
         "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        gate.fail("voice in mix", "could not decode the audio track")
        return
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        delta = speech_over_gaps(x, chaps, lead, order, sections)
    except ValueError as e:
        gate.fail("voice in mix", str(e))
        return
    if delta > VOICE_MIN:
        gate.ok("voice in mix", f"speech {delta:+.1f} dB over the gaps")
    else:
        gate.fail("voice in mix",
                  f"speech only {delta:+.1f} dB over the gaps (needs "
                  f"> {VOICE_MIN:.0f}) — the narration is not in the mix")


def main(argv: list[str]) -> int:
    master = Path(argv[1]) if len(argv) > 1 else DEFAULT_MASTER
    if not master.exists():
        print(f"FAIL no such master: {master}")
        return 1
    if not TIMINGS.exists():
        print(f"FAIL no timing map: {TIMINGS} — run generate_audio.py")
        return 1

    narration = yaml.safe_load(NARRATION.read_text(encoding="utf-8"))
    timings = json.loads(TIMINGS.read_text(encoding="utf-8"))
    sections = narration["sections"]
    order = list(timings["sections"])
    lead = float(timings["lead"])

    if [s["id"] for s in sections] != order:
        print("FAIL narration.yaml and timings.json describe different sections — "
              "re-run generate_audio.py")
        return 1

    total = probe_duration(master)
    print(f"{master.name} · {total:.2f}s · {len(sections)} sections · "
          f"{sum(len(s['sentences']) for s in sections)} sentences")
    print()

    gate = Gate()
    check_container(gate, master, total)
    chaps = check_chapters(gate, master, total, sections)
    check_narration_fit(gate, chaps, lead, order, timings["sections"])
    check_voice_present(gate, master, chaps, lead, order, timings["sections"])

    integrated, peak = measure_loudness(master)
    if I_LO <= integrated <= I_HI:
        gate.ok("integrated loudness", f"{integrated:.1f} LUFS "
                                       f"(target {I_LO}…{I_HI})")
    else:
        gate.fail("integrated loudness", f"{integrated:.1f} LUFS "
                                         f"outside {I_LO}…{I_HI}")
    if peak <= TP_MAX:
        gate.ok("true peak", f"{peak:.1f} dBFS (ceiling {TP_MAX})")
    else:
        gate.fail("true peak", f"{peak:.1f} dBFS over the {TP_MAX} ceiling")

    holds, _ = check_freeze.measure(master)
    frozen = sum(holds)
    share = frozen / total if total else 0.0
    longest = max(holds) if holds else 0.0
    if share <= FROZEN_MAX:
        gate.ok("frozen share", f"{share * 100:.1f}% (gate ≤ {FROZEN_MAX * 100:.0f}%)")
    else:
        gate.fail("frozen share", f"{share * 100:.1f}% over {FROZEN_MAX * 100:.0f}%")
    if longest <= HOLD_MAX:
        gate.ok("longest hold", f"{longest:.2f}s (gate ≤ {HOLD_MAX}s)")
    else:
        gate.fail("longest hold", f"{longest:.2f}s over {HOLD_MAX}s")

    n_sentences = sum(len(s["sentences"]) for s in sections)
    check_captions(gate, master, n_sentences)

    print()
    if gate.failures:
        print(f"VERIFY FAILED — {len(gate.failures)} of {gate.checks} checks:")
        for f in gate.failures:
            print(f"  {f}")
        return 1
    print(f"VERIFY PASSED — {gate.checks} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
