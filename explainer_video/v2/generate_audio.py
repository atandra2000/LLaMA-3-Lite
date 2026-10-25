"""Generate neural narration for the LLaMA-3-Lite architecture explainer.

Text comes from ``narration.yaml`` — the single source — never from this file.
Each sentence is synthesised on its own so the cue map is exact: clips are
silence-trimmed, joined with 200 ms gaps into one track per section, and the
resulting sentence offsets land in ``audio/timings.json``. That file is the
only timing source ``render_scenes.py`` reads.
"""
import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts
import yaml

HERE = Path(__file__).resolve().parent
NARRATION = HERE / "narration.yaml"
OUT_DIR = HERE / "audio"
SENT_DIR = OUT_DIR / "sentences"
SAMPLE_RATE = 24000          # edge-tts native; keep the voice at its own rate
TRIM_THRESHOLD = "-45dB"
TRIM_KEEP = 0.05             # silence kept at each edge so words breathe


def load_narration():
    """(voice, rate, timing, sections, sentences) from the single source.

    ``sections`` is ``[(section_id, whole-section text)]`` — the shape tools and
    tests pin. ``sentences`` is ``[(section_id, sentence_id, text)]`` and drives
    the per-sentence synthesis.
    """
    doc = yaml.safe_load(NARRATION.read_text(encoding="utf-8"))
    sections = [(s["id"], " ".join(x["text"].strip() for x in s["sentences"]))
                for s in doc["sections"]]
    sentences = [(s["id"], x["id"], x["text"].strip())
                 for s in doc["sections"] for x in s["sentences"]]
    return doc["voice"], doc["rate"], doc["timing"], sections, sentences


VOICE, RATE, TIMING, SECTIONS, SENTENCES = load_narration()


async def synth(sentence_id: str, text: str) -> Path:
    """One edge-tts clip per sentence, so a line can be re-recorded alone."""
    SENT_DIR.mkdir(parents=True, exist_ok=True)
    out = SENT_DIR / f"{sentence_id}.mp3"
    tts = edge_tts.Communicate(text, VOICE, rate=RATE)
    await tts.save(str(out))
    return out


def to_pcm(mp3: Path) -> bytes:
    """Silence-trimmed mono 16-bit PCM at SAMPLE_RATE, lead and tail alike."""
    trim = (f"silenceremove=start_periods=1:start_threshold={TRIM_THRESHOLD}"
            f":start_silence={TRIM_KEEP},areverse,"
            f"silenceremove=start_periods=1:start_threshold={TRIM_THRESHOLD}"
            f":start_silence={TRIM_KEEP},areverse")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp3), "-af", trim,
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True)
    return r.stdout


def assemble(pcms: list[bytes], gap_s: float) -> tuple[bytes, list[tuple[float, float]]]:
    """Join clips with ``gap_s`` of silence; return the PCM and (start, end) pairs."""
    gap = b"\x00\x00" * round(gap_s * SAMPLE_RATE)
    buf = bytearray()
    spans: list[tuple[float, float]] = []
    t = 0.0
    for i, pcm in enumerate(pcms):
        if i:
            buf += gap
            t += gap_s
        dur = len(pcm) / 2 / SAMPLE_RATE
        spans.append((round(t, 4), round(t + dur, 4)))
        buf += pcm
        t += dur
    return bytes(buf), spans


def write_wav(pcm: bytes, path: Path) -> None:
    """PCM → WAV in one ffmpeg pass (no encoder delay to skew the cue map)."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE),
         "-ac", "1", "-i", "-", str(path)],
        input=pcm, check=True)


def duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


async def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    gap_s = float(TIMING["sentence_gap"])
    timings = {
        "version": 1,
        "lead": float(TIMING["lead"]),
        "tail": float(TIMING["tail"]),
        "sentence_gap": gap_s,
        "sample_rate": SAMPLE_RATE,
        "sections": {},
    }
    by_section: dict[str, list[tuple[str, str]]] = {}
    for sec_id, sent_id, text in SENTENCES:
        by_section.setdefault(sec_id, []).append((sent_id, text))

    total = 0.0
    for sec_id, _text in SECTIONS:
        pcms = []
        for sent_id, text in by_section[sec_id]:
            pcm = to_pcm(await synth(sent_id, text))
            pcms.append(pcm)
            print(f"  {sent_id}: {len(pcm) / 2 / SAMPLE_RATE:.2f}s  "
                  f"({len(text.split())} words)")
        pcm_all, spans = assemble(pcms, gap_s)
        wav = OUT_DIR / f"{sec_id}.wav"
        write_wav(pcm_all, wav)
        dur = duration(wav)
        total += dur
        timings["sections"][sec_id] = {
            "file": wav.name,
            "duration": round(dur, 4),
            "sentences": {
                sent_id: {"start": spans[i][0], "end": spans[i][1], "text": text}
                for i, (sent_id, text) in enumerate(by_section[sec_id])
            },
        }
        print(f"{sec_id}: {dur:.2f}s  ({len(by_section[sec_id])} sentences)")

    timings["total_narration"] = round(total, 4)
    (OUT_DIR / "timings.json").write_text(json.dumps(timings, indent=2))
    print(f"TOTAL narration: {total:.1f}s = {total / 60:.1f} min")


if __name__ == "__main__":
    asyncio.run(main())
