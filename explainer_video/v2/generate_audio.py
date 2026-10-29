"""Generate neural narration for the LLaMA-3-Lite architecture explainer.

Text comes from ``narration.yaml`` — the single source — never from this file.
Each sentence is synthesised on its own so the cue map is exact: clips are
silence-trimmed, joined with 200 ms gaps into one track per section, and the
resulting sentence offsets land in ``audio/timings.json``. That file is the
only timing source ``render_scenes.py`` reads.

The voice is a clone of ``audio/Sample_voice_2.m4a``, made locally by
Chatterbox on Apple Metal. It runs in its own env because torch and the CMU
port of the model want their own pins:
``~/.local/share/chatterbox-venv/bin/python``. ``tools/chatterbox_worker.py``
generates every line in one process, since a model load costs about a minute
and a line costs about thirty seconds.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NARRATION = HERE / "narration.yaml"
OUT_DIR = HERE / "audio"
SENT_DIR = OUT_DIR / "sentences"
VOICE_SAMPLE = OUT_DIR / "Sample_voice_2.m4a"
WORKER = HERE / "tools" / "chatterbox_worker.py"
CLONE_PY = Path(os.environ.get(
    "CHATTERBOX_PY", Path.home() / ".local/share/chatterbox-venv/bin/python"))
EXAGGERATION = float(os.environ.get("CHATTERBOX_EXAGGERATION", 0.35))
SAMPLE_RATE = 24000          # the model's own rate; assembly runs here too
CHUNK = int(os.environ.get("CHATTERBOX_CHUNK", 12))   # lines per process
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


def synth_all(lines: list[tuple[str, str]], force: bool = False) -> None:
    """Write one clone WAV per sentence, skipping lines already in this voice.

    Resuming matters here: the full run is over an hour, and a single bad line
    should cost one line, not the whole job.
    """
    SENT_DIR.mkdir(parents=True, exist_ok=True)
    bad = [sid for sid, _ in lines if not re.fullmatch(r"s\d\d\.\d+", sid)]
    if bad:
        raise ValueError(f"not sentence ids: {bad[:5]} — SENTENCES rows are "
                         "(section, sentence, text), key the worker on sentence")
    # A clip older than the sample cannot carry that sample's voice, so mtime
    # decides what to redo. A stamp file cannot: clearing it on start made every
    # restart redo the same first lines, and leaving it made a crashed run look
    # finished. shortcut: an exaggeration change is invisible here, pass --force.
    ref_t = VOICE_SAMPLE.stat().st_mtime
    todo = {sid: text for sid, text in lines
            if force or not (SENT_DIR / f"{sid}.wav").exists()
            or (SENT_DIR / f"{sid}.wav").stat().st_mtime <= ref_t}
    if not todo:
        print(f"all {len(lines)} lines already on disk")
        return
    print(f"synthesising {len(todo)} of {len(lines)} lines")
    if not CLONE_PY.exists():
        raise FileNotFoundError(
            f"missing clone env: {CLONE_PY}\n"
            "create it with:  uv venv ~/.local/share/chatterbox-venv && "
            "uv pip install --python ~/.local/share/chatterbox-venv/bin/python "
            "chatterbox-tts 'setuptools<81'")
    job = OUT_DIR / "_jobs.json"
    items = list(todo.items())
    # The MPS pool keeps growing with every line and hits its 9 GB cap part-way
    # through a run. Nothing was actually freed by empty_cache once the run
    # started, so each process now gets CHUNK lines and exits instead.
    for i in range(0, len(items), CHUNK):
        job.write_text(json.dumps(dict(items[i:i + CHUNK]), indent=1))
        # -I: the worker must not pick up modules from the current directory.
        r = subprocess.run([str(CLONE_PY), "-I", str(WORKER), str(job),
                            str(SENT_DIR), str(VOICE_SAMPLE), str(EXAGGERATION),
                            str(CHUNK)], check=False)
        if r.returncode != 0:
            job.unlink(missing_ok=True)
            raise RuntimeError(f"chatterbox worker exited {r.returncode} "
                               f"after {i} of {len(items)} lines")
        print(f"chunk {i // CHUNK + 1}: {min(i + CHUNK, len(items))} of "
              f"{len(items)} lines done", flush=True)
    job.unlink(missing_ok=True)
    missing = [sid for sid in todo if not (SENT_DIR / f"{sid}.wav").exists()]
    if missing:
        raise RuntimeError(f"worker wrote no audio for: {', '.join(missing[:5])}")


def to_pcm(wav: Path) -> bytes:
    """Silence-trimmed mono 16-bit PCM at SAMPLE_RATE, lead and tail alike."""
    trim = (f"silenceremove=start_periods=1:start_threshold={TRIM_THRESHOLD}"
            f":start_silence={TRIM_KEEP},areverse,"
            f"silenceremove=start_periods=1:start_threshold={TRIM_THRESHOLD}"
            f":start_silence={TRIM_KEEP},areverse")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav), "-af", trim,
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True)
    return r.stdout


def assemble(pcms: list[bytes], gap_s: float) -> tuple[bytes, list[tuple[float, float]]]:
    """Join clips with ``gap_s`` of silence; return the PCM and (start, end) pairs."""
    gap_samples = round(gap_s * SAMPLE_RATE)
    gap = b"\x00\x00" * gap_samples
    gap_dur = gap_samples / SAMPLE_RATE      # the gap the buffer really holds
    buf = bytearray()
    spans: list[tuple[float, float]] = []
    t = 0.0
    for i, pcm in enumerate(pcms):
        if i:
            buf += gap
            t += gap_dur
        dur = len(pcm) / 2 / SAMPLE_RATE
        spans.append((t, t + dur))
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


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    force = "--force" in sys.argv
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

    # SENTENCES rows are (section, sentence, text); the worker keys on sentence.
    synth_all([(sent_id, text) for _, sent_id, text in SENTENCES], force=force)

    total = 0.0
    for sec_id, _text in SECTIONS:
        clips = [(sid, to_pcm(SENT_DIR / f"{sid}.wav"))
                 for sid, _ in by_section[sec_id]]
        for sent_id, pcm in clips:
            assert pcm, f"{sent_id}: empty clip"
            print(f"  {sent_id}: {len(pcm) / 2 / SAMPLE_RATE:.2f}s")
        pcms = [pcm for _, pcm in clips]
        pcm_all, spans = assemble(pcms, gap_s)
        wav = OUT_DIR / f"{sec_id}.wav"
        write_wav(pcm_all, wav)
        # From the PCM, not ffprobe: the spans come from the same buffer, so a
        # rounded probe cannot report the last sentence ending past the track.
        dur = len(pcm_all) / 2 / SAMPLE_RATE
        total += dur
        timings["sections"][sec_id] = {
            "file": wav.name,
            "duration": dur,
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
    main()
