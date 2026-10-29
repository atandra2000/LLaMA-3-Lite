"""Assemble the final LLaMA-3-Lite explainer MP4.

1. Probe each rendered scene's actual duration.
2. Concatenate scene videos (single clean x264 encode).
3. Place each section's narration at its scene's real start + LEAD
   (adelay + amix), layer the CC0 music bed under it (sidechain duck),
   two-pass linear loudnorm to -16 LUFS.
4. Mux video + audio, embed chapter markers, faststart.

Run after the 1080p render:  uv run python build_full_video.py
"""
import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
# QA renders can point at a smaller scene dir instead of the 1080p master.
MEDIA = Path(os.environ.get(
    "SCENE_MEDIA", HERE / "media" / "videos" / "render_scenes" / "1080p30"))
AUDIO_DIR = HERE / "audio"
BED = HERE / "assets" / "music" / "bed.mp3"
BED_LUFS = -28.0
OUT = HERE / "llama3_lite_architecture_explainer.mp4"

# The only timing source: exact per-sentence spans from the assembled tracks.
TIMINGS = json.loads((AUDIO_DIR / "timings.json").read_text())
LEAD = TIMINGS["lead"]
ORDER = list(TIMINGS["sections"])
# One scene per section, emitted by render_scenes.py as SectionNN.mp4.
SCENE_VIDS = [MEDIA / f"Section{i:02d}.mp4" for i in range(1, len(ORDER) + 1)]
# Chapter titles come from narration.yaml — never typed twice.
NARRATION = yaml.safe_load((HERE / "narration.yaml").read_text(encoding="utf-8"))
TITLES = [s["title"] for s in NARRATION["sections"]]


def build_sfx_track(vid_starts: list[float], total: float, out_wav: Path) -> None:
    """Synthesize subtle 3Blue1Brown-style warm marimba/harmonic plucks at cue boundaries."""
    sr = 48000
    n_samples = int(round(total * sr))
    buf = np.zeros((n_samples, 2), dtype=np.float32)

    # Warm pentatonic/harmonic palette (C4, D4, E4, G4, A4, C5)
    notes = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    pluck_dur = 0.14
    t_pluck = np.arange(int(sr * pluck_dur), dtype=np.float32) / sr
    env = (1.0 - np.exp(-t_pluck / 0.004)) * np.exp(-t_pluck / 0.032)

    cue_idx = 0
    for i, k in enumerate(ORDER):
        base = vid_starts[i] + LEAD
        for sid, span in TIMINGS["sections"][k]["sentences"].items():
            f0 = notes[cue_idx % len(notes)]
            # First cue of a chapter gets a slightly fuller fifth interval
            if sid.endswith(".1"):
                sig = (0.70 * np.sin(2 * np.pi * f0 * t_pluck)
                       + 0.30 * np.sin(2 * np.pi * (f0 * 1.5) * t_pluck)) * env * 0.016
            else:
                sig = (0.80 * np.sin(2 * np.pi * f0 * t_pluck)
                       + 0.20 * np.sin(2 * np.pi * (f0 * 2.0) * t_pluck)) * env * 0.010
            idx0 = int(round((base + span["start"]) * sr))
            idx1 = min(n_samples, idx0 + len(sig))
            if idx1 > idx0 >= 0:
                seg = sig[:idx1 - idx0]
                pan = 0.5 + 0.15 * np.sin(cue_idx * 1.3)
                buf[idx0:idx1, 0] += seg * (1.0 - pan) * 1.4
                buf[idx0:idx1, 1] += seg * pan * 1.4
            cue_idx += 1

    pcm = np.clip(buf * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _cap_stamp(sec: float, comma: bool) -> str:
    """SRT uses a comma before the milliseconds; VTT uses a full stop."""
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_captions(vid_starts: list[float]) -> int:
    """One cue per spoken sentence, at its absolute position in the master."""
    srt, vtt = ["\n"], ["WEBVTT", "\n"]
    n = 0
    for i, k in enumerate(ORDER):
        base = vid_starts[i] + LEAD
        for sid, span in TIMINGS["sections"][k]["sentences"].items():
            n += 1
            a = _cap_stamp(base + span["start"], True)
            b = _cap_stamp(base + span["end"], True)
            srt += [str(n), f"{a} --> {b}", span["text"].strip(), ""]
            a2 = _cap_stamp(base + span["start"], False)
            b2 = _cap_stamp(base + span["end"], False)
            vtt += [f"{a2} --> {b2}", span["text"].strip(), ""]
    OUT.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    OUT.with_suffix(".vtt").write_text("\n".join(vtt), encoding="utf-8")
    return n


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd[:6])}...\n{r.stderr[-2000:]}")


def parse_loudnorm_json(stderr: str) -> dict:
    """Extract the JSON block that loudnorm's measure pass prints on stderr."""
    i = stderr.rfind("\n{")
    if i < 0:
        raise RuntimeError(f"loudnorm printed no JSON block:\n{stderr[-500:]}")
    # raw_decode: ffmpeg may print more lines after the closing brace
    obj, _ = json.JSONDecoder().raw_decode(stderr[i + 1:])
    return obj


def measure_loudnorm(path: Path) -> dict:
    """First loudnorm pass over a file: returns its measured_* values."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         "loudnorm=I=-16:TP=-1.6:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    return parse_loudnorm_json(r.stderr)


def main() -> None:
    # Catch a stale timings.json before anything is encoded: the narration and
    # the cue map must describe the same set of sections.
    if len(TITLES) != len(ORDER):
        raise RuntimeError(
            f"narration.yaml has {len(TITLES)} sections but timings.json has "
            f"{len(ORDER)} — re-run generate_audio.py")
    if [s["id"] for s in NARRATION["sections"]] != ORDER:
        raise RuntimeError(
            "section order differs between narration.yaml and timings.json — "
            "re-run generate_audio.py")
    for v in SCENE_VIDS:
        assert v.exists(), f"missing {v}"

    durs = [probe_duration(v) for v in SCENE_VIDS]
    starts = [sum(durs[:i]) for i in range(len(durs))]
    total = sum(durs)
    print("scene durations:", [round(d, 2) for d in durs])
    print(f"total video: {total:.2f}s = {total / 60:.2f} min")

    # Timing rule: the narration must fit its scene with room to spare.
    for i, k in enumerate(ORDER):
        audio = TIMINGS["sections"][k]["duration"]
        if audio + LEAD > durs[i] - 0.1:
            raise RuntimeError(
                f"{k}: narration {audio:.2f}s + LEAD {LEAD:.2f}s does not fit "
                f"scene {durs[i]:.2f}s (needs {audio + LEAD + 0.1:.2f}s)")

    # ---- 1. concat video (re-encode once for uniform params) ---------------
    concat_list = HERE / "concat_list.txt"
    concat_list.write_text("".join(f"file '{v}'\n" for v in SCENE_VIDS))
    silent = HERE / "_video_silent.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(concat_list), "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", str(silent)])

    # ---- 2. build mixed audio: voice + ducked music bed + tactile sfx -------
    # base silence + one adelayed narration per section, mixed without scaling
    assert BED.exists(), f"missing music bed: {BED}"
    sfx_wav = HERE / "_sfx.wav"
    build_sfx_track(starts, total, sfx_wav)
    bed_gain = BED_LUFS - float(measure_loudnorm(BED)["input_i"])
    inputs: list[str] = ["-f", "lavfi", "-t", f"{total:.3f}",
                         "-i", "anullsrc=r=48000:cl=stereo"]
    filters = []
    for i, k in enumerate(ORDER):
        inputs += ["-i", str(AUDIO_DIR / TIMINGS["sections"][k]["file"])]
        ms = int(round((starts[i] + LEAD) * 1000))
        filters.append(f"[{i + 1}:a]aresample=48000,aformat=channel_layouts=stereo,"
                       f"adelay={ms}|{ms}[n{i}]")
    mix_in = "[0:a]" + "".join(f"[n{i}]" for i in range(len(ORDER)))
    filters.append(f"{mix_in}amix=inputs={len(ORDER) + 1}:"
                   f"duration=first:normalize=0[voice]")
    # loop the bed over the whole runtime, levelled to -28 LUFS, 1.5 s fades
    inputs += ["-stream_loop", "-1", "-t", f"{total:.3f}", "-i", str(BED)]
    inputs += ["-i", str(sfx_wav)]
    bed_idx = len(ORDER) + 1
    sfx_idx = len(ORDER) + 2
    filters.append(f"[{bed_idx}:a]aresample=48000,"
                   f"aformat=channel_layouts=stereo,"
                   f"volume={bed_gain:.2f}dB,"
                   f"afade=t=in:st=0:d=1.5,"
                   f"afade=t=out:st={total - 1.5:.3f}:d=1.5[bedv]")
    filters.append(f"[{sfx_idx}:a]aresample=48000,aformat=channel_layouts=stereo[sfxv]")
    # duck the bed under the voice, then mix voice + ducked bed + subtle sfx
    filters.append("[bedv][voice]sidechaincompress=threshold=0.015:ratio=12:"
                   "attack=20:release=300:makeup=1[ducked]")
    filters.append("[voice][ducked][sfxv]amix=inputs=3:duration=first:normalize=0,"
                   "alimiter=limit=0.95[outa]")
    mix_wav = HERE / "_mix.wav"
    run(["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(filters),
         "-map", "[outa]", "-c:a", "pcm_s16le", str(mix_wav)])
    sfx_wav.unlink(missing_ok=True)

    # Two-pass loudnorm (linear) on the final mix. The I target sits above the
    # -16 gate because alimiter below removes loudness with the peaks.
    m = measure_loudnorm(mix_wav)
    # linear=true sets the gain to hit I=-16 and then leaves the peaks alone,
    # so a TP target in the same chain is not enforced. alimiter is what actually
    # holds the ceiling. 0.72 is -2.85 dBFS, because the AAC encode lifts
    # inter-sample peaks by about a decibel on top of whatever the limiter sets.
    ln = ("loudnorm=I=-15.3:TP=-1.6:LRA=11:"
          f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
          f"offset={m['target_offset']}:linear=true,"
          "alimiter=limit=0.72:level=disabled,"
          "aresample=48000")
    narr = HERE / "_narration.m4a"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(mix_wav),
         "-af", ln, "-c:a", "aac", "-b:a", "192k", str(narr)])

    # ---- 3. chapters ---------------------------------------------------------
    def ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s = rem / 1000
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    # The AAC encode pads the tail, so the container runs past the scene sum.
    # The last chapter has to cover that or the file ends mid-chapter.
    audio_end = probe_duration(narr)
    chap_lines = [";FFMETADATA1"]
    for i, k in enumerate(ORDER):
        end = starts[i] + durs[i]
        if i == len(ORDER) - 1:
            end = max(end, audio_end)
        chap_lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                       f"START={int(round(starts[i] * 1000))}",
                       f"END={int(round(end * 1000))}",
                       f"title={i + 1:02d} · {TITLES[i]}"]
    chap_file = HERE / "_chapters.txt"
    chap_file.write_text("\n".join(chap_lines))

    # ---- 4. mux ---------------------------------------------------------------
    run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(narr),
         "-i", str(chap_file), "-map", "0:v", "-map", "1:a",
         "-map_metadata", "2", "-c:v", "copy", "-c:a", "copy",
         "-movflags", "+faststart", str(OUT)])

    n_cues = write_captions(starts)
    print(f"captions: {OUT.with_suffix('.srt').name} + "
          f"{OUT.with_suffix('.vtt').name}  ({n_cues} cues)")

    final = probe_duration(OUT)
    a_drift = abs(final - total)
    print(f"final: {OUT.name}  {final:.2f}s  (a/v drift {a_drift:.3f}s)")
    silent.unlink(missing_ok=True)
    narr.unlink(missing_ok=True)
    mix_wav.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)
    chap_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
