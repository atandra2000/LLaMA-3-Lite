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
from pathlib import Path

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

    # ---- 2. build mixed audio: voice + ducked music bed --------------------
    # base silence + one adelayed narration per section, mixed without scaling
    assert BED.exists(), f"missing music bed: {BED}"
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
    filters.append(f"[{len(ORDER) + 1}:a]aresample=48000,"
                   f"aformat=channel_layouts=stereo,"
                   f"volume={bed_gain:.2f}dB,"
                   f"afade=t=in:st=0:d=1.5,"
                   f"afade=t=out:st={total - 1.5:.3f}:d=1.5[bedv]")
    # duck the bed under the voice, then mix and clip-protect
    filters.append("[bedv][voice]sidechaincompress=threshold=0.015:ratio=12:"
                   "attack=20:release=300:makeup=1[ducked]")
    filters.append("[voice][ducked]amix=inputs=2:duration=first:normalize=0,"
                   "alimiter=limit=0.95[outa]")
    mix_wav = HERE / "_mix.wav"
    run(["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(filters),
         "-map", "[outa]", "-c:a", "pcm_s16le", str(mix_wav)])

    # two-pass loudnorm (linear) on the final mix
    m = measure_loudnorm(mix_wav)
    ln = ("loudnorm=I=-16:TP=-1.6:LRA=11:"
          f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
          f"offset={m['target_offset']}:linear=true,"
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

    chap_lines = [";FFMETADATA1"]
    for i, k in enumerate(ORDER):
        end = starts[i] + durs[i]
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
