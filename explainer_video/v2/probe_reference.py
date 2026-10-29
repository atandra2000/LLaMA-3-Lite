"""Decisive probe: does heygen-voice-1-turbo actually use the reference audio?

Sends one sentence with three different references and compares the bytes:
  A. your sample, seconds 0-20
  B. your sample, seconds 20-40   (same speaker, different audio)
  C. an unrelated voice           (the old edge-tts track)
If all three are near-identical the reference is ignored; if A and B agree with
each other but differ from C the reference is read.
"""
import base64
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
SAMPLE = ROOT / "audio" / "voice_sample.m4a"
KEY = Path.home() / ".openrouter_key"
URL = "https://openrouter.ai/api/v1/audio/speech"
MODEL = "heygen/heygen-voice-1-turbo:free"
TEXT = "Rotary position embeddings turn each token into a dial."


def other_voice() -> Path:
    old = sorted((ROOT / "audio" / "_old").glob("*.wav"))
    if not old:
        raise SystemExit("no old voice track to compare against")
    return old[0]


def ref(src: Path, ss: float, t: float = 20.0) -> str:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(ss), "-i", str(src),
         "-t", str(t), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
         "-f", "wav", "-"], capture_output=True, check=True)
    return "data:audio/wav;base64," + base64.b64encode(r.stdout).decode()


def call(label: str, data_ref: str) -> bytes:
    body = {"model": MODEL, "input": TEXT, "response_format": "pcm",
            "input_references": [{"type": "input_audio",
                                  "input_audio": {"data": data_ref, "format": "wav"}}]}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY.read_text().strip(),
                 "Content-Type": "application/json", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        audio, ctype = resp.read(), resp.headers.get("content-type", "")
    rate = int(ctype.split("rate=")[1].split(";")[0]) if "rate=" in ctype else 44100
    out = Path(f"/tmp/probe_{label}.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(rate),
                    "-ac", "1", "-i", "-", str(out)], input=audio, check=True)
    print(f"  {label:8s} {len(audio) / 2 / rate:5.2f}s  "
          f"sha={hashlib.sha256(audio).hexdigest()[:16]}  -> {out}")
    return audio


if __name__ == "__main__":
    other = other_voice()
    print(f"model {MODEL}\ntext  {TEXT!r}\nother {other.name}\n")
    a = call("A_user00", ref(SAMPLE, 0.0))
    b = call("B_user20", ref(SAMPLE, 20.0))
    c = call("C_other", ref(other, 0.0))
    print("\nA==B (same speaker):", a == b)
    print("A==C (different speaker):", a == c)
