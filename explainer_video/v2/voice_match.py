"""How close is a candidate clip to the user's own voice?

Compares pitch (median f0) and long-term average spectrum against
audio/voice_sample.m4a. Lower numbers = closer to the real speaker.
"""
import subprocess
import sys

import numpy as np

SR = 16000
SAMPLE = "audio/voice_sample.m4a"
USER_F0 = 172.0  # measured in the earlier session


def pcm(path: str) -> np.ndarray:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def f0_series(x: np.ndarray) -> np.ndarray:
    """Autocorrelation pitch per 40 ms frame, voiced frames only."""
    hop, win = 640, 1024
    lo, hi = SR // 400, SR // 70  # 70-400 Hz
    out = []
    for i in range(0, len(x) - win, hop):
        f = x[i:i + win]
        if np.sqrt((f ** 2).mean()) < 0.02:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] < 0.3:
            continue
        out.append(SR / (lo + k))
    return np.array(out)


def ltas(x: np.ndarray, n: int = 24) -> np.ndarray:
    """Coarse spectral shape, in dB, one band per output slot."""
    hop = 512
    frames = [x[i:i + hop] for i in range(0, len(x) - hop, hop)]
    specs = np.abs(np.fft.rfft(np.array(frames) * np.hanning(hop), axis=1))
    prof = 20 * np.log10(specs.mean(0) + 1e-9)
    edges = np.linspace(0, len(prof), n + 1).astype(int)
    return np.array([prof[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])


def report(path: str, label: str) -> None:
    x = pcm(path)
    f = f0_series(x)
    s = ltas(pcm(SAMPLE))
    c = ltas(x)
    shape = float(np.corrcoef(s - s.mean(), c - c.mean())[0, 1])
    med = float(np.median(f)) if len(f) else float("nan")
    print(f"{label:12s} median f0 {med:6.1f} Hz (yours {USER_F0}, "
          f"off {abs(med - USER_F0):5.1f})   spectral match {shape:5.3f}")


if __name__ == "__main__":
    report(SAMPLE, "YOUR SAMPLE")
    for p in sys.argv[1:]:
        report(p, p.split("/")[-1][:12])
