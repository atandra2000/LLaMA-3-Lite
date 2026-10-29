"""Clone the user's voice with Chatterbox, one WAV per sentence.

Runs in the isolation venv that holds torch and chatterbox
(``~/.local/share/chatterbox-venv/bin/python``); ``generate_audio.py`` calls it
and does the trimming and assembly itself.

The model weights are large and the load costs about a minute, so this takes a
JSON job list and generates every line in one process rather than starting up
once per sentence.

Usage:
    python chatterbox_worker.py jobs.json out_dir ref_audio [exaggeration]
"""
import json
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    jobs_path, out_dir, ref = Path(argv[1]), Path(argv[2]), argv[3]
    exaggeration = float(argv[4]) if len(argv) > 4 else 0.35
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    # MPS holds about 9 GB on this machine and the model keeps about 5 GB of it
    # resident. Stop after max_lines and let the caller start a fresh process;
    # a 167-line run died of MPS OOM at line 41.
    max_lines = int(argv[5]) if len(argv) > 5 else len(jobs)

    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device {dev}  lines {len(jobs)}  exaggeration {exaggeration}",
          flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=dev)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (sid, text) in enumerate(jobs.items(), 1):
        if i > max_lines:
            print(f"STOPPED after {max_lines} lines; restart for the rest",
                  flush=True)
            return 0
        t = time.time()
        wav = model.generate(text, audio_prompt_path=ref,
                             exaggeration=exaggeration)
        ta.save(str(out_dir / f"{sid}.wav"), wav.cpu(), model.sr)
        # Freed per line: without this the MPS pool grows until it OOMs
        # part-way through a chunk, even though every line looks identical.
        if dev == "mps":
            torch.mps.empty_cache()
            print(f"  mps {torch.mps.current_allocated_memory() / 1e9:.2f} GB",
                  flush=True)
        secs = wav.shape[-1] / model.sr
        print(f"[{i}/{len(jobs)}] {sid} {secs:.2f}s audio in {time.time() - t:.1f}s",
              flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
