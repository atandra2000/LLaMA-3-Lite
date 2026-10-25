# LLaMA-3-Lite Architecture Explainer — v2

Motion-graphics explainer video: **`llama3_lite_architecture_explainer.mp4`**
(5 min 01 s · 1920×1080 · 30 fps · H.264 + AAC · chapter markers embedded).

**Design language ("Aurora", v3):** ambient deep-indigo/teal gradient background
with drifting light orbs and a faint dot grid, glass panels and pill chips,
layered glow accents (iris / blue / gold / mint / violet / rose), bundled
**Inter** display type with **Barlow Semi Condensed** tracked technical labels,
fluid entrances, glow pulses and data-flow flashes. Defined in
`render_scenes.py` (`_build_ambient`, `card`, `pill`, `glow_dot`, `glow_ring`,
`section_header`).

Voiceover script: **`VOICEOVER_SCRIPT.md`** — generated from `narration.yaml` by
`tools/build_script_doc.py`; never edit it by hand.

## Content (9 chapters, architecture only — no data/training)

1. The model as a function (tokens → next-token probabilities)
2. Tokens become vectors (tokenizer, embedding lookup, meaning space)
3. The residual stream (read → transform → add back)
4. RMSNorm (root-mean-square math, vs LayerNorm, ε = 1e-5)
5. Self-attention (Q/K/V, scaled dot-product, causal mask, softmax, 8 heads)
6. Rotary positions — RoPE (pair rotations, frequency ladder, relative-distance property, θ = 500 000)
7. Grouped-query attention & QK-norm (8 Q : 4 KV, KV-cache halving)
8. SwiGLU feed-forward (gate/up split, SiLU plot, d_ff 4096)
9. Unembedding & autoregression (final norm, logits, softmax, sampling loop, recap)

## Where the numbers come from

`facts.json` is the only numeric source, built by `tools/build_facts.py` from
`config.py` and the LLaMA-3 tokenizer and gated by `tools/check_facts.py`
(byte-identical rebuild + independent recomputation):

| Fact | Value |
|---|---|
| Parameters | 514,364,416 (≈515M) — 513,840,128 at the config's 128,000 vocab |
| Vocabulary | **128,256** (`max(config, len(tokenizer))`, as `train.py` sizes the run) |
| Layers / d_model / heads | 16 / 1024 / 8 Q : 4 KV |
| Head dim / d_ff / context | 128 / 4096 / 2048 |
| RoPE θ / ε | 500,000 / 1e-5 |
| Demo token IDs | 791 · 8415 · 7731 · 389 · 279 · 5634 |

Phase 1a (`PLAN.md`) wired those facts into the scenes: the spec chips, token
chips, matrix dims, attention demo, cache counts and recap line all read
`facts.json` at import. The delivered v2 MP4 predates this wiring (it still shows
`128K` and the old token IDs); re-render to pick it up. The audio also still
carries the pre-1a lines for s01.2, s07.7 and s08.6 — Phase 2 re-records per
sentence from `narration.yaml`.

## Folder layout

```
llama3_lite_architecture_explainer.mp4   deliverable (5:01, 1080p30, pre-Phase-0 facts)
PLAN.md                                  remediation plan (Phase 0 done)
facts.json                               generated: every number on screen
narration.yaml                           generated-from source: spoken script + cue ids
VOICEOVER_SCRIPT.md                      generated: script + cue map
generate_audio.py        → audio/        Edge-TTS narration (text from narration.yaml)
render_scenes.py         → media/…/1080p30/  9 Manim scenes
build_full_video.py                      concat + audio sync + chapters → final MP4
assets/fonts/                            bundled Inter + Barlow Semi Condensed (OFL)
tools/                                   build_facts, check_facts, build_script_doc, lint_script
tests/                                   Phase 0 + 1a gates (pytest)
pyproject.toml / uv.lock / .venv/        uv-managed render environment (Python 3.12)
```

## Rebuild pipeline

```bash
uv sync                                              # manim + edge-tts + typst + tokenizers
uv run python tools/build_facts.py                   # config.py + tokenizer -> facts.json
uv run python tools/check_facts.py                   # fact gate (must exit 0)
uv run python tools/lint_script.py                   # <=22 words per sentence, cue ids
uv run python tools/build_script_doc.py              # -> VOICEOVER_SCRIPT.md
uv run pytest tests -q                               # Phase 0 + 1a gates
uv run python generate_audio.py                      # TTS narration -> audio/*.mp3 + durations.json
uv run manim render -qh --fps 30 --media_dir ./media \
    --disable_caching render_scenes.py \
    Section01 Section02 Section03 Section04 Section05 \
    Section06 Section07 Section08 Section09          # ~20 min
uv run python build_full_video.py                    # concat + sync audio + chapters -> final MP4
```

Timing contract: every scene fills exactly `narration + 0.30 s lead + 0.40 s tail`
seconds (tracked in `Section.sync_end()`); narration is placed at each scene's
measured start + 0.30 s, so A/V sync is exact by construction. The v2 cut does
not fully satisfy this (scenes overrun by 0.05–0.44 s); Phase 2 replaces
fraction-of-narration beats with per-sentence cues from `audio/timings.json`.

Type: nothing on screen may be below 12 pt — `render_scenes.py:pt()` raises, and
`tests/test_phase0.py` fails on a sub-12 literal.

Facts: no model figure is typed in `render_scenes.py`. `tests/test_phase1a.py`
fails if the old `128K`/token-ID literals return, if the attention weights stop
being the masked softmax of the scaled scores, if SwiGLU stops being a product,
or if the KV-cache stack stops halving by bar count. Set `USE_SYSTEM_FONTS=1` to
render with Avenir Next / DIN Alternate for diffing against the v2 baseline.

Gotcha worth knowing: `Scene.wait()` in Manim calls `self.play(Wait(...))`
internally — a `play()` override must not force a default `run_time`, or every
wait collapses to 1 s (this bit the first build; see `Section.play`).
