# LLaMA-3-Lite explainer — v3 remediation plan

**Status:** planning complete · implementation not started
**Baseline:** `main` @ `a68c2c6` · v2 master `llama3_lite_architecture_explainer.mp4` (301.2 s, 5:01, 1080p30)
**Scope:** all review findings — correctness, layout, rhythm, accessibility, pipeline, positioning
**Tracking:** check boxes as tasks land; update the status table and append to the log at the bottom.

---

## 1. Why this plan exists

The v2 cut is a strong design artifact with a correct block-level narrative, but it is not
publishable as-is:

- **Six factual deltas.** Five of six token IDs are wrong; the vocabulary is shown as 128,000
  while every real run builds 128,256; QK-norm is presented as LLaMA 3 lineage when LLaMA 3 has
  no QK-norm; the attention demo's softmax weights do not follow from its scores (and give mass
  to a masked key); SwiGLU gating is shown as a hard threshold.
- **Two broken key beats.** The final attention equation is drawn on top of the section header
  and the previous formula (2:36); the generation-loop arrow paints a filled wedge over the
  logits (4:44).
- **5 safe-area/collision defects** (ε pill clipped at 1:50, `× 16 blocks` on the residual band,
  FFN card under the watermark, and two captions flush to the frame edge).
- **~70% of the runtime is a still frame** (87 holds ≥1 s, 210.6 s of 301 s, longest 5.7 s),
  because reveals are timed by fraction of narration rather than by sentence.
- **No captions, no music, 8–10 pt labels, system-only fonts, and a timing contract the file
  does not satisfy** (scenes overrun by 0.05–0.44 s; planned total 299.2 s vs delivered 301.2 s).
- **The project's actual headline — the 78% memory stack — is absent**, and the end card
  ("and nothing else") actively closes the door on it.

The plan fixes correctness and legibility first (Phase 1), then rhythm and access (Phases 2–3),
then the systems story (Phase 4, gated).

---

## 2. Definition of done (machine-checkable)

| Gate | Now | Target | Enforced by |
|---|---|---|---|
| Fact deltas vs `config.py` / tokenizer | 6 | 0 | `tools/check_facts.py` |
| Text outside safe rect | 5 | 0 | bbox assertion in every scene |
| Key-beat collisions (2:36, 4:44) | 2 | 0 | golden frames + contact sheets |
| Per-scene budget error | ≤0.44 s | ≤0.033 s | `sync_end()` raises |
| Frozen ≥1 s share | 69.9% | ≤35% | `tools/verify_video.py` |
| Longest hold | 5.7 s | ≤3.5 s | same |
| Captions | none | SRT + VTT (+ optional burn-in) | `verify_video.py` |
| Loudness | −16.1 LUFS | −16.0 ±0.5, TP −1.5 | `ebur128` parse |
| Cross-machine render | macOS-only fonts | bundled or asserted | preflight |
| Discoverability | orphaned | linked from README + docs, cover image | manual |

---

## 3. Decisions

Defaults are chosen so work can start; revisit only if the A/B fails.

| # | Fork | Default | Why |
|---|---|---|---|
| D1 | Fonts | Bundle **Inter** (display) + **Barlow Semi Condensed** (tracked technical); register at import; keep Avenir/DIN behind `USE_SYSTEM_FONTS=1` | Removes the macOS-only failure; costs one type retune |
| D2 | TTS timing | **Per-sentence synthesis** → exact cues, re-recordable lines | Whole-section + forced alignment is fuzzier and needs Whisper |
| D3 | Scope | **Part 1 now**, Part 2 (systems) after the Phase 3 gate | Part 2 is a new production, not a fix |

Before committing D2, A/B the first 20 s against the current voice. If per-sentence prosody
flattens, fall back to whole-section audio with edge-tts `WordBoundary` alignment (no Whisper).

---

## 4. Target architecture

```
config.py ─┐
tokenizer ─┴─► tools/build_facts.py ─► facts.json ─┬─► narration.yaml ─► TTS ─► audio/timings.json
                                                   └─► render_scenes.py ──────────┘
                                                          │
                                     tools/verify_video.py ◄─ build_full_video.py ─► MP4 + SRT/VTT
```

New files: `facts.json`, `narration.yaml`, `tools/{build_facts,check_facts,build_script_doc,lint_script,build_timeline,verify_video,make_contact_sheet}.py`, `assets/fonts/`, `assets/music/bed.mp3`, `tests/`.

**Single-source rules**

- No number is typed in `render_scenes.py`; every figure comes from `facts.json`.
- No narration sentence exists outside `narration.yaml`; `VOICEOVER_SCRIPT.md` becomes generated.
- `audio/timings.json` is the only timing source; no `beat(fraction)` calls remain.

`facts.json` (abridged):

```json
{ "vocab": 128256, "vocab_label": "128,256", "params": 514364416,
  "tokens": {"sentence": "The cat sat on the mat",
             "items": [{"text": "The", "id": 791}, {"text": "cat", "id": 8415}, "..."]},
  "attention": {"raw": [[...]], "scaled": [[...]], "weights": [[...]], "mask": "upper"},
  "rope": {"pairs": 64, "theta": 500000} }
```

---

## 5. Phases

### Phase 0 — Foundations · 2 days

- [ ] **T0.1 Fact builder** — import `config.get_config()`, load the Llama-3 tokenizer, compute
      params (513.8 M @128,000 / 514.4 M @128,256), demo token IDs, attention
      raw → scaled → masked softmax, RoPE pair count. → `facts.json` (`tools/build_facts.py`, 0.5 d)
- [ ] **T0.2 Narration single source** — per-sentence `id` + `visual` notes; generate the
      Markdown script doc from it. → `narration.yaml`, `tools/build_script_doc.py` (0.5 d)
- [ ] **T0.3 Script lint + sentence split** — max 22 words per sentence; rewrite the long
      section-5 sentences into 12–18 word units. → `tools/lint_script.py` (1 d)
- [ ] **T0.4 Fonts** — bundle, register, retune the type scale (no label below 12 pt).
      → `assets/fonts/`, `render_scenes.py` header constants (0.5 d)

**Gate:** `python tools/check_facts.py` exits 0 after a rebuild; deleting `facts.json` and
rebuilding produces a byte-identical file.

### Phase 1 — Correctness and layout · 4 days

**1a. Facts into the scenes**

- [ ] **T1.1 Wire facts** — spec chips, token chips, matrix dims, `128K` → `128,256`,
      `515M` → `≈515M`. → `Section01`, `Section02`, `Section09.construct`
- [ ] **T1.2 Attention demo** — raw scores → `÷√d_k` strip → masked softmax → weights; the
      masked key gets `0.00`, not `0.05`. → `Section05.construct`
- [ ] **T1.3 QK-norm caveat** — narration + caption: "a Lite addition — LLaMA 3 itself has none".
      → `narration.yaml` s07, `Section07.construct`
- [ ] **T1.4 SwiGLU gating** — two rows (SiLU gate, up) → product row; drop the 0.3 "survival"
      threshold; narration says "scales each feature". → `Section08.construct`
- [ ] **T1.5 KV cache** — halve **height** (8 vs 16 bars), per the script's own wording; label
      head counts. → `Section07.construct`
- [ ] **T1.6 Wording** — "word" → "token" in the opening line. → `narration.yaml` s01

**1b. Layout defects**

- [ ] **T1.7 §4 relayout** — geometry inset + LN/RMS/ε chips inside y ≥ −3.45; remove dead
      `Circumscribe(mean_chip)`; remove or connect the floating √.
- [ ] **T1.8 §5 banner** — move `wcap` below the bars; fade the header before the banner, then
      place the banner at y ≈ 2.45; move `hcap` under the heads on two lines.
- [ ] **T1.9 §9 loop** — stroke-only (`fill_opacity=0`), routed **above** the token row; keep all
      four probability bars and draw the chip from a copy.
- [ ] **T1.10 Layout sweep** — §3 `× 16 blocks` off the band edge; §8 FFN card up to y = −3.1;
      §7 caption up; §1 header rule; §1 circle block 1 not 12; §9 recap brace over the two
      middle pills; watermark only on the end card; progress bar at y = −3.85 with chapter ticks.
- [ ] **T1.11 §6 RoPE** — delete the unlabeled 3×3 grid or make it a labeled score matrix;
      animate m continuously 0→7; constant-frequency rotation with period labels.
- [ ] **T1.12 Safe-rect guard** — `SAFE = (-6.7, 6.7, -3.6, 3.6)`; after each beat assert every
      text bbox is inside and log overlaps of top-level groups.

**Gate:** 480p render with `--check_layout` produces zero violations; one contact sheet per
scene reviewed.

### Phase 2 — Timeline and rhythm · 3 days

- [ ] **T2.1 Per-sentence audio** — trim silence, assemble with 200 ms gaps, emit
      `audio/timings.json` with exact sentence start/end. (1 d)
- [ ] **T2.2 Cue API** — `self.cue("s07.2")` waits until that sentence's offset + `LEAD`;
      replace all `beat(frac)` sites scene by scene; delete `beat()` at the end. (1.5 d)
- [ ] **T2.3 Motion pass** — ambient pulse updater, cue-synced `flow` pulses, a moving
      highlight for the spoken clause; no hold > 3.5 s. (0.75 d)
- [ ] **T2.4 Measure and iterate** — `freezedetect`; repeat until the freeze gate is met. (0.5 d)

**Timing rules**

- `budget = LEAD + section_audio + TAIL`, computed from `timings.json`.
- `sync_end()` uses the renderer clock (`self.time`), waits the remainder, then **raises** if
  `abs(self.time - budget) > 1/30`.
- `build_full_video.py` fails if `section_audio + LEAD > scene_duration - 0.1`.

**Gate:** total duration = sum of budgets ±0.1 s; no scene overruns; freeze gate met.

### Phase 3 — Sound, captions, packaging · 2.5 days

- [ ] **T3.1 Music bed** — CC0 track, −28 LUFS, sidechain-ducked under voice, 1.5 s fades;
      two-pass `loudnorm` (linear) on the final mix. (0.75 d)
- [ ] **T3.2 Captions** — SRT + VTT from timings, ≤42 chars/line, ≤2 lines, ≤17 cps, clause
      splits; `--burn` option. (0.75 d)
- [ ] **T3.3 Packaging** — cover/thumbnail 1280×720; README + `docs/README.md` links with
      poster; end-card repo URL + Part 2 teaser; hold 3 s then fade to black. (0.5 d)
- [ ] **T3.4 Verification + CI** — `verify_video.py`, `Makefile`, CI job (facts/timeline/bbox on
      480p, full render on tag). (0.5 d)

**Audio decisions:** keep voice mono; bed stereo; final stereo only if the bed is present. Stop
upsampling 24 kHz voice to 48 kHz stereo at 192 kbps.

**Gate:** `make verify` green on the final MP4; fresh clone on Linux reproduces frame 150.

### Phase 4 — Systems chapter (optional) · 5–8 days

Proceed only after the Phase 3 gate is signed off.

- [ ] **T4.1 Outline + script** — 90–120 s, same design system. (1 d)
- [ ] **T4.2 Render** — new `render_systems.py`: 92 GB problem → checkpointing → chunked CE →
      mmap cache → BF16/FA2 → fused SwiGLU/TF32 → 78% result → throughput. (3–5 d)
- [ ] **T4.3 Integrate** — Part 2 handoff on the end card; retire "and nothing else". (0.5 d)

---

## 6. Verification harness

```
make facts     # rebuild facts.json + diff
make check     # facts, timeline, bbox at 480p (~2 min)
make video     # 9 scenes at 1080p30 + assemble
make verify    # verify_video.py
make sheet     # 12 frames/scene contact sheets
```

`verify_video.py` asserts: duration; 9 chapters with expected starts; audio present;
loudness/TP; freeze share and longest hold; cue monotonicity; every sentence fits its scene.
Golden frames (3 per scene) live in `media/golden/`; a pixel diff > 2% fails.

---

## 7. Calendar and critical path

| Week | Work | Exit |
|---|---|---|
| 1 | Phase 0 + Phase 1 | Facts clean, collisions gone, guard green |
| 2 | Phase 2 + Phase 3 | Publishable Part 1 |
| 3–4 | Phase 4 | Systems chapter / Part 2 |

Critical path: T0.1 → T0.3 → T2.1 → T2.2 → T2.4 → T3.2. Scene relayout (T1.7–T1.11) runs in
parallel with the TTS work.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Font swap breaks every absolute layout | Do it in Phase 0; retune once; the guard catches the rest |
| Per-sentence TTS sounds choppy | 20 s A/B before committing; whole-section fallback |
| Cue migration touches every scene | Scene-by-scene, keep v2 renders for diffing, commit per scene |
| Music licensing | CC0 only; log the source in `assets/music/CREDITS.md` |
| Caption length | The T0.3 sentence split is a prerequisite, not optional |
| Part 2 erodes Part 1 quality | Hard gate after Phase 3 |

---

## 9. Findings index (what each task closes)

| ID | Finding | Evidence |
|---|---|---|
| C1 | 5 of 6 token IDs wrong | 791 / 9427 / 7727 / 297 / 7560 / 1246 vs 791 / 8415 / 7731 / 389 / 279 / 5634 |
| C2 | QK-norm attributed to LLaMA 3 | LLaMA 3 uses RMSNorm/RoPE/GQA/SwiGLU only; QK-norm is a Lite addition |
| C3 | Vocab 128,000 vs 128,256 | `train.py` sizes to `max(config, len(tokenizer))` |
| C4 | Softmax weights don't follow scores; masked key gets 0.05 | 0.1/1.4/0.6 ÷ √128 → softmax ≈ 0.32/0.35/0.33 |
| C5 | SwiGLU shown as a 0.3 threshold | Gating is multiplicative, not selective |
| C6 | KV cache "EXACTLY ½" encoded as width | `cache_stack(1.4)` vs `(0.7)`, both 16 bars |
| C7 | "word" vs "token" | Opening sentence |
| L1 | Attention banner overlays header + prior formula | 2:36 |
| L2 | Loop arrow fills a rose wedge over logits | 4:44 |
| L3 | ε pill / geometry caption below frame | 1:50 |
| L4 | Formula card overlaps LayerNorm pill | 1:50 |
| L5 | `W_O` caption flush to edge + watermark | 2:36 |
| L6 | `× 16 blocks` on residual band | 1:20 |
| L7 | Orphan grid; position arrow frozen at m=3 | 2:43–3:15 |
| L8 | Empty opening frame; end card fades to nothing | 0:00 / 5:00 |
| R1 | 69.9% frozen (87 holds ≥1 s, max 5.7 s) | `freezedetect` |
| R2 | Fraction-of-narration beats | `beat(frac)` at 72 call sites |
| R3 | Burst-then-freeze rhythm | 1–2 s reveal, 4–6 s hold |
| A1 | No captions | No subtitle stream |
| A2 | No music bed | Digital silence at chapter joins |
| A3 | 8–10 pt labels; `DIM` ≈ 3.4:1 contrast | `ELEMENT-WISE`, `PAIR 0 · FAST`, … |
| P1 | Timing contract false | Scenes +0.05–0.44 s; total 301.17 vs 299.20 |
| P2 | Fonts not bundled | Avenir Next / DIN Alternate are macOS-only |
| P3 | Narration in three places | `VOICEOVER_SCRIPT.md`, `generate_audio.py`, `render_scenes.py` |
| P4 | No fact gate | Numbers hand-typed |
| P5 | Video not linked anywhere | No reference in README/docs |
| S1 | 78% memory story absent | End card: "and nothing else" |
| S2 | No CTA or cover | — |

---

## 10. Out of scope

No redesign of the Aurora language; no 4K/10-bit unless banding is confirmed; no re-recording of
the voice beyond the edited lines; no chapters beyond Part 2; no changes to the LLaMA-3-Lite
model or its docs.

---

## 11. Progress log

| Date | Phase | Note |
|---|---|---|
| 2026-10-25 | planning | Plan written against `a68c2c6`; v2 master frozen as baseline |
