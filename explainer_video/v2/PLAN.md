# LLaMA-3-Lite explainer — v3 remediation plan

> **STATUS 2026-10-26 — DO NOT PUBLISH THE CURRENT MASTER.**
> The script was rewritten to a 12-section deep dive and then corrected after a
> production review. Everything measured in Phases 0–3 describes the old
> **63-sentence** cut. The new script is **167 sentences** (12 sections), and the
> review report is `REVIEW.md`.
> What this means:
> - Old Phase 2 checks covered the 63-sentence cut. They do not clear the new one.
> - Narration, timings, cues, scenes, captions, and the master all need regeneration.
> - Verification must expect 12 sections, not nine.
> - The "no re-recording" line in §10 conflicts with the new script. It is lifted.
> - Scene IDs drive tests and gates. Positional indices break after insertions.
> - Scene code is still structured around Sections 01–09. It needs a 12-scene re-cue.
> Recovery work is Phase R below. Phases 0–4 stay as history.

**Status:** Phase 0 complete · Phase 1a complete · Phase 1b complete · Phase 2 complete · Visual Redesign complete · Phase 3 in progress (T3.1 done) · Phase R in progress (R1, R2 done)
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

- [x] **T0.1 Fact builder** — import `config.get_config()`, load the Llama-3 tokenizer, compute
      params (513.8 M @128,000 / 514.4 M @128,256), demo token IDs, attention
      raw → scaled → masked softmax, RoPE pair count. → `facts.json` (`tools/build_facts.py`, 0.5 d)
- [x] **T0.2 Narration single source** — per-sentence `id` + `visual` notes; generate the
      Markdown script doc from it. → `narration.yaml`, `tools/build_script_doc.py` (0.5 d)
- [x] **T0.3 Script lint + sentence split** — max 22 words per sentence; rewrite the long
      section-5 sentences into 12–18 word units. → `tools/lint_script.py` (1 d)
- [x] **T0.4 Fonts** — bundle, register, retune the type scale (no label below 12 pt).
      → `assets/fonts/`, `render_scenes.py` header constants (0.5 d)

**Gate (met 2026-10-25):** `python tools/check_facts.py` exits 0 after a rebuild (37 checks);
deleting `facts.json` and rebuilding produces a byte-identical file
(`sha256 2a584995…`). Fonts render from `assets/fonts/` with `USE_SYSTEM_FONTS` unset.

### Phase 1 — Correctness and layout · 4 days

**1a. Facts into the scenes**

- [x] **T1.1 Wire facts** — spec chips, token chips, matrix dims, `128K` → `128,256`,
      `515M` → `≈515M`. → `Section01`, `Section02`, `Section09.construct`
- [x] **T1.2 Attention demo** — raw scores → `÷√d_k` strip → masked softmax → weights; the
      masked key gets `0.00`, not `0.05`. → `Section05.construct`
- [x] **T1.3 QK-norm caveat** — narration + caption: "a Lite addition — LLaMA 3 itself has none".
      → `narration.yaml` s07, `Section07.construct`
- [x] **T1.4 SwiGLU gating** — two rows (SiLU gate, up) → product row; drop the 0.3 "survival"
      threshold; narration says "scales each feature". → `Section08.construct`
- [x] **T1.5 KV cache** — halve **height** (8 vs 16 bars), per the script's own wording; label
      head counts. → `Section07.construct`
- [x] **T1.6 Wording** — "word" → "token" in the opening line. → `narration.yaml` s01

**1b. Layout defects**

- [x] **T1.7 §4 relayout** — geometry inset + LN/RMS/ε chips inside y ≥ −3.45; remove dead
      `Circumscribe(mean_chip)`; remove or connect the floating √.
- [x] **T1.8 §5 banner** — move `wcap` below the bars; fade the header before the banner, then
      place the banner at y ≈ 2.45; move `hcap` under the heads on two lines.
- [x] **T1.9 §9 loop** — stroke-only (`fill_opacity=0`), routed **above** the token row; keep all
      four probability bars and draw the chip from a copy.
- [x] **T1.10 Layout sweep** — §3 `× 16 blocks` off the band edge; §8 FFN card up to y = −3.1;
      §7 caption up; §1 header rule; §1 circle block 1 not 12; §9 recap brace over the two
      middle pills; watermark only on the end card; progress bar at y = −3.85 with chapter ticks.
- [x] **T1.11 §6 RoPE** — delete the unlabeled 3×3 grid or make it a labeled score matrix;
      animate m continuously 0→7; constant-frequency rotation with period labels.
- [x] **T1.12 Safe-rect guard** — `SAFE = (-6.7, 6.7, -3.6, 3.6)`; after each beat assert every
      text bbox is inside and log overlaps of top-level groups.

**Gate (met 2026-10-25):** 480p15 render of all 9 scenes with `--check_layout` —
`0 safe-rect violations, 0 partial overlaps` on every scene. One contact sheet per scene
(`media/sheets/Section01..09.png`) reviewed against the finding it closes.

### Phase 2 — Timeline and rhythm · 3 days

- [x] **T2.1 Per-sentence audio** — trim silence, assemble with 200 ms gaps, emit
      `audio/timings.json` with exact sentence start/end. (1 d)
- [x] **T2.2 Cue API** — `self.cue("s07.2")` waits until that sentence's offset + `LEAD`;
      replace all `beat(frac)` sites scene by scene; delete `beat()` at the end. (1.5 d)
- [x] **T2.3 Motion pass** — ambient pulse updater, cue-synced `flow` pulses, a moving
      highlight for the spoken clause; no hold > 3.5 s. (0.75 d)
- [x] **T2.4 Measure and iterate** — `freezedetect`; repeat until the freeze gate is met. (0.5 d)

**Timing rules**

- `budget = LEAD + section_audio + TAIL`, computed from `timings.json`.
- `sync_end()` uses the renderer clock (`self.time`), waits the remainder, then **raises** if
  `abs(self.time - budget) > 1/30`.
- `build_full_video.py` fails if `section_audio + LEAD > scene_duration - 0.1`.

**Gate (met 2026-10-25):** 9/9 scenes render with `sync_end()` happy — no budget
raise, no `cue` LATE warning. `tools/check_freeze.py` on the assembled cut:
`2 holds ≥1 s · 1.0% frozen · longest hold 1.73 s`. Budget error is bounded by one
frame (`sync_end` slack is `1 / config.frame_rate`), and `build_full_video.py`
refuses a scene whose narration does not fit.

**Gate:** total duration = sum of budgets ±0.1 s; no scene overruns; freeze gate met.

### Phase 3 — Sound, captions, packaging · 2.5 days

- [x] **T3.1 Music bed** — CC0 track, −28 LUFS, sidechain-ducked under voice, 1.5 s fades;
      two-pass `loudnorm` (linear) on the final mix. (0.75 d)
- [ ] **T3.2 Captions** — SRT + VTT from timings, ≤42 chars/line, ≤2 lines, ≤17 cps, clause
      splits; `--burn` option. (0.75 d)
- [ ] **T3.3 Packaging** — cover/thumbnail 1280×720; README + `docs/README.md` links with
      poster; end-card repo URL + Part 2 teaser; hold 3 s then fade to black. (0.5 d)
- [ ] **T3.4 Verification + CI** — `tools/verify_video.py` and the `Makefile` are
      written and exercised (T3.4a, done below). What remains is the CI job and
      the golden-frame diff. `build_timeline.py` and `make_contact_sheet.py`
      were never written. (0.5 d)

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
make check     # facts, script lint, doc sync, tests (fast, no render)
make layout    # every scene at 480p with the safe-rect guard
make facts     # rebuild facts.json + verify
make audio     # re-record narration, rebuild audio/timings.json
make scenes    # every scene at 1080p30
make video     # scenes + assemble the master
make verify    # tools/verify_video.py
make all       # check + audio + video + verify
```

The scene list is discovered from `render_scenes.py`, so a new section does not
mean editing the Makefile. Still unbuilt from this section's original list: a
`build_timeline.py`, `make sheet` / `make_contact_sheet.py`, and the CI job.

`verify_video.py` asserts, against the artefacts a reader receives: duration and
an audio track; one chapter per narration section, contiguous and titled from
`narration.yaml`; each section's speech plus lead fits inside its chapter, with
monotonic sentence spans; integrated loudness −16.0 ±0.5 LUFS and true peak at
or below −1.5 dBFS; frozen share ≤35% with no hold over 3.5 s; SRT and VTT both
present, one cue per sentence. Golden frames (3 per scene) live in
`media/golden/`; a pixel diff > 2% fails.

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

## 5b. Phase R — Recovery for the 167-sentence production · 5–7 days

The review in `REVIEW.md` set this order. Each step gates the next.

- [x] **R1 Script lock** — correct every P0 claim, add the residual-stream and
      pre-norm foundation, contrast training with generation, soften the strong
      claims. `narration.yaml` 159 → 167 sentences (12 sections), lint 0 warnings,
      `VOICEOVER_SCRIPT.md` regenerated, 70 tests green. (0.5 d)
- [x] **R2 Plan docs** — this banner, Phase R, and the same scope note in
      `README.md` and `REDESIGN_PLAN.md`. (0.25 d)
- [ ] **R3 Animatic** — low-resolution 12-scene pass first. Check comprehension,
      timing, and mobile text size before any 1080p render. (1.5 d)
- [ ] **R4 TTS re-record** — new timings for 167 sentences; A/B the hard lines.
      The no-re-recording constraint is lifted. (1 d)
- [ ] **R5 Scene re-cue** — 12 scenes instead of 9; new sections 02/04/08/11;
      cue by scene ID, not by position. (2 d)
- [ ] **R6 Render, package, gate** — 1080p master, captions, chapters, end card
      with the "Almost In F" credit, then `make verify`. (1 d)

**Gate:** `make verify` green on a master whose narration is the 167-sentence
script; freeze share ≤35%; longest hold ≤3.5 s; 12 chapters present.

---

## 10. Out of scope

No redesign of the Aurora language; no 4K/10-bit unless banding is confirmed; no chapters beyond
Part 2; no changes to the LLaMA-3-Lite model or its docs. Re-recording is in scope for Phase R:
the new script needs it.

---

## 11. Progress log

| Date | Phase | Note |
|---|---|---|
| 2026-10-25 | planning | Plan written against `a68c2c6`; v2 master frozen as baseline |
| 2026-10-25 | 0 | T0.1–T0.4 landed. `facts.json` (vocab 128,256 · 514,364,416 params · tokens 791/8415/7731/389/279/5634 · attention raw→scaled→masked softmax · RoPE 64 pairs) built and gated by 37 checks; `narration.yaml` is the only narration source (63 sentences, max 18 words) with `VOICEOVER_SCRIPT.md` and `generate_audio.py` generated/loaded from it; Inter + Barlow Semi Condensed bundled and registered, `MIN_LABEL_SIZE = 12` enforced by `pt()` and an AST test; 15 tests green |
| 2026-10-25 | 1a | T1.1–T1.6 landed. `render_scenes.py` loads `facts.json` once; spec chips (`≈515M`, `1,024`, `128,256`, `2,048`), token chips/IDs, embedding and unembedding dims, the Section 5 attention demo (raw → ÷√d_k → masked softmax → `0.12/0.86/0.02/0.00`), Section 7 head counts and cache (16 vs 8 bars, same width), Section 9's recap figures, and §6's θ/context/slowest-pair labels all come from facts. SwiGLU now shows SiLU(gate) and up rows resolving into their product (no 0.3 threshold); QK-norm carries the “a Lite addition — LLaMA 3 itself has none” caveat in s07.8 and on screen; s01.2 says token. `narration.yaml` edited (audio re-record is T2.1), `VOICEOVER_SCRIPT.md` regenerated, 31 tests green, all nine scenes render at 480p15, and frame checks on 1–9 are clean. |
| 2026-10-25 | 1b | T1.7–T1.12 landed. §4 geometry inset at `[-5.0,-2.5]` with the √ on the mean→RMS arrow and the dead `Circumscribe(mean_chip)` dropped; §5 header fades before the banner at y=2.45, `wcap` under the bars, two-line `hcap` under the heads; §9 loop stroke-only above the token row with all four bars kept; §3 `× 16 blocks` at `[5.5,1.25]`; §8 FFN card at y=−3.1; §7 QK strip at y=−2.15; §1 header rule (no `stretch(0.01`) and ring on `stack[0]`; §9 recap brace over the two middle pills; watermark end-card only; progress bar at y=−3.85 with one tick per chapter; §6 orphan 3×3 grid gone, m sweeps 0→7 continuously, period labels from `facts.rope.samples`. Safe-rect guard `SAFE=(-6.7,6.7,-3.6,3.6)` runs at every beat and raises in `sync_end()`. Gate met: 9/9 scenes at 480p15 with `--check_layout`, `0 safe-rect violations, 0 partial overlaps` each, 55 tests green, contact sheets `media/sheets/Section01..09.png` reviewed. Budget note for Phase 2: sec02 +0.11 s, sec05 +0.14 s, sec06 +0.14 s, sec07 +0.31 s over; sec01/sec04 under by 0.01 s (trips the `rem < 1/30` wording branch in `sync_end`). |
| 2026-10-25 | 2 | T2.1–T2.4 landed. `generate_audio.py` now synthesises per sentence (63 clips in `audio/sentences/`), trims silence, joins with 200 ms gaps and writes `audio/timings.json` — the only timing source; `render_scenes.py` reads it and `durations.json` is gone. `beat(fraction)` is deleted; 64 `self.cue("sNN.n")` calls wait on `LEAD + sentence start` and self-correct against the absolute offset. Run times snap to whole frames (`_snap`) so the tracked clock equals the rendered one. `sync_end()` fills the remainder then **raises** outside one frame of budget. T2.3: the ambient field drifts and breathes on a shared clock, the clause playhead slides across the progress track and flares at every cue. `build_full_video.py` reads `timings.json`, places `.wav` tracks and refuses a scene whose narration does not fit. Gate met: 9/9 scenes render with no budget raise and no late cue; `tools/check_freeze.py` reports `2 holds ≥1 s · 1.0% frozen · longest hold 1.73 s` (from 87 holds / 69.9% / 5.7 s); 70 tests green; layout gate still `0 safe-rect violations, 0 partial overlaps`. Two content fixes fell out: s05.5/s05.6 swapped so the mask precedes the softmax it feeds, and §1/§2/§8 tails trimmed to their clause windows. Drive-by on the §5 attention beat (pre-existing, from Phase 1b): `v = x W_V` sat on top of `wcap` and `zero_note`. `glow_dot` returns a VGroup whose halo is 0.97 wide, so `next_to(vdot, UP)` measured from the halo and parked the label 0.6 units too high. The three projection labels now anchor on the dot core (`dot[-1]`), matching what the arrows already did. Why the gate missed it: `group_overlaps` needs `w*h ≥ 0.06` and the real overlap was 0.045, and `safe_violations` never compares two glyphs. |
| 2026-10-26 | redesign | Aesthetic overhaul landed. Swapped "Aurora" dashboard/card look for a 3Blue1Brown-inspired mathematical canvas: flat matte blackboard (#0B0E14), ambient blooms/orbs disabled, bottom scrubber rail removed, clunky card/pill wrappers excised. Matrix brackets added to self-attention demo, projection collisions resolved, coordinate crosshair axes added to RoPE dials, slide headers fade out so geometry fills the 1080p canvas. Full 9-scene 1080p30 render complete with 0 safe-rect violations, 0 partial overlaps, 70/70 tests passing, master video assembled at 296.60s with 0.000s A/V drift. |
| 2026-10-26 | 3 | T3.1 landed. `assets/music/bed.mp3` = Komiku "The space is conforting" (CC0, archive.org, source logged in `assets/music/CREDITS.md`); FreePD is offline, so the track came from the Komiku CC0 item on archive.org. `build_full_video.py` loops the bed over the runtime, levels it to −28 LUFS (−8.6 dB from its raw −19.4), ducks it under the voice with `sidechaincompress` (threshold 0.015, ratio 12, 20 ms / 300 ms), 1.5 s fades, then two-pass linear `loudnorm` on the final mix. Final master: 296.60 s, 0.000 s A/V drift, −16.3 LUFS integrated, TP −1.5 dBFS. 72 tests green (`tests/test_phase3.py` pins the loudnorm JSON parse — real ffmpeg prints lines after the JSON block, so it uses `raw_decode`). |
| 2026-10-26 | 3 | Bed swapped on taste rejection (Komiku too busy, had a beat). New bed: Kevin MacLeod **"Almost In F"** — slow contemplative piano, 3b1b-style, per user reference. **Deviation from the CC0-only rule:** this track is **CC BY 4.0**, so the artist credit line in `assets/music/CREDITS.md` must appear on the end card (T3.3). Rebuilt master: 296.60 s, 0.000 s A/V drift, **−16.0 LUFS** integrated, **TP −1.4 dBFS** (0.1 over the −1.5 target; within one dither step of loudnorm's TP ceiling). Duck and fades unchanged. |
| 2026-10-26 | script | Narration rewritten for depth (user ask: "much more explanatory and in depth", runtime limit lifted). `narration.yaml`: 63 → **94 sentences** (sec01 7, sec02 10, sec03 8, sec04 10, sec05 18, sec06 10, sec07 12, sec08 9, sec09 10), mean 12.1 words, max 18, lint clean with 0 warnings. Every stage now carries its reason (why scale control, why ÷√d_k, why the causal mask, why GQA shrinks memory, why SwiGLU gates). Fact fix folded in: vocabulary is spoken as **128,256**, not 128,000 (s02.2, s09.2). Humanizer pass applied: no em dashes, no staged run-ups ("Let's open it up" gone), no "at its core" / "the heart of" / "the elegance". End card no longer says "and nothing else". `VOICEOVER_SCRIPT.md` regenerated. **Cascade still open:** new timings need a TTS re-record, `render_scenes.py` needs re-cue (64 cue sites) plus any new visuals the deeper lines call for, then 9-scene re-render and reassembly. User authorized adding animations/elements to support narration. |
| 2026-10-26 | script v2 | Script expanded to a 3b1b-style deep dive (user: "don't hesitate to add more sections", "powerful and in detail", "just like 3blue1brown"). Structure: 9 → **12 sections**. New sections: **02 the objective** (why next-token prediction is the only training signal, surprise/loss, no labels), **04 embeddings** (split out of tokens: learned meaning space, the honest 2-D sketch note), **08 QK-norm** (its own beat: per-head RMSNorm on q/k, 128-number gamma), **11 the generation loop** (split out of output). Attention split into **05 matching** (q/k/v, scores, ÷√d_k) and **06 blending** (mask, softmax, heads). `narration.yaml`: 94 → **159 sentences**, mean 12.3 words, max 18, lint 0 warnings. `tools/lint_script.py`: the 18-word cap now keys on attention section **ids** (`sec05_matching`, `sec06_blend`) instead of the old positional index 5. Real figures woven in from `facts.json`: token IDs 8415/5634, the `sat` row weight 0.86 on `cat` with 0.00 on the masked cell, KV-cache math 16×2×4×128 = 16,384 per token and 33M at full context, FFN ~12M vs attention ~3M params per block, embedding and unembedding ~131M each (just over half the model). Projected runtime **~13–15 min** at the current voice rate. **Video rework is now a restructure, not a re-cue:** 12 scenes instead of 9, new section 02/04/08/11 scenes, user has authorized new animations. |
| 2026-10-26 | R1 | Production review landed as `REVIEW.md` (verdict: do not publish the current master). R1 script lock applied to `narration.yaml`: **159 → 167 sentences**, lint 0 warnings, `VOICEOVER_SCRIPT.md` regenerated, 70 tests green (same 2 honest skips). P0 claim fixes: s02.2 now says cross-entropy is the objective with z-loss at one ten-thousandth (was "the only one"); s04.19/s11.2 say a separate, untied output matrix (no weight tying implied); s04.8 marks the cat/kitten sketch as illustrative; s06.14 says query projections are per head but keys and values are shared by pairs; s07.1 says shuffling tokens shuffles the outputs to match (permutation-equivariant, not "scores unchanged"); s07.11 gives the fastest pair a full turn in about six point three tokens; s07.12 says the rotations combine into a relative turn; s07.13/s07.14 scope the distance claim to the positional part and keep content in the score; s08.3 says RMSNorm controls scale (no strict bound); s08.5 gives one shared 128-number gain per layer for queries and one for keys; s09.5/s11.11/s11.12 present the KV cache as an inference-time optimization and say this code recomputes the full sequence; s09.10 calls the cache a growing inference-time memory cost; s09.14 states the four-vs-eight baseline; s09.15 keeps the attention operation but allows quality change; s10.11 gives the real SiLU shape (half the input near zero, nearly unchanged for large positive); s10.13 lets the gate shrink, pass, or amplify. Foundation added: **sec04 s04.13–s04.18** carry the residual stream and the pre-norm block picture (RMSNorm → attention/SwiGLU → add back); **sec02 s02.12–s02.13** contrast parallel training with one-token-at-a-time generation. Softens applied in s01.4, s01.6, s02.7, s04.6, s04.7, s04.9, s05.10, s05.13, s05.14, s08.6. Renumbered inside sec02 and sec04 only; scene code still references the old IDs and needs the R5 re-cue. |
| 2026-10-26 | R2 | Plan docs caught up. The `STATUS 2026-10-26 — DO NOT PUBLISH` banner, the Phase R block, and the matching scope note in `README.md` and `REDESIGN_PLAN.md` were already in the working tree; R2's box was simply never ticked. Stale counts corrected: `REVIEW.md` said 159 sentences in three places (now 167), and the `test_phase2.py` skip reason said 159 cues (now 167). |
| 2026-10-26 | R4a | TTS re-record landed. `generate_audio.py` → 167 clips, 12 section tracks, `audio/timings.json` rebuilt from `narration.yaml` alone. Measured **736.2 s = 12.3 min** of speech (edge-tts `en-US-AndrewMultilingualNeural` at +8%), so the new master lands near **12.5 min** with lead and tail — against 4:56 for the old cut. Per-section: sec01 53.7s/12, sec02 63.3s/15, sec03 69.9s/14, sec04 83.4s/19, sec05 56.9s/15, sec06 60.6s/16, sec07 76.3s/16, sec08 30.8s/7, sec09 66.1s/15, sec10 69.8s/16, sec11 61.9s/13, sec12 43.5s/9. `test_timings_cover_every_section` un-skipped and green; suite is **74 passed, 1 skipped**. Still open on R4: the A/B listening pass on the hard lines. `synth()` has no cache, so every re-record is a full 20 min network job — worth a text-hash skip before R5 iterations. |
| 2026-10-26 | T3.4a | Release gate written and exercised. `tools/verify_video.py` checks what a reader receives: duration and an audio track; one chapter per narration section, contiguous and titled from `narration.yaml`; speech + lead fits inside its chapter with monotonic spans; integrated −16.0 ±0.5 LUFS and true peak ≤ −1.5 dBFS; frozen share ≤35% and no hold over 3.5 s; SRT and VTT present with one cue per sentence. It reuses `check_freeze.measure` and `parse_ebur128` (pure, pinned in `tests/test_phase3.py`). Pointed at the current master it fails **15 of 20** checks and independently reproduces REVIEW.md's freeze numbers exactly (75.8% frozen, 10.67 s longest hold) plus the −1.4 dBFS true-peak overage. `Makefile` added: `check` / `layout` / `facts` / `audio` / `scenes` / `video` / `verify` / `all`. The scene list is discovered from `render_scenes.py` rather than hardcoded, so the 9 → 12 change does not mean editing the Makefile. `build_full_video.py` now derives its scene count from `timings.json` and its chapter titles from `narration.yaml`, and targets `TP=-1.6` so the master lands under the −1.5 dBFS ceiling. **Left unbuilt from §6's original list:** `build_timeline.py`, `make sheet` / `make_contact_sheet.py`, golden-frame diffing, and the CI job. |
| 2026-10-27 | R5 | **Scene re-cue to the locked 167-sentence script landed.** All 12 sections now cue `narration.yaml` ids: `sec01_function`, `sec02_objective` (already clean), then `sec03_tokens` (new tokenizer art), `sec04_embedding` (old Section02 embedding matrix + Section03 residual rail + Section04 pre-norm fold-in), `sec05_matching` and `sec06_blend` (old Section05 attention split), `sec07_rope` (old Section06, renumbered), `sec08_qknorm` and `sec09_gqa` (old Section07 split), `sec10_swiglu` (old Section08), `sec11_loop` and `sec12_recap` (old Section09 split). **167 `self.cue()` calls, one per spoken sentence, no missing / stray / duplicate** (`test_every_sentence_has_exactly_one_cue` un-skipped). `tools/check_beats.py` added: reads the same arithmetic the scene runs (run_time sums against sentence gaps) and flags a LATE cue or a >2.5 s hold before a render is paid for; `tools/check_freeze.py` remains the freeze authority. Per-section gate on all 12: `--check_layout -qm -r 1920,1080` → **0 safe-rect violations, 0 partial overlaps**, no cue LATE, `sync_end()` inside one frame of budget; `check_freeze.py` → **frozen share 0.0%, longest hold 0.00 s** on every scene (gate ≤35% / ≤3.5 s). Long holds closed on the nice-to-have list: s01.4 3.56 s and s02.6 2.83 s now carry a closing beat (s01.11's reported 3.83 s was a `check_beats` loop-count blind spot, not a real hold — the tool now expands `for` bodies). Suite **75 passed, 1 skipped**. The one skip is `test_section4_geometry_caption_stays_in_frame`: the RMSNorm unit-scale sketch left the locked script with the deep-dive cut, and the global safe-rect guard now covers caption placement. Pinned art re-homed and the pins retargeted to the new coordinates — `x16` at `[3.90,2.35]` clear of the residual rail, the attention banner at `[0,2.45]`, `wcap`/`hcap` in `sec06_blend`, `fx_card` at `[3.45,-3.1]` in `sec10_swiglu`, the generation loop stroke-only above the token row in `sec11_loop`, the recap brace and end-card watermark in `sec12_recap`. Two layout defects fell out and were fixed by measurement, not by eye: `LaggedStart` leaves a wrapper `Group` in the scene whose bbox spans every child and reads as a collision (replaced with multi-animation `play`), and a `Typst` formula needs multi-letter names quoted (`"logit"_i`). README's chapter list and scene count moved 9 → 12. **Master gate: `make verify` PASSED 13 of 13** on `llama3_lite_architecture_explainer.mp4` — 745.10 s = 12.42 min, 12 chapters titled from `narration.yaml`, 167 caption cues in both SRT and VTT, integrated **−16.3 LUFS** (window −16.5…−15.5), true peak **−2.3 dBFS** (ceiling −1.5), frozen share **0.0%**, longest hold **0.00 s**. Three master-side fixes were needed to get there: the last chapter has to cover the AAC-padded tail (chapters ended 0.067 s before the file), `build_full_video.py` now writes the SRT and VTT it was always supposed to, and `linear=true` loudnorm cannot enforce a TP target at all — an `alimiter` after it does. The limiter removes loudness with the peaks, so the loudnorm `I` target sits at −15.3 and the mix lands at −16.3 after limiting. |
