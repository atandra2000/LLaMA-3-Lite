# Audit — cue breakdown and production state

Run before the R5 re-cue. Every claim in `CUE_BREAKDOWN.md` was checked against
`narration.yaml`, `audio/timings.json`, `render_scenes.py`, `facts.json`, the
tests and the docs. Nothing was changed except one wording fix in the breakdown.

## Verdict

The data is clean. The scene code is broken in exactly the way the re-cue
expects. Two hygiene items sit in `audio/`. One number needed a label fix.

## Clean

| check | result |
|---|---|
| sentence ids, narration vs timings | 167 both sides, unique, no missing, no extra |
| sentence text, narration vs timings | 0 mismatches |
| spans | monotonic, inside their section track |
| section durations | sum 736.246 s = `total_narration` |
| speech clips | 167 in `audio/sentences/` |
| section tracks | 12 new `.wav` in `audio/`, all named by `timings.json` |
| lint shape | max 18 words (s03.2), none over the 22 cap, no empty `visual` or `screen` |
| facts in speech | 128,256 vocab, ≈515M params, 1024 width, theta 500000, seq_len 2048, 1024→4096 FFN — all match `facts.json` |

## Findings

### F1 · the tree cannot render past Section01 — expected, unfixed

`BUDGETS` is built from `ORDER = list(TIMINGS["sections"])`, so its keys are the
twelve new section ids. `setup()` reads `BUDGETS[self.SEC]`. Only `Section01`
has a `SEC` value that still exists there (`sec01_function`). `Section02` through
`Section09` carry old ids (`sec02_tokens`, `sec03_residual`, `sec04_rmsnorm`,
`sec05_attention`, `sec06_rope`, `sec07_gqa`, `sec08_swiglu`, `sec09_output`)
and raise at setup.

Even `Section01` is not good: its five cues (`s01.1`–`s01.5`) now resolve to the
new sentence spans, which are not the spans the animation was cut against, and
the scene ignores `s01.6`–`s01.12`. It would render, paced wrong.

This is the mid-cue state. No fix inside the audit. R5 is the fix.

### F2 · 17 stale audio files sat beside the live ones — moved

`audio/` held 30 files: 12 new section tracks, `timings.json`, and 17 leftovers
from the 63-sentence era — `sec02_tokens`, `sec03_residual`, `sec04_rmsnorm`,
`sec05_attention`, `sec06_rope`, `sec07_gqa`, `sec08_swiglu`, `sec09_output`
(each `.wav` and `.mp3`) plus `sec01_function.mp3`.

`build_full_video.py` takes names only from `timings.json` and joins them under
`AUDIO_DIR`, so the pipeline never picked a stale file. The risk was a human or a
new script reading the folder and trusting the names.

Resolved 2026-10-26: all 17 moved to `audio/_old/`. `audio/` now holds the 12
tracks `timings.json` names plus `timings.json`. The move skipped any name
`timings.json` references, and the suite stayed at 74 passed, 1 skipped. Nothing
deleted; `audio/_old/` can go when the old cut goes.

### F3 · one label fixed

`CUE_BREAKDOWN.md` said "speech total 736.2 s". That figure is the narration
total including the 200 ms sentence gaps. Pure speech is 705.246 s; gaps add
31.0 s. The breakdown now states both.

For the record: `section-map.md` says "705.2 s of speech", which is correct.
`PLAN.md` calls 736.2 s "speech", which is loose by the same measure.

### F4 · prose counts that flip at R5

`README.md` line 23 "9 chapters" and line 66 "9 Manim scenes" describe the code
as it stands, which is 9 scenes. Both go wrong the moment scene 10 lands. Not
wrong today. Fix in the R5 pass, with the `ALL_SECTIONS` list.

### F5 · skill examples carry old section ids

`explainer-video-render-qa/SKILL.md` names `sec06_rope`, `sec07_gqa` and
`sec05_attention` in its examples. Those are real ids in the current code, so the
examples work now. They go stale with the renumber. The skill already warns
about this case.

### F6 · cleared, was a false alarm

`timings.json` stores `file` as a bare name (`sec01_function.wav`). That is fine:
`build_full_video.py` joins it under `audio/`. A check that looks the bare name
up from the project root will report every track missing. Nothing to fix.

## Scene inventory as it stands

Nine classes, 64 cues, all old ids. Old ids look like the new sentence ids and
are not the same lines.

| class | SEC (old) | CHAPTER | cues |
|---|---|---|---|
| Section01 | sec01_function | 01 · THE FUNCTION | s01.1–s01.5 |
| Section02 | sec02_tokens | 02 · EMBEDDINGS | s02.1–s02.7 |
| Section03 | sec03_residual | 03 · RESIDUAL STREAM | s03.1–s03.4 |
| Section04 | sec04_rmsnorm | 04 · RMSNORM | s04.1–s04.6 |
| Section05 | sec05_attention | 05 · SELF-ATTENTION | s05.1–s05.10 |
| Section06 | sec06_rope | 06 · ROTARY POSITIONS | s06.1–s06.9 |
| Section07 | sec07_gqa | 07 · GQA + QK-NORM | s07.1–s07.8 |
| Section08 | sec08_swiglu | 08 · SWIGLU | s08.1–s08.7 |
| Section09 | sec09_output | 09 · OUTPUT HEAD | s09.1–s09.8 |

Target is 12 classes, 167 cues, ids equal to the sentence ids in `narration.yaml`.

## Test gate

74 passed, 1 skipped. The skip is `test_every_sentence_has_exactly_one_cue`,
reason "scene re-cue pending (12 sections, 167 cues)". Honest. It un-skips when
the re-cue lands.

## What the audit does not cover

- Per-old-cue visual detail inside `render_scenes.py`. The breakdown maps at
  section level. Row-level reuse still needs a look at the code block behind each
  old cue during R5.
- A/B listening on the hard lines (R4 remnant).
- Whether the new voice track clips or hesitates. Ears, not scripts.
