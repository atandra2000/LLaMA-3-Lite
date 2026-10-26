# Explainer video production review

Date: 2026-10-26  
Branch: `feat/explainer-v3-redesign`  
Reviewed commit: `0a2d71f`  
Scope: Review and recommendations only. No implementation.

## Verdict

**Do not publish the current master as the new explainer.**

The mathematical style is strong. The new script has good depth. Several technical claims still need correction. The plan also treats completed work on the old cut as if it covers the new script.

## Current state

| Item | Verified state |
|---|---|
| New narration | 12 sections, 167 sentences |
| Estimated new runtime | About 13-15 minutes |
| Current MP4 | 9 scenes, 63 sentences, 296.60 seconds |
| Voice and timings | Still belong to the old script |
| Scene code | Still structured around Sections 01-09 |
| Tests at current HEAD | 70 passed, 2 skipped |
| Skipped work | TTS re-record and 167-cue scene re-cue |
| Repository changes from the review | None |

The two skips are honest. They also show that the new production is incomplete.

## P0: Fix the script before recording

These are release blockers.

| Sentences | Required correction |
|---|---|
| `s02.2` | Cross-entropy is the main objective. Z-loss is also enabled at weight `1e-4`. |
| `s04.13`, `s11.2` | The output head is a separate, untied matrix. Do not imply weight tying. |
| `s04.8` | "Cat near kitten" is an imagined example. No trained checkpoint proves this geometry. |
| `s06.14` | Eight query heads share four key/value heads in pairs. Each head does not own all three projections. |
| `s07.1` | Attention without position is permutation-equivariant. Shuffling tokens shuffles the matching outputs. |
| `s07.11` | The fastest shown rotary pair has a period of about 6.3 tokens. It does not complete a turn in one step. |
| `s07.12` | Query and key rotations combine into a relative rotation. They do not simply cancel. |
| `s07.13`, `s07.14` | RoPE makes the positional contribution relative. Content still affects the attention score. |
| `s08.3` | RMSNorm controls scale. A learned gain prevents a claim of a strict magnitude bound. |
| `s08.5` | Each layer has one shared 128-value query gain and one shared key gain. They are not separate per head. |
| `s09`, `s11` | Present KV caching as a possible inference optimization. The current generation code recomputes the full sequence. |
| `s09.10` | Call the cache a growing inference-memory cost. Do not call it the main memory cost without a baseline. |
| `s09.14` | State the baseline: four KV heads halve the cache compared with eight KV heads. |
| `s09.15` | GQA preserves the attention operation. It can still change model capacity or quality. |
| `s10.11` | Near zero, SiLU passes about half the input. Large positive inputs pass nearly unchanged. |
| `s10.13` | The SwiGLU gate can shrink, pass, or amplify a value. |

Also soften these claims:

- Use "LLaMA-3-style" in `s01.4`. QK-norm is a Lite addition.
- Avoid "every layer shares those dimensions" in `s01.6`.
- Say next-token training needs no extra manual labels in `s02.7`.
- Use "can learn semantic geometry" in `s04.6-7`.
- Say the table embeds tokens in `s04.9`.
- Include vector magnitude in the dot-product explanation in `s05.10`.
- Describe scaling as controlling score growth and softmax saturation in `s05.13-14`.
- Frame QK-norm stability as its purpose in `s08.6`, unless measurements are supplied.

## P0: Restore the missing foundation

The new script needs a clear residual-stream explanation.

Use one simple picture:

```text
residual stream
      |
      |- RMSNorm -> attention --|
      |                         |- add back
      |- RMSNorm -> SwiGLU -----|
      |                         |- add back
      v
next block
```

Explain these ideas before the detailed attention chapters:

- Each token carries a vector through the network.
- Attention and SwiGLU read normalized copies.
- Their results are added back to the original vector.
- This is the pre-norm structure used by `DecoderBlock`.
- Final RMSNorm then makes sense at the output head.

Also contrast training with generation:

- Training predicts at every position in parallel.
- Generation reveals one new token at a time.

## P1: Improve the story

The strongest creative change is to use one worked example throughout.

Follow "The cat sat on the ___" from tokens to the final prediction. Keep the same colors and objects. Let each chapter transform them.

Recommended story changes:

1. Open with the final prediction.
2. Ask how 515 million parameters produce that one choice.
3. Rewind to tokens.
4. Build the network one operation at a time.
5. Return to the opening prediction for the recap.

Add a useful parameter-budget reveal. The input embedding and output projection contain about **51.1% of all parameters**. That is a memorable fact about this model.

Keep formulas attached to visible objects. A score should light up the two vectors that produced it. A rotation should move the actual query and key. A cache calculation should grow with the token sequence.

Consider moving sampling controls into a short bonus chapter:

- Temperature changes randomness.
- Top-k limits the candidate count.
- Top-p limits the candidate probability mass.

## P1: Improve visuals and pacing

The blackboard style works. The main weakness is canvas use.

- Several diagrams occupy a small part of the screen.
- The final architecture labels are too small for comfortable mobile viewing.
- The sampled attention frames near 135-145 seconds show crowding or transition ghosting.
- Recheck those beats as settled 1080p frames before calling them defects.
- The sparse frame near 280 seconds appears during a transition into the architecture recap. Do not log it as a confirmed blank-scene bug.

The current master also fails the freeze gate:

- Frozen share: **75.8%**, target at most 35%.
- Longest hold: **10.67 seconds**, target at most 3.5 seconds.
- Holds lasting at least one second: **65**.

Ambient motion can pass a freeze counter without teaching anything. Each long sentence needs a meaningful visual change.

Build a low-resolution animatic first. Check comprehension, timing, and mobile text size before rendering all 12 scenes at 1080p.

## P1: Repair the plan

Keep old phase results as history. Add a current-status banner and a recovery phase for the 167-sentence production.

Update the live scope in:

- `explainer_video/v2/PLAN.md`
- `explainer_video/v2/README.md`
- `explainer_video/v2/REDESIGN_PLAN.md`

The updated plan must show:

- The old Phase 2 checks covered the 63-sentence cut.
- Narration, timings, cues, scenes, captions, and the master now need regeneration.
- Verification must expect 12 sections, not nine.
- The no-re-recording constraint conflicts with the new script.
- Scene IDs should drive tests and gates. Positional indices will break after insertions.

## P2: Finish audio, packaging, and checks

Audio measurements on the current master:

- Integrated loudness: **-16.0 LUFS**. This meets the target.
- True peak: **-1.4 dBFS**. This is 0.1 dB above the -1.5 dBFS target.

Remaining work:

- Re-record the corrected narration.
- A/B difficult sentences for voice pacing and emphasis.
- Add captions and 12 chapters.
- Add the end card and music attribution.
- Credit "Almost In F" by Kevin MacLeod under CC BY 4.0.
- Create the title, thumbnail, description, and repository link.
- Add the planned `Makefile` and verification tools.
- Automate timing, cue, layout, freeze, loudness, and packaging checks.
- Test a fresh-clone Linux build.
- Add golden-frame checks for important settled beats.

## P3: Optional expansions

These can come after the main film:

- A second video on the measured memory-optimization stack.
- Shorts for RoPE, GQA, and SwiGLU.
- Downloadable architecture diagrams.
- A small companion notebook.
- Additional caption languages.
- An appendix for KV caching and sampling.

## Recommended order

1. Correct and lock the script.
2. Restore residual-stream and RMSNorm teaching.
3. Produce the full animatic.
4. Record narration only after the animatic works.
5. Render, package, and run the release gates.

## Review limits

The review did not include a new full 1080p render, a full listening pass, or fresh TTS. The visual findings came from sampled frames and the existing master.
