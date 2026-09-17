# Archify quality-review delivery evidence

Refreshed 2026-09-17. Scope: only LLaMA-3-Lite's diagrams and guide. No model, training or data implementation changed. [Guide](index.html).

## Acceptance status

- **5/5 diagrams delivered, each 9/9 showcase checks, zero composition errors or warnings.**
- **Fresh automated browser evidence passed for all 5 exact delivered hashes.** Light-theme measurements cover 1440×900, 1600×1000, 1920×1080 and 2048×1320. Light/dark endpoint screenshots accompany each receipt. This does not establish intermediate dark-theme geometry.
- **Perceptual review: skipped (image input unavailable).** An image read was attempted, but the provider rejected image input. Automated receipts correctly retain `visualReview: pending`. Earlier visual approval does not certify these regenerated artifacts.
- **The proposed 12px essential / 11px secondary target is not met.** The measured minimum contextual text below is smaller. Containment and the Archify 6px gate are not premium readability certification.
- Browser interaction/mobile guide review could not run through the bridge (broken-pipe failure). Guide local links, navigation anchors, unique IDs and source symbols passed static checks. Focus/search/reset, representative export and narrow-screen guide behavior remain unverified for this revision.

## Source and configuration pin

Source revision: `c99a925e97339fbaa771488f4036817ad535b90e`. Configuration: `config.py:get_config`. These pins identify reviewed model code, not the later documentation commit. Configured dimensions, calculated sizes and target corpus/training budgets are not fresh GPU measurements. No corpus inventory, full training run or GPU benchmark was performed.

## Exact artifact bindings

| HTML | Type | Specification | Spec SHA-256 | HTML SHA-256 | Min node px at 1440×900 | Browser |
|---|---|---|---|---|---:|---|
| [data.html](data.html) | architecture | [data.architecture.json](data.architecture.json) | `c6e9b1dd68cfed154df8ca70eb50efd8f550d6f6365c18c22fb22afb958523e3` | `bf449b148f3dbc0f6a6cf25e326983de974167883c5ce08541a2e3a20e13c2c0` | 7.78 | pass |
| [model.html](model.html) | architecture | [model.architecture.json](model.architecture.json) | `2a74c294cf2dd0313efbf06693b83c4eeddfee0d49f3f6bc7bbbf5180f70e75e` | `c58c698cfb789be0a415bccd76290e765c90b6c4fc9d79cae6966978d9c6d944` | 8.00 | pass |
| [optimization.html](optimization.html) | architecture | [optimization.architecture.json](optimization.architecture.json) | `30d90d7ac6c74f5fab45608f717c6476209b3d3fef43f1b7bc518565c8fd8ba6` | `5a685bed45c2442decf1e6134508bd01ee6d3198cd83e1b65837c82c6b982770` | 8.39 | pass |
| [system.html](system.html) | architecture | [system.architecture.json](system.architecture.json) | `c33fa1c45303e41b853b77dd2f6f40326e69b75b3c167f5f8910e2cba8ed7481` | `d869f571a87193e9c10be734ba5ac4a50d0c8d27728b091398f1c65c7cd5b077` | 8.00 | pass |
| [training.html](training.html) | workflow | [training.architecture.json](training.architecture.json) | `d0f988c1b4f74cc775fd16e1e8cc470ff3406ceb54e8970ec00daad03c7713cb` | `18382df299475910e3664d8c1bfd765f05bbb5e82154ba350bb29e03dd6981b0` | 8.00 | pass |

Delivery JSON includes byte counts. Each HTML has a `.delivery.json`, `.visual-check.json`, contact sheet and four PNGs. Validation JSON records the last successful static check.

## Corrections and source mapping

- Overview: evaluation and artifact/telemetry paths are independent cadence branches, not mandatory serial stages.
- Model: four enclosed operations form the repeated 16× block. Final norm has distinct hidden-state/loss and inference-logits branches. Sources: `model.py:DecoderBlock.forward`, `model.py:Transformer.forward`.
- Data: partitioning precedes window slicing, int64 allocation and batching. The chunk-aligned held-out tail is not document-disjoint. Sources: `data/prepare_data.py:concat_shards_to_cache`, `data/shared_data/loader.py:build_training_data`.
- Training: workflow schema v2 now supplies accumulation yes/no branches, an actual continue-to-fetch return and conditional final exit. The existing `training.architecture.json` filename is retained for links, but its declared type and delivery command are **workflow**. No exhausted-iterator or warmup-gradient bug was repaired. Source: `train.py:train_model`.
- Optimization: six owner boxes belong to four explicit budgets (GPU working memory, loss temporaries, host memory, compute throughput). Incidental causal arrows removed.
- Browser correction rounds: 1 each for model and overview (excess authored height compacted), 0 for the other three. No clipping, hidden overflow, internal scrolling or typography reduction used. Final browser evidence passed. This is not a perceptual correction count.

## Documentation checks

The first `env -u PYTHONPATH python3 tests/test_doc_refs.py` attempt was blocked by missing `wandb`. Installed the real `wandb==0.26.1` package into an isolated scratch venv with system-site-packages, then ran the same script: **OK: 23 docs, all symbol citations resolve, no line anchors**. No stubs or reference deletions. A LibreSSL warning was emitted, but the gate exited zero. No `scripts/check_docs.py` exists. Static guide check: 38 local/navigation links and 18 distinct symbol citations passed. No repository dependency files changed.

## Regeneration

From the repository root, validate and deliver a changed candidate, then run visual-check only if delivery succeeds:

```bash
node ~/.agents/skills/archify/bin/archify.mjs validate architecture docs/diagrams/atlas/data.architecture.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/atlas/data.architecture.json docs/diagrams/atlas/data.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/atlas/data.html --json
node ~/.agents/skills/archify/bin/archify.mjs validate architecture docs/diagrams/atlas/model.architecture.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/atlas/model.architecture.json docs/diagrams/atlas/model.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/atlas/model.html --json
node ~/.agents/skills/archify/bin/archify.mjs validate architecture docs/diagrams/atlas/optimization.architecture.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/atlas/optimization.architecture.json docs/diagrams/atlas/optimization.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/atlas/optimization.html --json
node ~/.agents/skills/archify/bin/archify.mjs validate architecture docs/diagrams/atlas/system.architecture.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/atlas/system.architecture.json docs/diagrams/atlas/system.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/atlas/system.html --json
node ~/.agents/skills/archify/bin/archify.mjs validate workflow docs/diagrams/atlas/training.architecture.json --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs deliver workflow docs/diagrams/atlas/training.architecture.json docs/diagrams/atlas/training.html --quality showcase --json
node ~/.agents/skills/archify/bin/archify.mjs visual-check docs/diagrams/atlas/training.html --json
```

Redirect successful delivery output to the matching `.delivery.json`, refresh the receipt table and verify hashes. Never retain a prior visual-pass claim after changing a specification.
