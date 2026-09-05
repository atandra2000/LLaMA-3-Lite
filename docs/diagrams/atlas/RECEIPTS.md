# Archify delivery evidence

The [illustrated guide](index.html) links five standalone architecture diagrams. Training and data use flow semantics within the architecture renderer.

All five: **9/9 showcase checks, 0 composition errors, 0 warnings; automated browser evidence passed.**

Chrome checked 1440×900, 1600×1000, 1920×1080 and 2048×1320. All required viewport measurements passed horizontal/vertical containment, minimum projected text size and viewer-control clearance. Light/dark screenshots were captured at the smallest and largest sizes.

Image-based perceptual review: **passed** for all five, inspecting the 2048×1320 light and 1440×900 dark screenshots. Labels, arrows, nodes and cards are legible and uncropped; the maps occupy the desktop with clear vertical rhythm. This review does not establish search, focus, presentation or export interaction correctness; those controls were not exercised. Visual correction rounds after delivery: **0**.

The long-form index.html guide has static link/anchor validation; it does not carry the diagram browser receipt. It is a scrolling article, not a first-screen diagram.

Initial workflow/dataflow candidates were rejected by layout/readability checks and were not delivered. Their final unresolved diagnostics were projected context text of 5.98px against a 6px minimum (workflow) and leftmost node bounds (dataflow). The delivered architecture maps preserve pipeline explanations and pass the same showcase quality gate.

## Artifact bindings

### System

- Diagram type: `architecture`
- Output: [system.html](system.html)
- Specification SHA-256: `1dac4e5c5b735ad7bb5c7fbbb734b0eee067ac089ea82504bd9e634c6d61e7e0` (4,530 bytes)
- Artifact SHA-256: `70337a3692575c16caa5e0ff1fde2250153c36ea98e38ab06b725dfd5fb72fbf` (712,328 bytes)
- [Delivery receipt](system.delivery.json) · [Browser receipt](system.visual-check.json) · [Screenshot contact sheet](system.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Model

- Diagram type: `architecture`
- Output: [model.html](model.html)
- Specification SHA-256: `91f0e35bbf0231d250f590d7a40d93f61268d040684cbf04d443023bdee2bc25` (4,475 bytes)
- Artifact SHA-256: `de9b111849f14254e92c27b7fd079cea3c3c361a2d1b60c647b742985acd9bc5` (711,929 bytes)
- [Delivery receipt](model.delivery.json) · [Browser receipt](model.visual-check.json) · [Screenshot contact sheet](model.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Data

- Diagram type: `architecture`
- Output: [data.html](data.html)
- Specification SHA-256: `da1e041eed20cdd4e8ed533228fde05d0cc2814a133b7e9cdd5996efcf90fb46` (4,377 bytes)
- Artifact SHA-256: `ab8991d72c7802c59b83371e28d635cf0137ed1b1b2fc17f9eeaf25abdc02559` (712,349 bytes)
- [Delivery receipt](data.delivery.json) · [Browser receipt](data.visual-check.json) · [Screenshot contact sheet](data.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Training

- Diagram type: `architecture`
- Output: [training.html](training.html)
- Specification SHA-256: `99c4cea417ae1a97ba3f9c42383458246bb6d0d4601513ceb79b4404b7d9588a` (4,481 bytes)
- Artifact SHA-256: `a4eba4069d50515350888ca5d53b546196c6a84da6ab41a5e623817464d5c676` (711,660 bytes)
- [Delivery receipt](training.delivery.json) · [Browser receipt](training.visual-check.json) · [Screenshot contact sheet](training.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Optimization

- Diagram type: `architecture`
- Output: [optimization.html](optimization.html)
- Specification SHA-256: `dd33236a9408ed5aba3ade21171a3d5e6b1df77c8a51801833dd4cefa669b67d` (3,140 bytes)
- Artifact SHA-256: `1bdd8e2fb1d7e54209bc12fd8e4aa6066569b06006382d06805f1fe37bfaefa8` (705,705 bytes)
- [Delivery receipt](optimization.delivery.json) · [Browser receipt](optimization.visual-check.json) · [Screenshot contact sheet](optimization.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

## Verification limits

Parameter count at configured vocabulary: 513,840,128, obtained with a PyTorch meta-device model. CUDA was unavailable. No pretraining, real-data preparation, GPU speedup, memory benchmark or W&B integration was run. The 92→20 GB figure remains a documented estimate in this guide.

Specifications and HTML artifacts are frozen together by Archify deliver. To update a map, edit its JSON, run validate, then deliver and visual-check again; do not patch generated HTML.
