# LLaMA-3-Lite Explainer v3: Mathematical Storytelling & Visual Redesign Plan
**Engineering and Art Direction Specification**

## 1. Executive Summary & Aesthetic Thesis
This document defines the complete visual redesign of the LLaMA-3-Lite explainer video (`v2/`).

The prior "Aurora" design suffered from pervasive generative UI defaults: dark dashboard styling, ambient floating orbs, multi-step radial gradients, translucent frosted-glass cards, all-caps tracked pill tags, and a persistent bottom progress rail. These decorative elements compressed the actual mathematics into cramped, low-contrast corners.

This redesign completely replaces the dashboard paradigm with **high-craft mathematical cinema** inspired by 3Blue1Brown (Grant Sanderson / Manim), unified with rigorous data visualization and interface craft standards (drawing from `frontend-design`, `dataviz`, and `impeccable`).

All audio assets, narration sentences (64 beats across 9 sections), factual counts (`facts.json`), and total video run time (296.45 seconds) remain strictly preserved.

---

## 2. Design System & Token Architecture

### 2.1 The Mathematical Stage (`STAGE`)
- **Stage Surface**: Uniform matte blackboard `#0B0E14` (Lab: $L=6.2, a=0.3, b=-2.8$). Zero drifting bokeh, zero multi-layer gradient orbs, zero glow shaders.
- **Reference Grid**: Faint Cartesian grid lines (`#1E293B`, 1.0px stroke, 0.4 opacity) that appear only when geometric space or vector transformation is being demonstrated.
- **Safe Zone**: Adheres strictly to `SAFE = (-6.7, 6.7, -3.6, 3.6)` in Manim units (1920×1080 16:9 safe frame).

### 2.2 Color Tokens & Contrast Accessibility (WCAG AAA & OKLab Validated)
Every semantic color is computed for high contrast against `#0B0E14` (all $> 7:1$, exceeding WCAG AA 4.5:1) and maintains pairwise OKLab $\Delta E \ge 15$ for normal and color-vision-deficient (CVD) viewers.

| Token | Hex Value | Contrast | Semantic Role | Secondary Encoding (Never Color Alone) |
|---|---|---|---|---|
| `STAGE_BG` | `#0B0E14` | Baseline | Stage canvas | Matte flat background |
| `INK_PRIMARY` | `#F8FAFC` | 18.5:1 | Hero equations, dominant symbols | 36–42pt bold Typst math |
| `INK_MUTED` | `#94A3B8` | 7.5:1 | Labels, matrix brackets, axes | 18–20pt regular text, direct labels |
| `INK_DIM` | `#475569` | 2.6:1 | Inactive grid lines, guides | 1.0px dashed lines |
| `COLOR_QUERY` | `#60A5FA` | 7.6:1 | Query vectors ($Q$), attention search | Vector arrow with bold label $\mathbf{q}$ |
| `COLOR_KEY` | `#FBBF24` | 11.6:1 | Key vectors ($K$), memory keys | Vector arrow with bold label $\mathbf{k}$ |
| `COLOR_VALUE` | `#34D399` | 10.1:1 | Value vectors ($V$), memory values | Vector arrow with bold label $\mathbf{v}$ |
| `COLOR_ROPE` | `#A78BFA` | 7.1:1 | Rotary dials, frequencies ($\theta, m$) | Polar dial with angular arc $\Delta\theta$ |
| `COLOR_OUTPUT` | `#FB7185` | 7.2:1 | Unembedding, logits, sampling | Probability simplex column, loop arrow |
| `COLOR_GATE` | `#38BDF8` | 9.0:1 | SwiGLU gating path | Curve plot of $\text{SiLU}(x)$, gate icon $\odot$ |

*Non-Negotiable Dataviz Rule*: Text labels always use `INK_PRIMARY` or `INK_MUTED`. Colored marks (arrows, dots, arcs, chips) sit beside them to establish identity.

### 2.3 Typographic Scale & Hierarchy
Typography uses the registered bundled `Inter` for clean prose and `MathTypst` for formulas.
- **Hero Identity**: 38pt – 44pt (`MathTypst`), centered mathematical thesis.
- **Sub-Equations**: 24pt – 30pt (`MathTypst`), active derivation terms.
- **Direct Object Labels**: 18pt – 20pt (`Inter`), sentence case, placed adjacent to vectors and matrices.
- **Axis & Dimension Subscripts**: 14pt – 16pt (`Inter`), dimensional anchors ($d=1024, d_k=128$).
- *Absolute Bans*: No all-caps tracked eyebrows. No floating pill wrappers. No em-dash container suffixes (`"TITLE — description"`).

### 2.4 Mathematical Motion Primitives
- `TransformMatchingTex`: Formulas morph element-by-element rather than fading in/out.
- `VectorProjection`: Live perpendicular dashed lines projecting query onto key vectors with real-time cosine angle sweeps.
- `RotaryClock`: Concentric polar dials displaying frequency gear ratios with counter-rotating angles.
- `MatrixHeatmap`: Monochromatic sequential brightness ramp (`#0B0E14` $\to$ `#FBBF24`), strictly avoiding deceptive multi-hue rainbow ramps.

---

## 3. Comprehensive 64-Beat Storyboard & Choreography

### Section 01: The Model as a Function (`sec01_function` · 5 beats · 24.3s)
- **s01.1** ("Every large language model is, at its core, a single mathematical function.")
  - *Geometry*: Clean center stage. A large mathematical definition draws on screen:
    $$f : \mathcal{V}^* \longrightarrow \Delta^{|\mathcal{V}|}$$
  - *Choreography*: Top brace marks domain $\mathcal{V}^*$ ("Arbitrary sequence of input tokens"); bottom brace marks codomain $\Delta^{|\mathcal{V}|}$ ("Probability distribution over vocabulary").
- **s01.2** ("It takes a sequence of tokens — and returns a probability for every possible next token.")
  - *Choreography*: Left side materializes discrete token tiles $\mathbf{t} = [t_1, t_2, \dots, t_N]$ in `INK_PRIMARY`. An arrow traverses through operator $f$, expanding on the right into a calibrated vertical probability bar chart where bar heights sum exactly to $1.0$.
- **s01.3** ("LLaMA-3-Lite is a compact, from-scratch implementation of Meta's LLaMA 3 architecture.")
  - *Choreography*: Function $f$ expands outward into a sleek, wireframe blueprint of the decoder block stack.
- **s01.4** ("Sixteen decoder blocks, a model dimension of one thousand twenty-four, and about five hundred fifteen million parameters.")
  - *Choreography*: 16 discrete block segments stack vertically in clean architectural symmetry. Dimensional annotations annotate the stack: $L=16$, $d_{\text{model}}=1024$, $N_{\text{params}} \approx 515\text{M}$.
- **s01.5** ("Let's open it up and follow one token all the way through.")
  - *Choreography*: The camera smoothly dollies into Block 1 as token vector $\mathbf{x}_0$ illuminates and crosses the input threshold.

### Section 02: Tokens Become Vectors (`sec02_tokens` · 7 beats · 33.1s)
- **s02.1** ("But before any mathematics can happen, text must become numbers.")
  - *Geometry*: Word string `"The architecture"` centered in clean white typography.
  - *Choreography*: A vertical scissor line cleanly splits the text into subword segments.
- **s02.2** ("A tokenizer splits text into subword pieces from a vocabulary of one hundred twenty-eight thousand tokens.")
  - *Choreography*: Segments snap into token indices: $[791, 8415, 7731]$. A vertical index strip displays the vocabulary boundary: $\{0, 1, \dots, 128255\}$.
- **s02.3** ("Each token ID indexes a row of the embedding matrix.")
  - *Geometry*: A majestic 2D matrix grid $W_{\text{embed}} \in \mathbb{R}^{128256 \times 1024}$ materializes with bracket delimiters.
  - *Choreography*: An amber selector bracket highlights row $i = 791$.
- **s02.4** ("That matrix holds one hundred twenty-eight thousand vectors, each one thousand twenty-four numbers long.")
  - *Choreography*: The selected row glides forward towards the viewer, unfurling its 1024 numerical values $[e_1, e_2, \dots, e_{1024}]$.
- **s02.5** ("That lookup places every token as a point in a high-dimensional space, where similar meanings sit close together.")
  - *Geometry*: Matrix dissolves into a 3D geometric vector space with subtle grid lines.
  - *Choreography*: A bright vector arrow shoots from the origin to point $\mathbf{x}_{\text{token}}$. Nearby semantic neighbor vectors ("model", "network") cluster with subtle dotted distance metrics.
- **s02.6** ("From here on, the model never sees characters again.")
  - *Choreography*: The text glyphs dissolve into thin air.
- **s02.7** ("Only these vectors.")
  - *Choreography*: The vector arrow pulses with directional clarity; coordinate numbers softly glide along its shaft.

### Section 03: The Residual Stream (`sec03_residual` · 4 beats · 19.8s)
- **s03.1** ("These vectors flow through the network along a single channel: the residual stream.")
  - *Geometry*: A continuous horizontal mathematical pipeline (the residual highway) spans the safe width of the screen.
  - *Choreography*: The vector $\mathbf{x} \in \mathbb{R}^{1024}$ enters from the left edge and travels steadily rightward.
- **s03.2** ("Each of the sixteen blocks reads from it, computes a transformation, and adds the result back.")
  - *Choreography*: A smooth bifurcated detour branch splits off the highway, enters an operator node $\mathcal{F}(\mathbf{x})$, and rejoins the highway at a mathematical direct-sum node $\oplus$.
- **s03.3** ("The stream is never overwritten, only refined, layer by layer.")
  - *Geometry*: The equation $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$ writes elegantly above the junction.
  - *Choreography*: The vector arrow $\mathbf{x}_l$ does not reset; its tip receives the vector addition $\Delta \mathbf{x}_l$ tip-to-tail, shifting angle and magnitude with continuity.
- **s03.4** ("Each token's vector accumulates context, turning a bare lookup into a rich, position-aware representation.")
  - *Choreography*: The stream accelerates across 16 sequential addition nodes; with each pulse, the vector’s internal representation gains geometric richness and color depth.

### Section 04: RMSNorm: Regulating Scale (`sec04_rmsnorm` · 6 beats · 29.2s)
- **s04.1** ("Inside each block, the first operation is normalization.")
  - *Geometry*: Zoom into the gateway of Block $l$. Vector $\mathbf{x}$ arrives at the RMSNorm stage.
- **s04.2** ("As activations flow deeper, their scale can grow, and attention's dot products are sensitive to exactly that.")
  - *Choreography*: Vector $\mathbf{x}$ visibly stretches, its magnitude inflating ($||\mathbf{x}|| \gg \sqrt{d}$), causing downstream variance warnings.
- **s04.3** ("RMSNorm measures a vector's root mean square, the square root of the average of its squared components.")
  - *Geometry*: Hero equation appears center screen:
    $$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
  - *Choreography*: Component bars $x_i$ square geometrically into areas, average into a mean square line, and the radical $\sqrt{\cdot}$ computes the scaling radius.
- **s04.4** ("It divides that out, then rescales with a learned gain.")
  - *Choreography*: The vector divides by $\text{RMS}(\mathbf{x})$, contracting its tip onto the unit sphere $||\bar{\mathbf{x}}|| = 1.0$. Learnable vector $\gamma$ scales individual coordinate axes.
- **s04.5** ("Unlike LayerNorm, there is no mean subtraction — saving computation without hurting stability.")
  - *Choreography*: LayerNorm's formula $(x_i - \mu)$ appears side-by-side; the mean $\mu$ term strikes through in clean crimson and fades, leaving only pure scale division.
- **s04.6** ("The vector keeps its direction, but enters attention with controlled magnitude.")
  - *Choreography*: A radial angle guide confirms that the vector's spatial orientation is identical ($\theta = \text{const}$), while its magnitude is strictly bounded.

### Section 05: Self-Attention: The Core Computation (`sec05_attention` · 10 beats · 51.4s)
- **s05.1** ("Now the heart of the model: self-attention.")
  - *Geometry*: The normalized vector $\bar{\mathbf{x}}$ stands poised at center left.
- **s05.2** ("Every token's vector is projected three ways — into a query, a key, and a value.")
  - *Choreography*: The vector passes through three clean linear operator gates, splitting into three distinct colored arrows:
    - $\mathbf{q} = \bar{\mathbf{x}} W_Q$ (`#60A5FA`, Royal Blue)
    - $\mathbf{k} = \bar{\mathbf{x}} W_K$ (`#FBBF24`, Warm Amber)
    - $\mathbf{v} = \bar{\mathbf{x}} W_V$ (`#34D399`, Emerald Green)
- **s05.3** ("Each query takes a dot product against every key in the sequence, producing a score for every pair.")
  - *Geometry*: 2D vector coordinate space.
  - *Choreography*: Query vector $\mathbf{q}_i$ projects orthogonally onto key vectors $\mathbf{k}_1, \dots, \mathbf{k}_T$. Dashed perpendicular lines drop to key shafts, calculating scalar dot products $S_{ij} = \mathbf{q}_i^\top \mathbf{k}_j$.
- **s05.4** ("Those scores are scaled by one over the square root of the key dimension.")
  - *Choreography*: Formula scales: $\frac{S_{ij}}{\sqrt{d_k}} = \frac{S_{ij}}{\sqrt{128}}$. Numbers scale smoothly into a matrix grid.
- **s05.5** ("A causal mask hides the future: each token can look back at everything before it, never ahead.")
  - *Geometry*: Attention score matrix $\mathbf{S}$.
  - *Choreography*: A triangular shading descends over the upper triangle ($j > i$); those entries instantly snap to $-\infty$.
- **s05.6** ("Then a softmax turns them into weights that sum to one.")
  - *Choreography*: Matrix rows morph through $\text{softmax}(\cdot)$; raw numbers become normalized distribution bars summing to $1.00$.
- **s05.7** ("Finally, the weights blend the value vectors.")
  - *Choreography*: Value vectors $\mathbf{v}_1, \dots, \mathbf{v}_t$ scale in length according to attention weights $\alpha_{ij}$.
- **s05.8** ("Each output is a weighted average of what came before.")
  - *Choreography*: Scaled value vectors translate tip-to-tail, summing into the single attended vector $\mathbf{z}_i = \sum_j \alpha_{ij} \mathbf{v}_j$.
- **s05.9** ("And this runs not once, but eight times, in parallel heads with independent projections.")
  - *Choreography*: The stage fans out into 8 distinct parallel geometric planes, showing all 8 heads calculating independent subspaces concurrently.
- **s05.10** ("Their outputs are concatenated and mixed by one last matrix.")
  - *Choreography*: The 8 head vectors stack into a composite column $[\mathbf{z}_1; \dots; \mathbf{z}_8]$ and pass through projection matrix $W_O \in \mathbb{R}^{1024 \times 1024}$ to form output update $\Delta \mathbf{x}_{\text{attn}}$.

### Section 06: Rotary Positions — RoPE (`sec06_rope` · 9 beats · 38.6s)
- **s06.1** ("As defined, attention is order-blind — it sees a bag of tokens.")
  - *Choreography*: Token positions scramble randomly; the calculated dot product values remain frozen, demonstrating total permutation invariance.
- **s06.2** ("Positions enter through rotary embeddings.")
  - *Geometry*: Rotary matrix operator $R_{\Theta, m}^d$ appears in violet (`#A78BFA`).
- **s06.3** ("Each head's query and key vectors are split into pairs.")
  - *Choreography*: The 128-dimensional vector divides into 64 2D coordinate planes: $[(x_1, x_2), (x_3, x_4), \dots, (x_{127}, x_{128})]$.
- **s06.4** ("Each pair is rotated by an angle proportional to the token's position.")
  - *Geometry*: A 2D Cartesian plane with unit circle.
  - *Choreography*: Vector pair $(x_1, x_2)$ rotates smoothly counter-clockwise by angle $\theta = m \cdot \theta_i$.
- **s06.5** ("The pairs rotate at different frequencies.")
  - *Geometry*: Three concentric dials appear side-by-side representing frequency bands $\theta_0, \theta_{16}, \theta_{63}$.
- **s06.6** ("Some spin fast, pinning down nearby positions; others barely move, encoding long-range structure.")
  - *Choreography*: As position $m$ steps $0 \to 1 \to 2 \dots$, the high-frequency dial spins briskly while the lowest-frequency dial creeps imperceptibly.
- **s06.7** ("The elegance: a rotated query meeting a rotated key gives a special dot product.")
  - *Choreography*: Query $\mathbf{q}$ at position $m$ and Key $\mathbf{k}$ at position $n$ are plotted on the same dial.
- **s06.8** ("It depends only on the distance between the tokens, never on absolute position.")
  - *Choreography*: Both vectors rotate simultaneously around the circle as sequence offset shifts; the angle between them $\Delta\theta = (m - n)\theta_i$ remains perfectly stationary!
- **s06.9** ("With theta at five hundred thousand, the slowest pairs stay smooth across the entire two-thousand-forty-eight token context.")
  - *Choreography*: A horizontal timeline stretches to $m=2048$; the slowest dial completes less than one quarter revolution, preserving global context coordinates.

### Section 07: Grouped-Query Attention & QK-Norm (`sec07_gqa` · 8 beats · 36.2s)
- **s07.1** ("Attention itself is grouped.")
  - *Geometry*: Architectural layout of attention heads.
- **s07.2** ("LLaMA-3-Lite uses eight query heads but only four key-and-value heads.")
  - *Choreography*: 8 blue Query circles ($Q_1 \dots Q_8$) align across the top; 4 amber/green KV pairs ($KV_1 \dots KV_4$) align below.
- **s07.3** ("Each KV head is shared by two query heads.")
  - *Choreography*: Geometric branching brackets connect $Q_1, Q_2 \to KV_1$; $Q_3, Q_4 \to KV_2$, illustrating the 2:1 grouping ratio.
- **s07.4** ("The KV cache stores keys and values for every layer and every past token.")
  - *Geometry*: A 3D tensor memory stack materializes, accumulating past token slices along the time axis.
- **s07.5** ("Halving the KV heads halves that memory, and the cache is what grows with context.")
  - *Choreography*: A side-by-side memory footprint bar comparison shows standard MHA (8 KV heads) shrinking cleanly by 50% under GQA (4 KV heads).
- **s07.6** ("And before rotation, queries and keys each pass through RMSNorm.")
  - *Choreography*: Dedicated RMSNorm operator symbols clamp directly onto $Q$ and $K$ branches immediately before RoPE rotation.
- **s07.7** ("It is called QK-norm, and it bounds their magnitude so logits stay well-behaved.")
  - *Choreography*: A scatter plot of logit values is shown bounded neatly within $[-C, C]$, preventing variance explosions.
- **s07.8** ("A Lite addition — LLaMA 3 itself has none.")
  - *Choreography*: A clean mathematical notation notes this deliberate stabilization feature in LLaMA-3-Lite.

### Section 08: SwiGLU: The Gated Feed-Forward (`sec08_swiglu` · 7 beats · 31.9s)
- **s08.1** ("After attention comes the second sublayer: a feed-forward network applied to each token independently.")
  - *Geometry*: Stream vector arrives at the Feed-Forward Block.
- **s08.2** ("The vector is expanded to four thousand ninety-six dimensions, then compressed back to one thousand twenty-four.")
  - *Choreography*: Vector width expands geometrically from 1024 to 4096, forming a wide computational matrix, before tapering back to 1024.
- **s08.3** ("LLaMA uses a gated variant, SwiGLU.")
  - *Geometry*: Core equation writes in `MathTypst`:
    $$\text{SwiGLU}(\mathbf{x}) = \left(\text{SiLU}(\mathbf{x} W_{\text{gate}}) \odot \mathbf{x} W_{\text{up}}\right) W_{\text{down}}$$
- **s08.4** ("The expansion splits into two paths.")
  - *Choreography*: Input vector branches into two parallel pathways: Gate path and Up path.
- **s08.5** ("One passes through the SiLU activation, a smooth ReLU.")
  - *Geometry*: 2D function plot.
  - *Choreography*: Plot of $\text{SiLU}(x) = x \cdot \sigma(x)$ draws in Sky Blue (`#38BDF8`), contrasting with a dashed standard ReLU line to emphasize continuous differentiability.
- **s08.6** ("It gates the other path element by element, scaling each feature rather than switching it off.")
  - *Choreography*: Values from the SiLU curve act as continuous multipliers $\odot$ onto the Up path vector, dynamically throttling or boosting feature amplitudes.
- **s08.7** ("Then the result is projected back.")
  - *Choreography*: Matrix $W_{\text{down}}$ contracts the 4096-dimensional activation back to 1024 dimensions, adding back into the residual stream.

### Section 09: Unembedding & Autoregression (`sec09_output` · 8 beats · 31.8s)
- **s09.1** ("Stack sixteen of these blocks, and the stream reaches a final RMSNorm.")
  - *Choreography*: Camera zooms out to reveal all 16 stacked blocks; residual stream emerges from Block 16 and passes through final RMSNorm.
- **s09.2** ("The output head scores every vocabulary item — a dot product against one hundred twenty-eight thousand embeddings, producing logits.")
  - *Choreography*: Final vector $\mathbf{x}_{\text{final}}$ multiplies against $W_{\text{embed}}^\top$, generating a dense 128,256-dimensional logit vector $\mathbf{z}$.
- **s09.3** ("A softmax turns them into probabilities.")
  - *Choreography*: Logits exponentiate and normalize into probability simplex bars summing to 1.0.
- **s09.4** ("The model samples the next token and appends it.")
  - *Choreography*: The highest probability candidate highlights; its token text emits and appends to the input prompt line.
- **s09.5** ("The whole network runs again — one forward pass per generated token.")
  - *Choreography*: A sleek feedback loop curves from output back to input; the full network pulses as the new forward pass begins.
- **s09.6** ("And that is the entire architecture: an embedding, a residual stream, and sixteen blocks.")
  - *Geometry*: Global architectural blueprint aligns in balanced mathematical harmony.
- **s09.7** ("Normalized, rotary, grouped, gated computation, then an unembedding head.")
  - *Choreography*: The four distinctive symbols highlight in succession: RMSNorm, RoPE Dial, GQA Group, SwiGLU Gate.
- **s09.8** ("Five hundred fifteen million parameters — and nothing else.")
  - *Choreography*: Clean, authoritative closing title with exact parameter count: $\approx 515\text{M}$ parameters. Frame rests in quiet elegance.

---

## 4. Implementation Phasing & Prototype Gate

### Step 1: Core Mathematical Geometry Helpers (`render_scenes.py`)
- Remove `_build_ambient` (blooms, drifting circles) and `_build_chrome` (progress bar, chapter pill).
- Introduce clean helper classes: `CartesianStage`, `VectorArrow`, `MatrixGrid`, `PolarDial`.

### Step 2: Prototype Gate (Section 05 & Section 06)
- Rebuild Section 05 (Self-Attention) and Section 06 (RoPE).
- Verify with layout check:
  ```bash
  .venv/bin/python render_scenes.py Section05 Section06 --check_layout -qm -r 1920,1080 --format=mp4
  ```
- Inspect contact sheets and video frames for mathematical elegance, zero clutter, and flawless motion.

### Step 3: Full Stack Rebuild (Sections 01–04, 07–09)
- Implement remaining scenes according to the storyboard.
- Maintain object constancy of the token vector throughout.

### Step 4: Test Suite Synchronization
- Update exact coordinate tests in `tests/test_phase1b.py` to match new mathematical positions.
- Verify 100% pass on all 70 test assertions in `tests/`.

---

## 5. Verification & Acceptance Criteria
1. **Zero AI Slop**: No floating background orbs, no rounded glass cards, no pill badges, no bottom progress scrubber.
2. **Strict Time Budget**: Total video duration matches $296.45\text{s} \pm 0.1\text{s}$.
3. **Motion Rhythm**: Freeze ratio $\le 35\%$, no static hold $> 3.5\text{s}$.
4. **Layout Safety**: All text and formulas remain within `SAFE = (-6.7, 6.7, -3.6, 3.6)`.
5. **Color & Accessibility**: All text contrast $\ge 7:1$, semantic marks $\Delta E \ge 15$, secondary encoding on every mark.
