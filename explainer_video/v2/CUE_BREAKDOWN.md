# Cue breakdown — Inside LLaMA-3-Lite — The Complete Architecture

167 cues, one per spoken sentence. The cue id is the sentence id in `narration.yaml`, so a rename touches both. Times are measured from `audio/timings.json` (R4a re-record): lead 0.3 s, tail 0.4 s, gap 0.2 s. Pure speech 705.2 s. Narration total with the gaps 736.2 s = 12.3 min. Times in the tables are section-clip local.

The screen column is the `visual` line from `narration.yaml`. It is the brief. The old-code column names the scene in `render_scenes.py` that can carry the visual. That mapping is section level; the code block behind each old cue still needs a look during the re-cue. Anything marked `new` has no source code.

Old cue ids look like the new sentence ids and are not the same lines. `s01.1` in the old code is a different line from `s01.1` in `narration.yaml`.

## Sections

| # | section | chapter | cues | span | build | old code |
|---|---|---|---|---|---|---|
| 1 | `sec01_function` | 01 · THE FUNCTION | 12 | 53.7 s | carry | Section01 |
| 2 | `sec02_objective` | 02 · THE OBJECTIVE | 15 | 63.3 s | new | — |
| 3 | `sec03_tokens` | 03 · TOKENS | 14 | 69.9 s | new | — |
| 4 | `sec04_embedding` | 04 · EMBEDDINGS | 19 | 83.4 s | merge | Section02 + Section03 + Section04 |
| 5 | `sec05_matching` | 05 · ATTENTION: MATCHING | 15 | 56.9 s | split | Section05 |
| 6 | `sec06_blend` | 06 · ATTENTION: BLENDING | 16 | 60.6 s | split | Section05 |
| 7 | `sec07_rope` | 07 · ROTARY POSITIONS | 16 | 76.3 s | renumber | Section06 |
| 8 | `sec08_qknorm` | 08 · QK-NORM | 7 | 30.8 s | split | Section07 |
| 9 | `sec09_gqa` | 09 · GQA & THE KV CACHE | 15 | 66.1 s | split | Section07 |
| 10 | `sec10_swiglu` | 10 · SWIGLU | 16 | 69.8 s | renumber | Section08 |
| 11 | `sec11_loop` | 11 · THE GENERATION LOOP | 13 | 61.9 s | split | Section09 |
| 12 | `sec12_recap` | 12 · THE WHOLE PICTURE | 9 | 43.5 s | split | Section09 |

Build keys: `carry` — keep the scene, grow it. `new` — write the scene from nothing. `merge` — one new scene from several old ones. `split` — one old scene becomes two. `renumber` — same visuals, new ids.

## 01 · The model as a function

`sec01_function` · 01 · THE FUNCTION · 12 cues · 53.7 s · build **carry**

**Screen.** Title card to token chips t1 t2 t3 to arrow to next-token probability bars ("one function: tokens to probabilities") to spec chips to demo sentence "The cat sat on the mat" with IDs to vertical silhouette of the 16-block stack; one block circled.

**Old code.** Section01 — all of it; 5 cues grow to 12. Old cues: Section01 s01.1–s01.5 (5).

| cue | says | screen | old code |
|---|---|---|---|
| `s01.1` 00:00.0–00:02.8 | A large language model is one mathematical function. | Title card settles; "one mathematical function" takes the glow. | Section01 |
| `s01.2` 00:03.0–00:08.1 | It reads a sequence of tokens and returns a probability for every possible next token. | Token chips t1 t2 t3 enter left; the arrow runs right into the probability bars. | Section01 |
| `s01.3` 00:08.3–00:12.8 | That is the whole interface: text in, a distribution over the next token out. | The input and output labels stamp onto the two ends of the demo. | Section01 |
| `s01.4` 00:13.0–00:18.6 | LLaMA-3-Lite is a compact, from-scratch model built in the LLaMA-3 style. | The first spec chips build in beneath the demo. | Section01 |
| `s01.5` 00:18.8–00:25.4 | Sixteen decoder blocks, a model dimension of one thousand twenty-four, and about five hundred fifteen million parameters. | The remaining chips land together; caption holds. | Section01 |
| `s01.6` 00:25.6–00:28.4 | The same model width runs through all sixteen blocks. | Spec chips pulse once in sequence. | Section01 |
| `s01.7` 00:28.6–00:31.0 | Take the sentence on screen: The cat sat on the mat. | The demo sentence writes itself in under the chips. | Section01 |
| `s01.8` 00:31.2–00:35.0 | Six tokens, six IDs, and one question: what comes next? | ID numbers settle under the chips; a question mark hangs over the gap at the end. | Section01 |
| `s01.9` 00:35.2–00:41.0 | One token walks in as an ID and walks out as a probability over the whole vocabulary. | A single chip travels the demo path from ID to bars. | Section01 |
| `s01.10` 00:41.2–00:45.0 | Follow that walk from raw text to the sampling loop, one stage at a time. | Demo fades; the 16-block silhouette rises, embedding above and unembedding below, then one block is circled. | Section01 |
| `s01.11` 00:45.2–00:49.9 | Each stage exists to fix a specific problem the previous stage leaves open. | Stage ticks light one by one along the silhouette. | Section01 |
| `s01.12` 00:50.1–00:53.7 | The loop at the end is what turns one prediction into a paragraph. | A short paragraph types itself under the circled block. | Section01 |

## 02 · Why next-token prediction

`sec02_objective` · 02 · THE OBJECTIVE · 15 cues · 63.3 s · build **new**

**Screen.** Training card: real text with the next token hidden; probability gauge on the hidden token; loss dial labelled "surprise" turning down; weight grid updating; empty tag row (no labels) fading out.

**Old code.** — — no source code; objective card. Old cues: none — new build.

| cue | says | screen | old code |
|---|---|---|---|
| `s02.1` 00:00.0–00:03.2 | Before any architecture, ask what the model is trained to do. | Section header; an empty objective card opens. | new — no source |
| `s02.2` 00:03.4–00:08.9 | The objective is next-token cross-entropy, plus a small z-loss at one ten-thousandth. | "next-token prediction" writes itself onto the card. | new — no source |
| `s02.3` 00:09.1–00:12.1 | Training shows the model real text with the next token hidden. | A sentence scrolls in; its last token blinks out behind a mask. | new — no source |
| `s02.4` 00:12.3–00:15.2 | The model assigns that hidden token a probability. | A probability gauge ticks up on the hidden slot. | new — no source |
| `s02.5` 00:15.4–00:19.0 | When the probability is low, training adjusts the weights to raise it. | The weight grid shuffles as the gauge climbs. | new — no source |
| `s02.6` 00:19.2–00:24.6 | That adjustment is the whole of learning: every weight moves to make real text more likely. | The grid settles; a "more likely" arrow rises beside the text. | new — no source |
| `s02.7` 00:24.8–00:31.3 | Next-token training needs no extra manual labels: no sentiment tags, no answers, no rules. | The tag row fades out empty under the objective card. | new — no source |
| `s02.8` 00:31.5–00:36.1 | The loss measures surprise: how low the model's probability was for the true token. | The surprise dial reads high, then turns down as probability rises. | new — no source |
| `s02.9` 00:36.3–00:40.5 | Pushing surprise down is what shapes the five hundred fifteen million parameters. | Parameter tiles mould themselves as the dial falls. | new — no source |
| `s02.10` 00:40.7–00:46.0 | From that single objective, grammar, style and world knowledge settle into the weights. | Three faint overlays drift into the weight grid and dissolve. | new — no source |
| `s02.11` 00:46.2–00:48.3 | Nothing in the code says what a noun is. | A "noun" label tries to stick to the grid and slides off. | new — no source |
| `s02.12` 00:48.5–00:52.2 | Training runs that prediction at every position in parallel. | Six prediction marks light along the training sentence at once. | new — no source |
| `s02.13` 00:52.4–00:56.1 | Generation instead reveals one new token at a time. | The marks collapse to one; a single token writes at the end. | new — no source |
| `s02.14` 00:56.3–01:00.5 | The structure you are about to see is the machinery that makes the prediction possible. | The 16-block silhouette ghosts in behind the objective card. | new — no source |
| `s02.15` 01:00.7–01:03.3 | Each piece is there because the objective demands it. | The silhouette resolves to full colour as the card fades. | new — no source |

## 03 · Text becomes IDs

`sec03_tokens` · 03 · TOKENS · 14 cues · 69.9 s · build **new**

**Screen.** "The cat sat on the mat" to six token chips with real IDs (791, 8415, 7731, 389, 279, 5634); byte-pair merge demo; character bar vs word table vs subword bar; vocab count 128,256.

**Old code.** — — no source code; tokenizer, BPE, vocab. Old cues: none — new build.

| cue | says | screen | old code |
|---|---|---|---|
| `s03.1` 00:00.0–00:04.5 | Text cannot enter a matrix multiply, so the tokenizer goes first. | Section header; the sentence writes itself in at centre. | new — no source |
| `s03.2` 00:04.7–00:11.7 | It splits text into subword pieces from a vocabulary of one hundred twenty-eight thousand two hundred fifty-six tokens. | The sentence breaks into six token chips, each with its ID below. | new — no source |
| `s03.3` 00:11.9–00:17.2 | The scheme is byte-pair encoding: the most frequent character pairs merge into new tokens. | Merge demo: two letters glide together and stamp a new token. | new — no source |
| `s03.4` 00:17.4–00:21.5 | Building it starts from single characters and keeps merging the most common pair. | A merge ladder builds upward from letters to whole words. | new — no source |
| `s03.5` 00:21.7–00:27.4 | Common words stay whole, and rare words break into smaller pieces the model has seen before. | Whole-word chips glow solid; a longer chip splits into fragments. | new — no source |
| `s03.6` 00:27.6–00:33.4 | Characters alone would make sequences far too long; whole words alone would explode the vocabulary. | Two scale bars contrast: a long character bar and a huge word table. | new — no source |
| `s03.7` 00:33.6–00:38.9 | Subwords sit in between and cover any input, including words the model has never seen. | The subword chip bar settles between the two extremes. | new — no source |
| `s03.8` 00:39.1–00:41.7 | Their IDs come straight from the vocabulary table. | ID numbers light up under each of the six chips. | new — no source |
| `s03.9` 00:41.9–00:48.2 | The word cat becomes eight thousand four hundred fifteen; the word mat becomes five thousand six hundred thirty-four. | Those two IDs flash on the cat and mat chips. | new — no source |
| `s03.10` 00:48.4–00:52.3 | Each ID is just a row number; it carries no meaning of its own. | The numbers lift clear of the chips and hover as bare indices. | new — no source |
| `s03.11` 00:52.5–00:55.0 | The tokenizer knows nothing about cats. | A cat icon shrugs at the tokenizer box and fades. | new — no source |
| `s03.12` 00:55.2–00:59.2 | Its only job is to cut text into pieces the model has a row for. | Scissors tick along the sentence, leaving valid chips behind. | new — no source |
| `s03.13` 00:59.4–01:05.5 | The config lists one hundred twenty-eight thousand rows; the tokenizer holds two hundred fifty-six more. | Two counts compare on a small counter card. | new — no source |
| `s03.14` 01:05.7–01:09.9 | The embedding table sizes itself to the tokenizer, not to the config. | The table grows two hundred fifty-six rows to match; 128,256 stamps on top. | new — no source |

## 04 · IDs become vectors

`sec04_embedding` · 04 · EMBEDDINGS · 19 cues · 83.4 s · build **merge**

**Screen.** Embedding matrix with the `cat` row highlighted, pulled out as a numeric vector x = E[t2]; formula x_i = E[t_i]; row length 1024; 2-D sketch of the meaning space (cat/kitten close, car far); high-D cloud pull-back; residual-stream rail with a pre-norm block diagram (RMSNorm to attention and SwiGLU, results added back).

**Old code.** Section02 + Section03 + Section04 — embed matrix and meaning sketch; residual rail; pre-norm block. Old cues: Section02 s02.1–s02.7 (7), Section03 s03.1–s03.4 (4), Section04 s04.1–s04.6 (6).

| cue | says | screen | old code |
|---|---|---|---|
| `s04.1` 00:00.0–00:03.9 | An ID is only a row number; the row it points to carries the meaning. | Section header; an index arrow points into the matrix. | Section02 |
| `s04.2` 00:04.1–00:08.0 | That lookup is the embedding matrix, one row per vocabulary item. | The embedding matrix appears; the `cat` row highlights. | Section02 |
| `s04.3` 00:08.2–00:13.5 | Each row holds one thousand twenty-four numbers, so every token becomes a vector of that length. | The row length label lands: 1024. | Section02 |
| `s04.4` 00:13.7–00:19.0 | A lookup is all it takes: the ID picks the row, and the row becomes the token's vector. | The highlighted row pulls out as x = E[t2]. | Section02 |
| `s04.5` 00:19.2–00:22.2 | There is no computation here yet, only a learned table. | The matrix stays still; a lookup tick marks it. | Section02 |
| `s04.6` 00:22.4–00:27.3 | Training can move these rows so tokens used in similar ways land near each other. | The 2-D meaning-space sketch fades in; points drift into clusters. | Section02 |
| `s04.7` 00:27.5–00:31.8 | This is how it can learn semantic geometry: directions that carry meaning. | Axis arrows label themselves on the sketch. | Section02 |
| `s04.8` 00:32.0–00:35.9 | An illustrative sketch might place cat near kitten and far from car. | The cat and kitten points link; the car point stays distant. | Section02 |
| `s04.9` 00:36.1–00:40.7 | Because the table is learned, it embeds each token by how that token is used. | New points settle into the sketch one by one. | Section02 |
| `s04.10` 00:40.9–00:46.0 | The vector is a point in one thousand twenty-four dimensions, far too many to picture at once. | The 2-D sketch pulls back into a cloud of faint high-D points. | Section02 |
| `s04.11` 00:46.2–00:50.1 | Two dimensions stand in for the rest whenever the screen shows a sketch. | The cloud flattens honestly back to the labelled 2-D axes. | Section02 |
| `s04.12` 00:50.3–00:54.0 | From this point on the model sees only vectors, never characters. | Characters dissolve off the chips; the vector sits alone at centre, glowing. | Section02 |
| `s04.13` 00:54.2–00:58.4 | From here, each token carries its vector through all sixteen blocks. | The vector rides downward through a 16-block rail. | Section03: residual rail, add-back arrows |
| `s04.14` 00:58.6–01:01.1 | That carried vector is called the residual stream. | The rail labels itself: residual stream. | Section03: residual rail, add-back arrows |
| `s04.15` 01:01.3–01:06.5 | Inside a block, attention and SwiGLU each read a normalized copy of the stream. | Two taps off the rail pass RMSNorm chips into attention and SwiGLU. | Section03 rail + Section04 pre-norm block |
| `s04.16` 01:06.7–01:09.8 | Each writes its result back onto the stream by simple addition. | Two add-back arrows return each result to the rail. | Section03 rail + Section04 pre-norm block |
| `s04.17` 01:10.0–01:13.5 | This pre-norm shape is the structure of every decoder block. | The whole block frame stamps PRE-NORM; the frame repeats. | Section03 rail + Section04 pre-norm block |
| `s04.18` 01:13.7–01:18.2 | A last RMSNorm at the output cleans that same stream before scoring. | A final RMSNorm chip lands on the rail before the output. | Section03 rail + Section04 pre-norm block |
| `s04.19` 01:18.4–01:23.4 | A separate, untied table of the same kind runs at the output, as you will see. | A faint echo of the matrix ghosts in near the unembedding row. | Section03: residual rail |

## 05 · Queries, keys and scores

`sec05_matching` · 05 · ATTENTION: MATCHING · 15 cues · 56.9 s · build **split**

**Screen.** Token row; query token circled; q/k/v projections with W_Q, W_K, W_V; 4x4 score matrix filling with dot products; aligned-arrow similarity flash; divide-by-sqrt(d_k) chip; variance bar flattening.

**Old code.** Section05 — q/k/v and score grid. Old cues: Section05 s05.1–s05.10 (10).

| cue | says | screen | old code |
|---|---|---|---|
| `s05.1` 00:00.0–00:03.6 | A single vector knows its own meaning and nothing about its neighbors. | Section header; one vector glows alone on the empty field. | Section05 |
| `s05.2` 00:03.8–00:07.5 | The next problem is context: how a token learns what surrounds it. | Surrounding tokens ghost in around the lone vector. | Section05 |
| `s05.3` 00:07.7–00:13.2 | Self-attention solves this in one layer, by letting every token look back at the tokens before it. | The token row lands; the query token is circled; look-back arcs draw. | Section05 |
| `s05.4` 00:13.4–00:18.7 | Each vector is projected three times: into a query, a key, and a value. | q, k and v projections appear with their W_Q, W_K, W_V labels. | Section05 |
| `s05.5` 00:18.9–00:20.7 | The query is what a token asks for. | The query badge pulses with a question mark. | Section05 |
| `s05.6` 00:20.9–00:23.0 | The key is what a token offers for matching. | The key badges light along the token row. | Section05 |
| `s05.7` 00:23.2–00:26.5 | The value is the information a token hands over once it is chosen. | The value badge slides forward from the row. | Section05 |
| `s05.8` 00:26.7–00:30.9 | All three projections are learned matrices, multiplied into the stream. | W_Q, W_K, W_V matrix tiles glow beside their arrows. | Section05 |
| `s05.9` 00:31.1–00:34.5 | A query takes a dot product with every key in the sequence. | The 4x4 score grid fills cell by cell. | Section05 |
| `s05.10` 00:34.7–00:38.6 | A large dot product means similar direction, long length, or both. | Two aligned arrows flash as their score cell jumps high. | Section05 |
| `s05.11` 00:38.8–00:41.7 | Each dot product fills one cell of the score matrix. | The last empty score cell snaps in. | Section05 |
| `s05.12` 00:41.9–00:44.7 | Those scores are divided by the square root of the key dimension. | The divide-by-sqrt(d_k) pill lands; the matrix dims behind it. | Section05 |
| `s05.13` 00:44.9–00:49.2 | Without scaling, scores grow with the dimension and the softmax saturates. | A spike bar shoots up, then flattens after the pill applies. | Section05 |
| `s05.14` 00:49.4–00:53.8 | Scaling controls that growth and keeps the softmax away from saturation. | The variance bar settles from wild to flat. | Section05 |
| `s05.15` 00:54.0–00:56.9 | The matrix on screen is the map of who will listen to whom. | The grid takes a soft glow; cell heights settle into place. | Section05 |

## 06 · Mask, softmax and heads

`sec06_blend` · 06 · ATTENTION: BLENDING · 16 cues · 60.6 s · build **split**

**Screen.** Causal mask with minus-infinity cells; softmax weight bars for the `sat` row showing 0.12 / 0.86 / 0.02 / 0.00; weighted blend into o3; row of 8 head badges; concat and W_O; green write arrow to the residual stream.

**Old code.** Section05 — mask, softmax, head blend. Old cues: Section05 s05.1–s05.10 (10).

| cue | says | screen | old code |
|---|---|---|---|
| `s06.1` 00:00.0–00:03.2 | Scores are still raw numbers; they need to become weights. | Section header; the score grid waits above empty weight bars. | Section05 |
| `s06.2` 00:03.4–00:06.3 | A causal mask hides the future from every query. | The upper triangle reddens and hatches; "future hidden". | Section05 |
| `s06.3` 00:06.5–00:11.0 | A token may attend to itself and to earlier tokens, never later ones. | Masked cells grey out; the lower triangle stays live. | Section05 |
| `s06.4` 00:11.2–00:15.2 | Training never allows a token to read the answer it is being asked to predict. | A future chip shows the answer, then blinks out behind the mask. | Section05 |
| `s06.5` 00:15.4–00:19.4 | Masked scores are pushed to negative infinity before the softmax runs. | Masked cells tick to minus infinity. | Section05 |
| `s06.6` 00:19.6–00:24.1 | Softmax turns the remaining scores into positive weights that sum to one. | Weight bars grow for the highlighted query row. | Section05 |
| `s06.7` 00:24.3–00:26.3 | Look at the row for the word sat on screen. | The sat row highlights on the matrix and the bars. | Section05 |
| `s06.8` 00:26.5–00:29.8 | It places zero point eight six of its weight on cat. | The cat bar rises to 0.86 and glows. | Section05 |
| `s06.9` 00:30.0–00:32.0 | The masked future cell holds zero. | The last bar sits at 0.00 under the hatched mask. | Section05 |
| `s06.10` 00:32.2–00:35.4 | The weights then blend the value vectors into a single output. | Weighted arrows converge from the token row into o3. | Section05 |
| `s06.11` 00:35.6–00:39.9 | The output is a weighted average of past values, so it stays near their neighborhood. | The o3 point settles inside the cloud of value vectors. | Section05 |
| `s06.12` 00:40.1–00:43.6 | Each token leaves attention carrying a summary of its own past. | The o3 box lights up with its weighted average. | Section05 |
| `s06.13` 00:43.8–00:47.9 | This whole computation runs eight times in parallel, one per head. | Eight head badges slide in under the diagram. | Section05 |
| `s06.14` 00:48.1–00:53.2 | Query projections are per head, but keys and values are shared by pairs of heads. | Projection sets fan out per head badge. | Section05 |
| `s06.15` 00:53.4–00:57.5 | The eight outputs are concatenated and multiplied by one output matrix. | Concat to W_O caption writes in. | Section05 |
| `s06.16` 00:57.7–01:00.6 | The result is written back into the residual stream. | A green write arrow returns to the band; the banner holds. | Section05 |

## 07 · Rotary positions: RoPE

`sec07_rope` · 07 · ROTARY POSITIONS · 16 cues · 76.3 s · build **renumber**

**Screen.** Shuffled sentences with identical scores; empty position slot; circle with a vector rotating by m-theta for m = 0 to 7; rotation matrix and complex form; 64-pair diagram; frequency ladder with period labels; property R_m q dot R_n k = g(m-n); chip theta = 500 000.

**Old code.** Section06 — renumber only. Old cues: Section06 s06.1–s06.9 (9).

| cue | says | screen | old code |
|---|---|---|---|
| `s07.1` 00:00.0–00:05.6 | Attention as written cannot see word order: shuffle the tokens and the outputs shuffle to match. | Two shuffled sentences land with identical scores. | Section06 |
| `s07.2` 00:05.8–00:11.6 | The scores only see vectors, and the vectors so far carry meaning without position. | Meaning chips label the vectors; a position slot sits empty. | Section06 |
| `s07.3` 00:11.8–00:14.6 | So the next piece adds position without touching the meaning. | A position tile slides toward the empty slot. | Section06 |
| `s07.4` 00:14.8–00:19.8 | Rotary embeddings fix this by rotating query and key vectors according to position. | Section header; a rotating arrow appears on the circle. | Section06 |
| `s07.5` 00:20.0–00:24.4 | Each vector is split into pairs of dimensions, sixty-four pairs per head. | The vector splits into paired dimensions across the circle. | Section06 |
| `s07.6` 00:24.6–00:29.0 | Each pair is read as a point in two dimensions, or as a complex number. | One pair zooms to a 2-D point with real and imaginary axes. | Section06 |
| `s07.7` 00:29.2–00:34.0 | Position m rotates every pair by an angle of m times a base frequency. | Rotation matrix and e^{im theta} write in; m animates 0 to 7. | Section06 |
| `s07.8` 00:34.2–00:37.7 | The rotation matrix on screen is the pair version of that turn. | The matrix cells align with the turning pair on the circle. | Section06 |
| `s07.9` 00:37.9–00:40.0 | Different pairs turn at different speeds. | Three circles appear, one per frequency band. | Section06 |
| `s07.10` 00:40.2–00:44.7 | Fast pairs pin down nearby structure; slow pairs mark long-range position. | Fast, medium and slow circles, each with its period label. | Section06 |
| `s07.11` 00:44.9–00:49.3 | The fastest pair shown completes a full turn in about six point three tokens. | The fastest circle completes a turn while the slowest barely moves. | Section06 |
| `s07.12` 00:49.5–00:55.5 | When a rotated query meets a rotated key, their rotations combine into one relative turn. | The two rotation arcs flash and neutralize on the score cell. | Section06 |
| `s07.13` 00:55.7–00:59.3 | So the positional part of their dot product depends on the distance between them. | R_m q dot R_n k = g(m-n) writes in. | Section06 |
| `s07.14` 00:59.5–01:05.2 | Content still shapes the score; position adds only how far apart the two tokens sit. | Two sequences of different start points show matching score curves. | Section06 |
| `s07.15` 01:05.4–01:11.5 | With theta at five hundred thousand, the slowest pairs stay smooth across two thousand forty-eight tokens. | theta = 500 000 chip; the slow circle's arc stretches over the context bar. | Section06 |
| `s07.16` 01:11.7–01:16.3 | Theta sets the slowest wavelength; a larger theta pushes it further. | The theta dial turns and the slowest arc lengthens with it. | Section06 |

## 08 · A Lite detail: QK-norm

`sec08_qknorm` · 08 · QK-NORM · 7 cues · 30.8 s · build **split**

**Screen.** QK-norm pipeline q to RMSNorm to RoPE and k to RMSNorm to RoPE; per-head gamma chip (128); magnitude bars clamping; caveat card "a Lite addition: LLaMA 3 itself has none".

**Old code.** Section07 — QK strip. Old cues: Section07 s07.1–s07.8 (8).

| cue | says | screen | old code |
|---|---|---|---|
| `s08.1` 00:00.0–00:03.3 | One Lite detail lands between projection and rotation. | Section header; the q and k paths pause before the RoPE dial. | Section07 |
| `s08.2` 00:03.5–00:08.1 | Queries and keys each pass through RMSNorm before RoPE is applied. | QK-norm pipeline fills: q to RMSNorm, k to RMSNorm. | Section07 |
| `s08.3` 00:08.3–00:13.2 | This controls the scale of each query and key, so the logits stay well behaved. | Magnitude bars clamp to a bound; the logits settle. | Section07 |
| `s08.4` 00:13.4–00:17.0 | The norm runs per head, on each query and key vector on its own. | Head badges each carry their own small norm chip. | Section07 |
| `s08.5` 00:17.2–00:22.4 | Each layer has one shared gain of one hundred twenty-eight numbers for queries, and one for keys. | The 128 gamma chip lights on a single head badge. | Section07 |
| `s08.6` 00:22.6–00:26.5 | The point of the step is training stability, at a tiny parameter cost. | A tiny parameter sliver sits beside a steady loss curve. | Section07 |
| `s08.7` 00:26.7–00:30.8 | LLaMA 3 itself has no QK-norm; the step is a Lite addition. | The QK-norm chip carries its caveat: a Lite addition, not LLaMA 3. | Section07 |

## 09 · Grouped-query attention and the KV cache

`sec09_gqa` · 09 · GQA & THE KV CACHE · 15 cues · 66.1 s · build **split**

**Screen.** Generation strip with tokens arriving one by one; KV cache stacks growing; math card (16 x 2 x 4 x 128 = 16 384 per token; tens of millions at full context); multi-head 8:8 diagram morphing to grouped 8:4 with pairing arcs; MHA vs GQA stacks with divide-by-2 arrow.

**Old code.** Section07 — GQA and KV cache. Old cues: Section07 s07.1–s07.8 (8).

| cue | says | screen | old code |
|---|---|---|---|
| `s09.1` 00:00.0–00:03.2 | Attention during training sees the whole sentence at once. | Section header; the full token row lights in one pass. | Section07 |
| `s09.2` 00:03.4–00:08.8 | Generation is different: the model writes one token, then the next, then the next. | The generation strip appends chips one at a time. | Section07 |
| `s09.3` 00:09.0–00:12.1 | Each new token still has to see everything before it. | Look-back arcs redraw from the newest chip over the whole strip. | Section07 |
| `s09.4` 00:12.3–00:17.3 | Recomputing every past key and value at every step would waste work. | A ghost recompute pass flashes over the old chips, then fades. | Section07 |
| `s09.5` 00:17.5–00:21.4 | An inference-time optimization can store those vectors in a KV cache. | Cache stacks rise beside the strip, one bar per past token. | Section07 |
| `s09.6` 00:21.6–00:25.4 | Each layer stores two vectors per token for every key-value head. | A K tile and a V tile pair up on one cache bar. | Section07 |
| `s09.7` 00:25.6–00:33.1 | With sixteen layers and four key-value heads, one token costs sixteen thousand three hundred eighty-four stored numbers. | Math card builds: 16 x 2 x 4 x 128 = 16 384. | Section07 |
| `s09.8` 00:33.3–00:36.0 | That count assumes one hundred twenty-eight numbers per head. | The 128 head-dim chip highlights on the math card. | Section07 |
| `s09.9` 00:36.2–00:41.5 | At two thousand forty-eight tokens of context, the cache passes thirty-three million numbers. | The cache stack grows against a context ruler to the full context. | Section07 |
| `s09.10` 00:41.7–00:44.7 | The cache is a growing memory cost at inference time. | A memory meter fills mostly with cache colour. | Section07 |
| `s09.11` 00:44.9–00:49.9 | LLaMA-3-Lite runs eight query heads and four key-value heads. | The multi-head diagram lands with head counts. | Section07 |
| `s09.12` 00:50.1–00:53.1 | Each key-value head is shared by two query heads. | Dense 8:8 lines collapse toward the grouped layout; pairing arcs land. | Section07 |
| `s09.13` 00:53.3–00:55.3 | This layout is grouped-query attention. | The GQA label stamps onto the diagram. | Section07 |
| `s09.14` 00:55.5–01:00.5 | Four key-value heads halve the cache against eight, while the math of attention stays the same. | MHA vs GQA stacks; the GQA stack halves in height; divide-by-2 arrow. | Section07 |
| `s09.15` 01:00.7–01:06.1 | The attention operation is unchanged, though a smaller key-value set can still change model quality. | Query fan-out stays full; only the KV stacks shorten. | Section07 |

## 10 · The gated feed-forward

`sec10_swiglu` · 10 · SWIGLU · 16 cues · 69.8 s · build **renumber**

**Screen.** Block diagram x (1024) to W_gate / W_up (4096) to SiLU to element-wise product to W_down to out (1024); live plot of SiLU(x) = x * sigmoid(x) against dashed ReLU; two rows resolving into their product; open vs closed gate features; parameter comparison bar (FFN vs attention); formula FFN(x) = W_down( SiLU(W_gate x) * W_up x ).

**Old code.** Section08 — renumber only. Old cues: Section08 s08.1–s08.7 (7).

| cue | says | screen | old code |
|---|---|---|---|
| `s10.1` 00:00.0–00:04.1 | Two sublayers sit in every block; attention is the first. | Section header; the block outline shows two slots. | Section08 |
| `s10.2` 00:04.3–00:08.0 | The second is a feed-forward network applied to each token on its own. | The lower slot fills with the x to out diagram. | Section08 |
| `s10.3` 00:08.2–00:13.1 | Attention mixes tokens; the feed-forward works on each vector in isolation. | A split view: arcs between tokens above, single-token paths below. | Section08 |
| `s10.4` 00:13.3–00:19.0 | It widens the vector from one thousand twenty-four to four thousand ninety-six, then compresses it back. | The expansion layer widens to 4096, then narrows to 1024. | Section08 |
| `s10.5` 00:19.2–00:23.4 | The wide middle gives the block room to transform features between the two widths. | Feature marks scatter through the wide layer as the vector crosses. | Section08 |
| `s10.6` 00:23.6–00:30.2 | Count the parameters and the feed-forward dominates: over twelve million per block against about three million for attention. | Parameter bars compare the two sublayers side by side. | Section08 |
| `s10.7` 00:30.4–00:33.0 | LLaMA gates this expansion with SwiGLU. | The SwiGLU label lands on the block. | Section08 |
| `s10.8` 00:33.2–00:37.5 | The wide layer splits into two parallel paths: a gate path and an up path. | The wide layer forks into a gate path and an up path. | Section08 |
| `s10.9` 00:37.7–00:39.9 | Both paths start from the same input vector. | The fork point glows once for both branches. | Section08 |
| `s10.10` 00:40.1–00:44.7 | The gate path passes through the SiLU activation, which is x times its own sigmoid. | The SiLU curve draws against a dashed ReLU; formula labels in. | Section08 |
| `s10.11` 00:44.9–00:50.9 | Near zero, SiLU passes about half the input; large positive inputs pass nearly unchanged. | Markers ride the SiLU curve at near-zero and deep-negative x. | Section08 |
| `s10.12` 00:51.1–00:53.9 | Element by element, the gate path scales the up path. | The product node pulses; the gate and up rows resolve into their product. | Section08 |
| `s10.13` 00:54.1–00:58.0 | The gate can shrink a feature, pass it unchanged, or amplify it. | Product row shows strong and weak features side by side. | Section08 |
| `s10.14` 00:58.2–01:00.9 | Only the product of the two paths flows onward. | A single output row leaves the product node; both inputs fade. | Section08 |
| `s10.15` 01:01.1–01:06.0 | Multiplicative gating lets one path decide how much of the other to keep. | The gate dial opens and closes while the up row scales with it. | Section08 |
| `s10.16` 01:06.2–01:09.8 | A final matrix projects the result back into the residual stream. | W_down closes the block; the formula writes in. | Section08 |

## 11 · Unembedding and autoregression

`sec11_loop` · 11 · THE GENERATION LOOP · 13 cues · 61.9 s · build **split**

**Screen.** Compact 16-block stack + final RMSNorm + W_U; logits row to softmax to probability bars; predicted token appended with a loop arrow; two equal-size table cards (embedding and unembedding, ~131M each); KV cache reuse note; paragraph typing out.

**Old code.** Section09 — unembedding and generation loop. Old cues: Section09 s09.1–s09.8 (8).

| cue | says | screen | old code |
|---|---|---|---|
| `s11.1` 00:00.0–00:04.2 | After sixteen blocks, one last RMSNorm settles the stream. | Compact 16-block stack; the final RMSNorm chip lands. | Section09 |
| `s11.2` 00:04.4–00:10.3 | A separate, untied matrix scores all one hundred twenty-eight thousand two hundred fifty-six candidates. | W_U chip; the logits row fills across the vocabulary ruler. | Section09 |
| `s11.3` 00:10.5–00:15.8 | Each score, called a logit, is one dot product between the final vector and one candidate row. | One logit cell expands to show its dot product. | Section09 |
| `s11.4` 00:16.0–00:22.6 | This unembedding table is as large as the embedding table: about one hundred thirty-one million parameters in each. | Two equal-size table cards face each other across the stack. | Section09 |
| `s11.5` 00:22.8–00:26.0 | Together, the two tables hold just over half of the model. | A pie bar splits: tables just past halfway, blocks filling the rest. | Section09 |
| `s11.6` 00:26.2–00:29.2 | Large logits mean the candidate fits the context well. | The tallest logit cell rises above its neighbors. | Section09 |
| `s11.7` 00:29.4–00:34.0 | A softmax turns the logits into probabilities over the whole vocabulary. | Logits morph into probability bars. | Section09 |
| `s11.8` 00:34.2–00:38.6 | Sampling picks one token, and that token is appended to the context. | The predicted chip appends to the context row. | Section09 |
| `s11.9` 00:38.8–00:41.2 | The full network then runs again for the next one. | The loop arrow strokes above the token row. | Section09 |
| `s11.10` 00:41.4–00:45.6 | One forward pass per generated token; that is the generation loop. | The loop arrow completes its stroke and holds. | Section09 |
| `s11.11` 00:45.8–00:51.8 | A KV cache would keep past keys and values, so each new pass computes only the newest token. | Cache bars stay lit while a single new token runs the stack. | Section09 |
| `s11.12` 00:52.0–00:57.4 | Without one, every step re-reads the whole sequence; this code does exactly that. | A ghost rerun of the full sequence flashes, then fades. | Section09 |
| `s11.13` 00:57.6–01:01.9 | What began as one function on a table of numbers now writes a paragraph. | The paragraph completes under the loop arrow. | Section09 |

## 12 · The path end to end

`sec12_recap` · 12 · THE WHOLE PICTURE · 9 cues · 43.5 s · build **split**

**Screen.** One clean end-to-end diagram: text to tokenizer to embedding to 16 blocks (RMSNorm, RoPE attention in groups, SwiGLU) to final norm to unembedding to sampling loop; recap card; end card with spec line.

**Old code.** Section09 — recap chain and end card. Old cues: Section09 s09.1–s09.8 (8).

| cue | says | screen | old code |
|---|---|---|---|
| `s12.1` 00:00.0–00:03.6 | Walk the whole path once more, from text to a sampled token. | Section header; the end-to-end diagram draws in one sweep. | Section09 |
| `s12.2` 00:03.8–00:07.7 | The tokenizer cuts the sentence into six pieces with integer IDs. | Token chips and IDs reappear at the diagram's start. | Section09 |
| `s12.3` 00:07.9–00:13.1 | The embedding table turns each ID into a vector of one thousand twenty-four numbers. | The matrix pulls rows out into vectors. | Section09 |
| `s12.4` 00:13.3–00:16.9 | The residual stream carries those vectors through sixteen blocks. | The stream band runs the length of the diagram. | Section09 |
| `s12.5` 00:17.1–00:24.5 | In each block, RMSNorm controls scale, attention mixes context, and the gated feed-forward rewrites features. | One block opens to show its three stages in sequence. | Section09 recap + Section04 pre-norm mention |
| `s12.6` 00:24.7–00:28.4 | Rotary embeddings keep the order of the tokens visible through all of it. | The RoPE dial turns quietly beside the block. | Section09 |
| `s12.7` 00:28.6–00:33.8 | Grouped-query attention keeps the cache affordable; QK-norm keeps the logits stable. | The GQA stacks and the QK-norm chip flash in place. | Section09 |
| `s12.8` 00:34.0–00:38.3 | The unembedding table scores every candidate, and sampling takes one. | The probability bars resolve to a single chosen chip. | Section09 |
| `s12.9` 00:38.5–00:43.5 | About five hundred fifteen million parameters, and one function from tokens to probabilities. | End card; spec line; fade out. | Section09 |

## Notes

- `sec09_gqa` `s09.7` (7.5 s) and `sec12_recap` `s12.5` (7.4 s) carry a 7-second line. Give each more than one beat.
- `render_scenes.py` still ends with `ALL_SECTIONS = [Section01, …, Section09]`. It needs twelve entries in narration order once the re-cue lands.
- The tree cannot render past `Section01` today. See `AUDIT.md` finding F1.
- Publish nothing until `make verify` passes on the new master. The current MP4 is the old 63-sentence cut.
