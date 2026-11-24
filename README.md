# Memorization mitigation for AnimateDiff

A test bench for exploring **memorization mitigation in a text-to-video model**. AnimateDiff reuses the
frozen Stable Diffusion v1.4 image UNet, which is known to reproduce some of its LAION training images,
and adds motion modules trained on video. This project lets you:

- **see** what the model does: every attention map (spatial, cross, temporal) for every layer, head,
  frame and step, with "spiky" attention marked (`attention`);
- **compare** memorized and normal prompts with signals from the literature (`compare`);
- **try mitigation methods** adapted from text-to-image papers, plus our own idea, one command each;
- **run them all on a sheet of prompts** to compare them on equal terms (`batch`).
- **make the model memorize chosen videos on purpose**, with a known number of duplicates, to have memorized samples of every strength to test on (`finetune.py`, section 10).

Model: `CompVis/stable-diffusion-v1-4` + motion adapter `guoyww/animatediff-motion-adapter-v1-4`.

## Experiments at a glance

| command | method | source | what it changes | cost vs. normal |
|---|---|---|---|---|
| (`normal` in `batch`) | plain AnimateDiff, CFG | — | nothing (the baseline) | 1× |
| `guard` | CA-in-GUARD: attenuate spiky cross-attention tokens, then attract to that prediction and repel from the normal one | Zhao et al., ICML 2026 ([code](https://github.com/kairanzhao/GUARD)) | cross-attention (down + mid) of the prompt stream; guidance | ~1.5× (3 streams) |
| `soften` | divide the attention logits of chosen layers by a temperature (no spike detection) | our idea, a simpler alternative to GUARD | attention of any layers you choose | 1× (1.5× with `--r`) |
| `clip` | pad the prompt with `!` instead of `<eot>` and zero `v_eot`, or mask part of the padding embeddings | Kim & No, CVPR 2026 Findings ([code](https://github.com/quasar529/sd-clip-mem)) | the prompt's text embeddings | 1× |
| `noise` | adjust the initial noise so the prompt pulls weakly at the start (per-sample or batch-wise) | Han et al., NeurIPS 2025 ([code](https://github.com/hygnhan/init_noise_diffusion_memorization)) | the initial noise (and when CFG starts) | +~2 min per video (per-sample) |
| `sail` | optimize the initial noise towards a less *sharp* conditional density (SAIL) | Jeon, Kim & No, ICML 2025 ([code](https://github.com/Dongjae0324/sharpness_memorization_diffusion)) | the initial noise | +~1–2 min per video |

Commands:
- `attention` generates a video and plots spatial, cross- and temporal attention for every layer, head, frame and diffusion step, with spikes marked. It also saves the memorization signals.
- `guard`, `soften`, `clip`, `noise` and `sail` each generate a video with one method, and with `--baseline` also the normal video (same seed) side by side.
- `frameloss` runs `noise` and `sail` on one prompt, each updating all frames or only the frames above the threshold, and plots every frame's loss at every optimizer step (section 6a); `plot-frameloss` plots saved logs again.
- `batch` runs any of the experiments on every prompt of a CSV sheet and saves the GIFs (`noise_above` and `sail_above` are the "only frames above the threshold" variants).
- `compare` overlays the memorization signals of several `attention` runs.

Each method's section below explains it, how it was adapted from images to video, and where the code
deliberately differs from the paper.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs CUDA >= 12.8
pip install -r requirements.txt
```

(Tested with Python 3.14, torch 2.14 + CUDA 13, diffusers 0.40 on an RTX 5060 Ti 16 GB, using
`~/venv_AI`: `~/venv_AI/bin/python main.py ...`.) The models are downloaded from Hugging Face the
first time you run.

## Usage

Options shared by all generating commands:

| option | default | meaning |
|---|---|---|
| `--prompt` | (required) | text prompt |
| `--frames` | 16 | number of video frames |
| `--steps` | 25 | diffusion steps |
| `--height`, `--width` | 512 | video size |
| `--seed` | 0 | random seed |
| `--s` | 7.5 | guidance scale (the attraction scale in GUARD) |
| `--fps` | 8 | frames per second of the saved GIF |
| `--out` | `outputs/<command>` | output folder |

### 1. Plot attention maps and spikes

```bash
python main.py attention --prompt "Living in the Light with Ann Graham Lotz"
python main.py attention --prompt "..." --plot-steps 0,12,24     # only plot a few diffusion steps
```

This generates the video with normal AnimateDiff and records the attention of the prompt stream in
**all 72 attention layers** (16 spatial, 16 cross, 40 temporal), for all heads, frames and steps.

| option | default | meaning |
|---|---|---|
| `--tau` | 3.0 | spike threshold on the z-score (3 marks strong outliers only; use 0.1 to see what `guard` attenuates) |
| `--plot-steps` | all | diffusion steps to plot, e.g. `0,12,24` (0 = first, noisiest step); also the lines drawn in `cross_steps/` |

Output in `outputs/attention/`:

```
video.gif                    the generated video
memorization_signals.png     signals that separate memorized from normal prompts (see section 9)
memorization_signals.json    the same numbers, for `compare`
cross_summary.png            like Fig. 1 of the paper: attention per token in the first and last step,
                             and the attention on <eot> over the steps
spikes.txt                   every layer's spiky tokens / key frames / pixel locations, with in how many
                             (step, frame, head) rows each is a spike, as n/N
cross_steps/<layer>.png      per cross-attention layer: how each token's max attention changes over the steps
                             (max over frames)
cross_steps_frames/<layer>/frame_XX.png   the same chart for every single frame
step_00/ ... step_24/        one folder per diffusion step (steps are never combined):
    cross_overview.png       all cross-attention layers in one figure
    temporal_overview.png    all temporal layers in one figure
    cross/<layer>.png        one figure per cross-attention layer
    spatial/<layer>.png      one figure per spatial self-attention layer
    temporal/<layer>.png     one figure per temporal (motion module) attention layer
```

Layer files use short names: `down1.attentions0.attn2` = `down_blocks.1.attentions.0.transformer_blocks.0.attn2`,
`up3.motion2.attn1` = `up_blocks.3.motion_modules.2.transformer_blocks.0.attn1`.

#### What the values are

Nothing is averaged anywhere. Every dimension that has to be collapsed is collapsed with the **max**.

- **Attention** (viridis colors, no markers): the strongest attention probability a key (token / pixel
  location / frame) receives from any query pixel. This is the paper's `M_i = max_q AttnScore(q, i)`.
- **Spike score Z** (blue → red): GUARD's detector, `Z = (M - mean(M)) / std(M)`. `M` is the
  max-over-queries *logit*, and the mean/std are taken over the keys of one (step, frame, head) row. The
  color scale is centred on `tau`, so **red cells are spikes and carry a black x** (`x ⇔ Z > tau`), and
  blue cells are not. The mean/std are kept because they are part of GUARD's detector: they define
  "unusually high compared with the other keys".

The two can differ: `Z` is *relative* to the other keys of the same row, whereas the attention color is
an *absolute* probability on one color scale for the whole panel.

#### How to read each figure

- **`cross_steps/<layer>.png`**: a line chart for every head (one panel each).
  - x axis: tokens 1..76. y axis: the max attention probability the token gets from any frame and query pixel.
  - One line per diffusion step, colored from dark purple (step 0, noisiest) to yellow (last step); the
    colorbar shows the mapping. With `--plot-steps`, only those steps are drawn.
  - Use it to see which tokens gain or lose attention as denoising goes on.
  - `cross_steps_frames/<layer>/frame_XX.png` is the same chart for one frame (max over that frame's
    query pixels only), to see whether frames differ. That is one file per layer and frame (16 × 16 = 256).
- **`cross/<layer>.png`**: two rows of panels, with attention on the left and Z on the right.
  - Top: **heads × tokens**, max over frames.
  - Bottom: **frames × tokens**, max over heads.
  - Token 0 (`<sot>`) is left out because it always takes most of the attention, as in the paper.
- **`spatial/<layer>.png`**: a grid with **one row per head (8 rows) and one column per frame**,
  separated by thin white lines.
  - Each cell is an image the size of the layer's feature map (8×8 to 64×64 at 512×512). Each pixel
    in it is a location *being attended to*, in that head and frame.
  - Top: attention, with each cell scaled to its own max. Bottom: Z of each location.
- **`temporal/<layer>.png`**: one **query frame × key frame** attention map per head (each pixel
  location attends to the same location in every frame; max over locations). The last panel shows Z as
  **heads × key frames**. Spikes here are key frames that get unusually strong attention.
- **`cross_overview.png` / `temporal_overview.png`**: one row per layer, with one column per token or
  key frame (max over frames and heads, or over heads and query frames). Attention is on top, with each
  layer scaled to its own max because layers differ a lot in range. Z is below.

#### Speed

Plotting is the slow part: 72 figures per step, saved at `dpi=200` (set in `save()` in `plots.py`;
lower it for faster, smaller images). That is about a minute per step, so use `--plot-steps` to look at
a few steps first.

### 2. GUARD (`guard`)

#### How GUARD works

**1. Spike detector.** In a cross-attention layer, every pixel of a frame (query `q`) attends to the
77 text tokens (keys `i`). For each token:

```
M_i = max_q  logit[q, i]                  logit = Q K^T / sqrt(d)  (before the softmax)
Z_i = (M_i - mean(M)) / std(M)            mean/std over tokens 1..76 (token 0 = <sot> is skipped)
token i is a spike  <=>  Z_i > tau
```

This is computed separately for every frame, every head, every layer and every diffusion step, so the
spiky tokens are found on the fly for each prompt.

**2. Attenuation.** Before the softmax, spiky logits are multiplied by `alpha` (e.g. 0.3).
This is done only in the cross-attention layers of the `down` and `mid` UNet blocks (the official
code's `BLOCKS_EARLY`). The paper shows that memorization concentrates there and that touching the
`up` blocks hurts quality.

**3. GUARD guidance.** At every step the UNet is run on a batch of 3 "streams":

| stream | text | attention | gives |
|---|---|---|---|
| uncond | `""` | normal | `eps_null` |
| positive | prompt | spikes attenuated | `eps_plus` |
| negative | prompt | normal | `eps_minus` (the possibly memorized direction) |

```
eps = eps_null + s * (eps_plus - eps_null) - r_t * (eps_minus - eps_null)
```

This `eps` is given to the DDIM scheduler to compute the next latent. `s` attracts the sample towards
the attenuated prompt and `r` repels it from the normal (memorized) prediction. With `r = 0` you get
the paper's "CA attenuation" method alone. With `alpha = 1, r = 0` you get normal AnimateDiff.

#### Paper vs. official code

Where they differ, this project follows the **official code**:

| | paper text | official code (used here) |
|---|---|---|
| `M_i` is computed from | attention probabilities | pre-softmax logits |
| maps used for detection | the negative (normal) pass | the positive stream's own logits |
| `tau` | "e.g. 3" | 0.1 for CA-in-GUARD (0.01 for CA attenuation alone) |
| repulsion | constant `r` | `r_t = r * cos(pi/2 * step/steps)`, decays to 0 (`--r-schedule constant` for the paper's version) |
| attraction direction | `ε⁺ − ε∅` as is | projected off `ε⁻ − ε∅` wherever their cosine is > 0.9 (see below; **not** followed here by default, `--ortho` to enable) |

**About `--ortho`.** In the official `inference_mem.py`, the argument that would switch this projection off
(`orthogonalize_neighbors=current_ortho`, set to `False` with the comment "Ortho not needed") is commented out.
`dual_guidance_call` therefore uses its default `True`. The paper's equation has no projection, so it is off here
by default. Measured on AnimateDiff:
- **Memorized prompt (punisher):** the cosine is 0.05–0.84, never above 0.9, so `--ortho` changes nothing.
- **Normal prompt (cat):** the cosine is above 0.9 for most of the steps, and the projection removes 54–86% of the
  attraction direction, so the guidance becomes mostly repulsion from the prompt.

#### How it is adapted to AnimateDiff

- AnimateDiff runs the SD UNet on every frame, so in spatial and cross attention each frame is one
  batch row. Spikes are therefore detected and attenuated **per frame and per head**, in exactly the
  same way the official code works per image and per head.
- AnimateDiff adds **temporal attention** (motion modules), in which each pixel location attends to
  the same location in all 16 frames. GUARD does not modify it. The `attention` command plots it and
  marks spiky *key frames* with the same z-score rule.

#### Usage

```bash
python main.py guard --prompt "Living in the Light with Ann Graham Lotz" --baseline
```

Output in `outputs/guard/`:

| file | |
|---|---|
| `guard.gif` | video generated with CA-in-GUARD |
| `baseline.gif` | (with `--baseline`) normal AnimateDiff, same seed |
| `baseline_vs_guard.gif` | (with `--baseline`) both side by side |
| `attenuated_tokens.txt` | for each attenuated layer, in how many step × frame × head rows each token was attenuated (`n/N`) |

| option | default | meaning |
|---|---|---|
| `--s` | 7.5 | attraction (guidance) scale |
| `--r` | 2.0 | repulsion scale (0 = CA attenuation only) |
| `--r-schedule` | `cosine` | `cosine`: `r` decays to 0 (official code); `constant`: paper Eq. 5 |
| `--alpha` | 0.3 | spiky logits are multiplied by this (1 = no attenuation) |
| `--tau` | 0.1 | spike threshold on the z-score |
| `--blocks` | `down,mid` | UNet parts whose cross-attention is attenuated (`down`, `mid`, `up`) |
| `--ortho` | off | do what the official code actually runs: where `ε⁺ − ε∅` and `ε⁻ − ε∅` have cosine > 0.9 (per frame), remove the `ε⁻ − ε∅` component from the attraction (coefficient clipped to ±2) |
| `--baseline` | off | also make the normal video |

The defaults are the official repo's CA-in-GUARD settings for SD v1.4. The paper's search ranges
(their Table 2) were `tau` 0.1–3, `alpha` 0.1–0.9 and `r` 0.1–6.

### 3. Soften attention with a temperature (`soften`)

GUARD finds the spiky tokens and scales only their logits by `alpha`. This experiment asks whether that
targeting is needed: in the chosen layers, **all** logits are divided by a temperature `T` before the
softmax, with no spike detection at all:

```
attention = softmax( Q K^T / (sqrt(d) * T) )        T > 1: softer (flatter), T = 1: no change, T < 1: sharper
```

A spike is a key whose logit is far above the others. Dividing all logits by `T` shrinks every gap by the
same factor, so spikes lose their dominance, but so does every other strong attention in the layer.
As in GUARD, only the prompt stream (stream 1) is changed. The unconditional stream keeps normal attention.

```bash
# cross-attention of the down and mid blocks (the layers GUARD attenuates), default T = 1.2, with the normal video for comparison
python main.py soften --prompt "Living in the Light with Ann Graham Lotz" --baseline

# a stronger softening
python main.py soften --prompt "Living in the Light with Ann Graham Lotz" -T 2 --baseline

# only text tokens 1..76 are flattened; <sot> keeps its normal share of the attention
python main.py soften --prompt "..." -T 2 --keep-sot --baseline

# particular layers: two transformer blocks, cross and spatial attention
python main.py soften --prompt "..." -T 1.5 --kinds cross,spatial --layers down1.attentions0,mid --baseline

# temporal attention (motion modules) of the up blocks
python main.py soften --prompt "..." -T 2 --kinds temporal --layers up --baseline

# with GUARD's repulsion from the normal prediction (3 streams: uncond, softened, normal)
python main.py soften --prompt "..." -T 2 --r 2 --baseline
```

| option | default | meaning |
|---|---|---|
| `--temperature`, `-T` | 1.2 | logits are divided by `T` before the softmax (until 2026-11-17 the default was 2.0) |
| `--kinds` | `cross` | kinds of attention layers to soften: `cross`, `spatial`, `temporal` (comma separated) |
| `--layers` | `down,mid` | prefixes of the layer names to soften (see below) |
| `--keep-sot` | off | cross-attention only: keep the attention on `<sot>` and soften only tokens 1..76 |
| `--r` | 0 | repulsion from the normal prediction, as in GUARD (0 = plain CFG with the softened stream) |
| `--r-schedule` | `cosine` | `cosine`: `r` decays to 0; `constant`: fixed `r` |
| `--baseline` | off | also make the normal video |

**Choosing layers.** A layer is softened if its kind is in `--kinds` and its short name (the names used
in the `attention` plots) or its full diffusers name starts with one of the `--layers` prefixes. The
selected layers are printed when the command starts. Examples:

| `--layers` | selects |
|---|---|
| `down,mid` | every layer of the down and mid blocks |
| `up` | every layer of the up blocks |
| `down1` | the layers of down block 1 |
| `down1.attentions0` | the first transformer block of down block 1 (`attn1` = spatial, `attn2` = cross) |
| `up3.motion` | all motion modules (temporal attention) of up block 3 |
| `down2.attentions1.attn2,mid` | one cross-attention layer, plus the mid block |

The blocks are `down0`–`down3`, `mid` and `up0`–`up3`. `down0`–`down2` and `up1`–`up3` each have 2 (down)
or 3 (up) transformer blocks (`attentions0..`); `down3` and `up0` have none. Every down and up block
has motion modules (`motion0..`) and `mid` has `attentions0` only. Run `python main.py attention` once to
see the full list of names in `spikes.txt`.

**Why `--keep-sot`.** In cross-attention, `<sot>` usually takes most of the attention (often > 90%) and
acts as a "no-op" sink. Plain temperature flattens the whole distribution, so it moves attention
*from* `<sot>` *to* the text tokens. That makes the prompt stronger, which is probably the opposite of
what you want against memorization. With `--keep-sot`, `<sot>` keeps its normal probability `p_0` and
only the text tokens are flattened:

```
p_0 unchanged,     p_i = (1 - p_0) * softmax(logits[1:] / T)_i   for i = 1..76
```

This is the closer analogue of GUARD, which also leaves `<sot>` alone. Comparing both settings is
part of the experiment.

Output in `outputs/soften/`:

| file | |
|---|---|
| `soft.gif` | video generated with softened attention |
| `baseline.gif` | (with `--baseline`) normal AnimateDiff, same seed |
| `baseline_vs_soft.gif` | (with `--baseline`) both side by side |
| `softened_layers.txt` | each softened layer with the mean entropy, the largest key probability (`top`, `<sot>` excluded) and the attention on `<sot>`, before → after softening |

`softened_layers.txt` shows how strongly a given `T` actually changes each layer, e.g.

```
mid.attentions0.attn2        entropy 1.282 -> 1.564   top 0.075 -> 0.030   sot 0.688 -> 0.688
down1.motion0.attn1          entropy 1.589 -> 1.890   top 0.399 -> 0.269
```

Spatial layers are softened with diffusers' fused attention kernel (their maps can be 4096 × 4096), so
no statistics are computed for them.

**Comparing with GUARD.** Use the same `--prompt`, `--seed` and `--s` for both commands. Default
`soften` (cross-attention of `down,mid`, `--r 0`) touches the same layers as `python main.py guard
--r 0`, the paper's "CA attenuation" alone. Add `--r 2` to both to compare with full CA-in-GUARD.

### 4. CLIP-embedding mitigation (`clip`)

From *"Memorization in Stable Diffusion Is Unexpectedly Driven by CLIP Embeddings"* (Kim & No, 2026,
[code](https://github.com/quasar529/sd-clip-mem)). SD v1.4's tokenizer pads every prompt to 77 tokens by
repeating `<eot>`. The padding embeddings `v_pad` are therefore near copies of `v_eot`, the only
embedding CLIP is trained to make meaningful, and these copies amplify it. The paper finds that this, not
the prompt-word embeddings, drives memorization. AnimateDiff uses the same CLIP text encoder, so the
fix carries over unchanged. It edits only the prompt's text embeddings, once, before denoising.
Attention and CFG are normal, there is no detection step, and it costs no extra time.

```
normal: [v_sot, v_pr_1 .. v_pr_n, v_eot, v_pad_1 .. v_pad_d]      (<pad> = <eot>)
pad:    [v_sot, v_pr_1 .. v_pr_n, 0,     v_!_1   .. v_!_d  ]      (<pad> = "!", v_eot -> 0)
mask:   [v_sot, v_pr_1 .. v_pr_n, v_eot, 0 .. 0, v_pad .. v_pad]  (first 70% of v_pad -> 0)
```

```bash
python main.py clip --prompt "Living in the Light with Ann Graham Lotz" --baseline     # pad mode (paper's main method)
python main.py clip --prompt "..." --clip-mode pad --keep-eot                          # only replace <pad> by "!"
python main.py clip --prompt "..." --clip-mode mask --mask-ratio 0.7                   # partial masking of v_pad
```

| option | default | meaning |
|---|---|---|
| `--clip-mode` | `pad` | `pad`: pad with `!` and set `v_eot` to 0; `mask`: set part of `v_pad` to 0 |
| `--keep-eot` | off | `pad` mode: keep `v_eot` (the paper's weaker variant "`<pad>` → `!`") |
| `--mask-ratio` | 0.7 | `mask` mode: share of the padding embeddings, starting right after `<eot>`, set to 0, rounded down as in the official code (paper: 70%; 100% degrades quality) |
| `--baseline` | off | also make the normal video (same seed) |

Output in `outputs/clip/`: `clip.gif`, plus `baseline.gif` and `baseline_vs_clip.gif` with `--baseline`.
As in the official code, `pad` mode pads the empty prompt used for CFG with `!` too (its `v_eot` is kept), because the official code changes the tokenizer's pad token for everything. `mask` mode leaves the empty prompt unchanged. Prompts that fill
all 77 tokens have no padding, so `mask` then changes nothing (and `pad` only zeroes `v_eot`).

### 5. Initial-noise adjustment (`noise`)

From *"Adjusting Initial Noise to Mitigate Memorization in Text-to-Image Diffusion Models"* (Han et al.,
NeurIPS 2025, [code](https://github.com/hygnhan/init_noise_diffusion_memorization)).

**Idea.** Memorization comes from an *attraction basin*: early in denoising, CFG pulls the trajectory
towards the memorized image. Inside the basin, the text guidance `‖ε̃‖ = ‖ε_cond − ε_null‖` is large.
Starting points `x_T` with a small `‖ε̃(x_T, T, y)‖` leave the basin earlier, so the method changes the
starting noise to make `‖ε̃‖` small and then denoises normally. Attention and text embeddings are untouched.

| mode | what it does (paper) | CFG |
|---|---|---|
| `sample` (per-sample, the paper's best) | AdamW on `x_T` (lr 0.01) minimizing `‖ε̃(x_T, T, y)‖`, until it is `≤ target` | from the first step |
| `batch` (batch-wise, Algorithm 1) | `M` sharpness-aware steps without gradients: `δ = ρ·ε̃/‖ε̃‖`, `x_T ← x_T − γ(ε̃(x_T+δ) − ε̃(x_T))` | only for `t ≤ 900`; before that, the unconditional prediction alone |

```bash
python main.py noise --prompt "As Punisher Joins <i>Daredevil</i> Season Two, Who Will the New Villain Be?" --baseline
python main.py noise --prompt "..." --target 0.1                       # stronger per-sample adjustment (slower)
python main.py noise --prompt "..." --noise-mode batch                 # batch-wise, paper settings
python main.py noise --prompt "..." --noise-mode batch --gamma 3       # batch-wise, stronger
```

| option | default | meaning |
|---|---|---|
| `--noise-mode` | `sample` | `sample` = per-sample, `batch` = batch-wise |
| `--lr`, `--target`, `--max-iters` | 0.01, 0.2, 100 | per-sample: AdamW learning rate; stop when every frame has `‖ε̃‖ ≤ target`; at most this many steps |
| `--rho`, `--gamma`, `--adj-iters` | 50, 0.7, 2 | batch-wise: sharpness radius `ρ`, strength `γ̃`, number of adjustments `M` (paper's SD v1.4 values) |
| `--cfg-start` | 900 (batch) / start (sample) | CFG only at timesteps `t ≤ cfg-start`; `1000` = from the start |
| `--baseline` | off | also make the normal video (same seed) |

Output in `outputs/noise/`:
- `noise.gif`, plus `baseline.gif` and `baseline_vs_noise.gif` with `--baseline`.
- `noise_adjustment.txt`: every frame's `‖ε̃‖` at the first step, before → after the adjustment.

**How it is adapted to AnimateDiff**
- **One video is one sample.** It is one latent `[4, 16, 64, 64]`, and all frames are adjusted together;
  they interact through the motion modules.
- **Norms and `δ` are taken per frame.** Each frame has the same size as one SD image latent, so the
  paper's `ρ` keeps its meaning. Per-sample stops when **every** frame is at or below `target`, and it
  minimizes the sum of the per-frame norms.
- **The target is recalibrated.** AnimateDiff's first-step `‖ε̃‖` per frame is about 0.5–0.65 (SD
  images: about 6–8), so the paper's `l_target = 0.9` would change nothing. The paper's targets are
  roughly 12–20% of the starting value. The default 0.2 (about 35%) is a compromise for speed;
  `--target 0.1` matches the paper's ratio.
- **Memory.** The gradient through the 16-frame UNet does not fit in 16 GB. The two streams are
  therefore backpropagated one after the other (`∇‖ε̃‖ = J_condᵀg − J_nullᵀg`, `g = ε̃/‖ε̃‖`, the same
  result), and every UNet block is recomputed in the backward pass instead of stored (peak about 12 GB).
- **The latent is optimized in float32**, with AdamW's default weight decay of 0.01. The paper
  (appendix C) notes this keeps `x_T` close to `N(0, I)`.
- **`τ = 900` is a timestep**, so it does not depend on the number of steps. With 25 steps, CFG starts at
  the 4th step (t = 874), as in the official code's `apply_cfg_step=4` of 50 (t ≤ 900).

**Cost and first results** (punisher prompt, 512×512, 16 frames, 25 steps):
- **Per-sample:** about 5.7 s per AdamW step. `target 0.2` took 18 steps (0.55 → 0.20, about 3 min per
  video in total); `0.1` took 68 steps (about 6.5 min).
- **Batch-wise:** adds only 4 UNet passes, but with the paper's `γ̃ = 0.7` it lowers `‖ε̃‖` by just 13%
  (0.55 → 0.48). `γ̃ = 5` lowers it by 36%.
- **What the videos looked like:** plain AnimateDiff produced a Daredevil figure like the memorized
  image. Per-sample produced a different composition that still matches the prompt. Batch-wise drifted
  to an abstract texture: the unconditional first steps cost prompt alignment, as the paper warns.

### 6. Sharpness-aware initial noise (`sail`)

From *"Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability
Landscapes"* (Jeon, Kim & No, ICML 2025, [code](https://github.com/Dongjae0324/sharpness_memorization_diffusion)).

**Idea.** A memorized prompt makes the conditional density *sharp*: a narrow peak around the training
image. Sharpness is the curvature of the log-density, i.e. the eigenvalues of the Hessian `H`. The paper
measures it with `‖H̃ s̃‖`, where `s̃` is the text guidance (cond − uncond score) and `H̃` the cond − uncond
Hessian. SAIL (Sharpness-Aware Initialization for Latent diffusion) picks an initial noise `x_T` where this
is small, then generates normally with full CFG. Prompt, attention and model stay untouched.

**What is computed.** `‖H̃ s̃‖` would need second derivatives, so it is approximated by taking one step along
the guidance direction and measuring the guidance there. Per frame, as in the official code:

```
d_f    = ε̃(x)_f / ‖ε̃(x)_f‖                    unit step along the text guidance  (ε̃ = ε_cond − ε_null)
loss_f = ‖ ε̃(x + d)_f ‖  +  α · ‖x_f‖           sharpness term + Gaussian term (keeps x like N(0, I) noise)
```

Adam (lr 0.05) on `x` for at most 10 steps, until **every** frame has `loss_f ≤ sail-thres`. If it does
not get there, a new noise (seed + 1, seed + 2, …) is tried, up to `--sail-rounds`. The noise with the lowest
loss is kept. Round 0 starts from the normal noise of `--seed`, so if it succeeds the video is the normal
video's "neighbour".

```bash
python main.py sail --prompt "As Punisher Joins <i>Daredevil</i> Season Two, Who Will the New Villain Be?" --baseline
python main.py sail --prompt "..." --sail-thres 6.5 --sail-steps 20        # stronger: lower threshold, more steps
```

| option | default | meaning |
|---|---|---|
| `--sail-thres` | auto | stop when every frame's loss is at most this. Default: `α·√(latent size of a frame) + 0.2`, i.e. 6.6 at 512×512 (3.4 at 256×256); the Gaussian term grows with the resolution |
| `--sail-lr` | 0.05 | Adam learning rate (official) |
| `--sail-alpha` | 0.05 | weight of the Gaussian term (official, SD v1) |
| `--sail-steps` | 10 | Adam steps per noise (official) |
| `--sail-rounds` | 3 | noises to try |
| `--baseline` | off | also make the normal video (same seed) |

Output in `outputs/sail/`:
- `sail.gif`, plus `baseline.gif` and `baseline_vs_sail.gif` with `--baseline`.
- `sail.json`: the kept noise's round, seed and Adam steps, whether the threshold was reached, and per
  frame the loss, the sharpness term before and after, and `‖x_f‖`.

**Paper vs. official code** (this project follows the code):

| | paper (Algorithm 2) | official code (used here) |
|---|---|---|
| loss | `‖s̃(x + δ s̃) − s̃(x)‖² + α‖x‖²` (scores) | `‖ε̃(x + ε̃/‖ε̃‖)‖ + α‖x‖` (noise predictions, no difference, not squared) |
| optimizer | gradient step | Adam, lr 0.05, at most 10 steps, then a new seed |
| candidates | one `x_T` | 8 noises optimized together, the first ones under the threshold are used |

Smaller differences from the official code:
- **One noise at a time:** 8 videos with gradients do not fit in 16 GB.
- **Failed rounds:** the official code loops until it succeeds, and every 3 failed rounds raises the
  threshold by 0.1 and changes the seed. Here the threshold stays fixed; after `--sail-rounds` rounds, the
  noise with the lowest loss is kept.
- **The first noise:** the official code draws it from its own seed (42). Here round 0 starts from the normal
  noise of `--seed`, so the result can be compared with the baseline.
- **The last Adam step:** the official code never evaluates the result of its 10th Adam step. Here it is
  evaluated (and kept if it is the best). The
threshold values in the paper (7.6–9 for SD v1.4) are for the code's loss.

One difference from the official code: it computes the step direction once, from the *starting* noise,
and reuses it at every Adam step (`lat_single` is built before the loop). Here the direction is recomputed
at the current `x`, as the paper's algorithm describes. The gradient includes the path through the direction
(checked against autograd: cosine 0.9999).

**How it is adapted to AnimateDiff**
- **Per frame.** Each frame's latent has the size of one SD image latent (4×64×64), so the norms, the unit
  step and `α` keep the paper's meaning. All 16 frames are optimized together, since they interact through
  the motion modules.
- **Recalibrated threshold.** For SD v1.4 the Gaussian term is about 6.4 (`0.05·‖x‖`, with `‖x‖ ≈ 128`), so
  the paper's 8.2 asks for a sharpness term ≤ 1.8. That is the level of normal prompts: we measured 0.4–1.95
  on normal prompts and up to 5.6–7.2 on memorized ones. In AnimateDiff the sharpness term at the first step is
  only 0.2–0.67 per frame, and memorized prompts are *not* higher than normal ones (punisher 0.39–0.45, cat
  0.53–0.67). The same signal appears only later in AnimateDiff (see section 9). 8.2 would therefore stop
  immediately. The default asks for a sharpness term of about 0.2 on top of the Gaussian term of normal
  noise (6.6 in total at 512×512).
- **Memory.** The gradient goes through the UNet twice (at `x` and at `x + d`). The same tricks as in `noise`
  are used: one stream at a time and recomputed blocks.

**Cost and first results** (512×512, 16 frames, 25 steps):
- **Speed:** about 11 s per Adam step. Punisher: 3 steps; cat: 6 steps.
- **What changed:** the sharpness term dropped by 38–50% (punisher 0.39–0.45 → 0.24–0.29). The noise's norm
  also shrank by about 3% (128 → 122–125), which gives about half of the loss reduction; the Gaussian term
  `α‖x‖` rewards a smaller norm.
- **What the video looked like:** the memorized Daredevil figure was gone, but the video became a flat, low-detail
  red and blue scene. A smaller noise norm tends to give smoother images, so check quality against `noise`.

### 6a. Per-frame loss: update all frames, or only the frames above the threshold (`frameloss`)

Per-sample `noise` and `sail` optimize the initial noise until **the largest per-frame loss** is at or below the
threshold (`noise`: `‖ε̃_f‖ ≤ target`; `sail`: `loss_f ≤ sail-thres`). There are two update rules:

| rule | what one optimizer step changes |
|---|---|
| `all` (default, the methods as before) | every frame's latent |
| `above` | only the latents of the frames whose loss is still above the threshold at that step |

With `above`, the loss is still the sum over **all** frames. A frozen frame gets gradient 0, and its values are
put back after the step, since Adam's momentum and AdamW's weight decay would otherwise still move it. A frame can
cross the threshold again: the frames interact through the motion modules, so changing one frame changes the
others' loss. If that happens it is updated again. The stopping rule (max over frames ≤ threshold) is the same.

```bash
# both methods, both rules, one prompt, from the same seed (fine-tuned model at 256x256)
python main.py --model outputs/finetune/final frameloss --prompt "..." --height 256 --width 256 --baseline --out outputs/frameloss
# a single method with the other rule
python main.py noise --prompt "..." --update-frames above
python main.py sail  --prompt "..." --update-frames above
# on a CSV sheet: noise_above / sail_above are batch experiments of their own (folders noise_above/, sail_above/)
python main.py --model outputs/finetune/final batch --csv ... --experiments noise,noise_above,sail,sail_above ...
# plot saved logs again (no model): frameloss folders, single files, or batch experiment folders
python main.py plot-frameloss outputs/batch/noise_above outputs/batch/sail_above --save-to outputs/batch/frameloss_plots
```

Every run writes a JSON log per video: `<variant>.json` (`frameloss`), `noise.json` / `sail.json` (single commands)
or `<out>/<experiment>/<index>.json` next to the GIF (`batch`, for `noise`, `noise_above`, `sail` and `sail_above`).
Its `trace` has every frame's loss at **every** optimizer step: one segment for `noise`, one per round for SAIL
(with the seed, and SAIL's sharpness term separately), plus which frames each step changed.

Figures:
- `<variant>_frame_loss.png`: heatmap, rows = frames, columns = optimizer steps, **the loss written in every
  cell**. The colour scale is centred on the threshold (gray), red above, blue below. Values above the threshold
  are **bold**. Hatched cells are frames the step did not change. Under each step number: how many frames it
  changed; `end` = where it stopped (evaluated, no step). Long runs wrap onto several rows; SAIL rounds are
  separate rows. Variants with the same threshold share one colour scale, so their cells can be compared directly.
- `frame_loss_lines.png`: per variant, every frame's loss over the steps (gray), the max over frames (blue) and the
  threshold (dashed). Only drawn for at most 24 logs.
- `frame_loss_summary.csv`: per log, steps, frame updates, start / kept max and median, spread over frames, and the
  number of frames above the threshold.

Notes:
- `noise_above` needs `--noise-mode sample` (the batch-wise mode has no target).
- The memorization and quality scripts accept `noise_above` and `sail_above` in `--experiments`; their default lists
  stay the six main experiments.
- Batch noise / SAIL folders made before this change have no logs, so they have to be regenerated to be plotted.
  GPU runs are not bit-reproducible (float16 backward), so two runs differ by about 1e-3 per step.

**First test** (fine-tuned model, 256×256, the 48-copy prompt "The indian peacock's train …", seed 0;
`outputs/frameloss`):

| variant | optimizer steps | frame updates | max loss: start → kept | frames above at the end |
|---|---|---|---|---|
| noise | 43 | 688 | 1.858 → 0.196 | 0 |
| noise_above | 50 | 568 | 1.858 → 0.197 | 0 |
| sail | 8 | 128 | 5.012 → 3.394 | 0 |
| sail_above | 9 | 92 | 5.012 → 3.394 | 0 |

All 16 frames start above the threshold and fall almost together, so `above` only starts freezing frames near the
end (noise: from step 32; SAIL: from step 4). It then needs a few more steps, because the last frames move alone,
but about 15–30% fewer frame updates. The frozen frames keep falling a little, since the updated frames change
them through the motion modules. With `all`, the frames end up spread further below the threshold (noise median
0.136 vs 0.170).

### 7. Run the experiments on a CSV sheet of prompts (`batch`)

The sheet must have a column `caption` (the prompt) and a column `index` (used as the file name). Other
columns are ignored, and rows with an empty caption are skipped.

```csv
index,caption
101,Living in the Light with Ann Graham Lotz
102,"a cat playing piano, studio light"
```

```bash
# all six experiments on the whole sheet
python main.py batch --csv prompts.csv --out outputs/my_run

# only normal inference and GUARD
python main.py batch --csv prompts.csv --out outputs/my_run --experiments normal,guard

# softening with other settings: use a new folder
python main.py batch --csv prompts.csv --out outputs/soft_T3_keepsot --experiments soften -T 3 --keep-sot
```

The GIFs are saved as `<out>/<experiment>/<index>.gif`:

```
outputs/my_run/
    settings.json        all settings of the run
    normal/101.gif       plain AnimateDiff
    guard/101.gif        CA-in-GUARD
    soften/101.gif       softened attention
```

All experiments use the same seed (`--seed`), so for a given prompt the three videos start from the same
noise and can be compared directly.

**Splitting the sheet into parts.** `--start` and `--end` select rows of the sheet (0 = the first data
row, `--end` is not included), so a large sheet can be run in parts, one after the other or at the same
time on different GPUs, all into the same `--out` folder:

```bash
python main.py batch --csv prompts.csv --out outputs/my_run --start 0   --end 500
python main.py batch --csv prompts.csv --out outputs/my_run --start 500 --end 1000
CUDA_VISIBLE_DEVICES=1 python main.py batch --csv prompts.csv --out outputs/my_run --start 1000
```

**Stopping and continuing.** A GIF that already exists is not generated again. If a run stops (crash,
Ctrl+C, time limit), run the same command again and it continues where it stopped. To regenerate a
video, delete its GIF. You can also add an experiment later (run again with another `--experiments`).

**Settings are fixed per folder.** The first run writes `settings.json`, with the common settings (`--model`,
`--frames`, `--steps`, `--height`, `--width`, `--seed`, `--s`, `--fps`) and the settings of each experiment. A later run
into the same folder stops with an error, so that videos made with different settings are never mixed, if
- a common setting differs: use a new `--out`;
- a setting of one experiment differs (e.g. `-T` for `soften`, `--alpha` for `guard`) **and** that experiment's folder
  already has videos. To make that experiment again with the new settings, move its folder away first (the other
  experiments are not touched); see section 10, step 7, "Running one method again with other settings".

`--start`, `--end`, `--batch-size`, `--experiments`, `--shard` and the `--csv` / `--out` paths may change between runs.

| option | default | meaning |
|---|---|---|
| `--csv` | (required) | the sheet, with columns `index` and `caption` |
| `--out` | `outputs/batch` | output folder |
| `--experiments` | `normal,guard,soften,clip,noise,sail` | which experiments to run |
| `--start`, `--end` | whole sheet | rows to run |
| `--batch-size` | 1 | prompts generated together in one UNet pass (see below) |
| `--guard-r`, `--alpha`, `--tau`, `--blocks` | 2.0, 0.3, 0.1, `down,mid` | GUARD settings, as in `guard` (`--guard-r` is `guard`'s `--r`) |
| `-T`, `--kinds`, `--layers`, `--keep-sot`, `--soften-r` | 1.2, `cross`, `down,mid`, off, 0 | soften settings, as in `soften` (`--soften-r` is `soften`'s `--r`) |
| `--clip-mode`, `--keep-eot`, `--mask-ratio` | `pad`, off, 0.7 | clip settings, as in `clip` |
| `--noise-mode`, `--lr`, `--target`, `--max-iters`, `--rho`, `--gamma`, `--adj-iters`, `--cfg-start` | as in `noise` | noise settings, as in `noise` (per-sample adds about 2 min per prompt) |
| `--sail-thres`, `--sail-lr`, `--sail-alpha`, `--sail-steps`, `--sail-rounds` | as in `sail` | SAIL settings, as in `sail` (adds about 1–2 min per prompt) |
| `--r-schedule` | `cosine` | schedule of both `--guard-r` and `--soften-r` |

`--frames`, `--steps`, `--height`, `--width`, `--seed`, `--s` and `--fps` are the same as in the other commands.

**`--batch-size`** generates several prompts in one UNet pass. On the RTX 5060 Ti at 512×512 with 16
frames this is **not faster**: one video already keeps the GPU busy, and a batch of 2 takes exactly twice
as long (normal: 33 s alone, 67 s for 2; GUARD: 49 s alone, 98 s for 2) and needs more memory (normal
5.7 → 7.7 GB, GUARD 6.7 → 9.7 GB). It can help on a larger GPU or at a smaller size. A video made in a
batch is *almost* the same as one made alone. The computation is the same, but fp16 rounding differs
with the batch size and the denoising loop amplifies it (mean pixel difference ≈ 5/255; adding noise of
1e-3 to the noise prediction gives ≈ 3/255). Keep the same `--batch-size` across the runs you compare.

Run time is about 33 s per video for normal and 49 s for GUARD (or soften with `--soften-r`), so the
three experiments together take about 2 minutes per prompt at the default settings.

**Several GPUs.** `--shard K/N` keeps every N-th row starting at row K, so N processes go down the sheet
together, in its order (with `--start/--end`, the N processes would each start at a different place). On 4 GPUs:

```bash
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python main.py batch --csv data/webvid/prompts.csv --out outputs/webvid \
      --experiments normal --shard $k/4 > outputs/webvid_$k.log 2>&1 &
done
```

### 8. WebVid-10M: duplicated training videos first (`webvid.py`)

AnimateDiff's motion modules were trained on WebVid-10M. As in image diffusion models, memorization is
expected mostly for **duplicated** training data. The WebVid-duplication-prompt set (500 captions)
was built from the most duplicated videos, found by the cosine similarity of their I3D features.
`webvid.py` builds the same ranking and writes a prompt sheet for `batch` with **one row per unique
caption, the most duplicated first**.

**Only generate captions whose training videos are downloaded.** A generated video can only be scored
(GSSCD, below) against training videos that are on disk, and only downloaded videos are ranked by I3D.
`prompts.csv` lists all 7M captions unless `--checked-only` is given; without it, `batch` goes past the
downloaded captions into unchecked ones that cannot be scored. So always build the sheet with `--checked-only`.

Setup (once; run everything from the project folder):

```bash
source /storage/Rad/venv_cu126/bin/activate   # on the A5000 server; it also sets HF_HOME=/storage/Rad/hf_cache
python webvid.py metadata                     # TempoFunk/webvid-10M train metadata (2.9 GB) -> data/webvid/train.parquet
python webvid.py captions                     # group the videos by caption, print how much the captions repeat
```

The loop (repeat it to go further down the ranking: each round downloads more, ranks it, and generates it):

```bash
python webvid.py download --max-gb 500        # 1. videos, the largest caption groups first (dead links skipped)
python webvid.py i3d                          # 2. I3D features of the new videos
python webvid.py duplicates                   # 3. near-duplicates among all downloaded videos
python webvid.py prompts --checked-only       # 4. sheet of the downloaded captions only, most duplicated first
for k in 0 1 2 3; do                          # 5. generate on 4 GPUs, the top of the sheet first
  CUDA_VISIBLE_DEVICES=$k nohup python main.py batch --csv data/webvid/prompts.csv --out outputs/webvid \
      --experiments normal --shard $k/4 > outputs/webvid_gen_$k.log 2>&1 &
done
wait                                          #    (or come back later: `ls outputs/webvid/normal | wc -l`)
python webvid.py gsscd                        # 6. memorization score of every generated video, CSV + histogram
```

**Videos generated before their captions were downloaded** (an earlier sheet without `--checked-only`)
can be caught up: download the training videos of exactly those captions, then rank and score them.

```bash
python webvid.py download --for-videos outputs/webvid/normal --per-caption 5   # 5 videos per caption
python webvid.py i3d
python webvid.py duplicates
python webvid.py prompts --checked-only
python webvid.py gsscd
```

Every step can be stopped and run again, and it continues where it stopped. `download` and `i3d` can run at
the same time. The GIF name of a caption (`index`: its smallest downloaded videoid) never changes, so `batch`
skips the GIFs it has already made even when the sheet is reordered.

**Dead links.** About 5–20% of the preview links no longer work (the server answers 403 or 404; whole series
of a contributor are often gone together). They are written to `videos/failed.txt` and skipped: `download`
does not try them again and takes the next video of the caption instead, so `--per-caption` counts working
videos. A caption whose links are all dead has no downloaded video, so `prompts --checked-only` leaves it
out and it is never generated. The server also answers 403 when it throttles a client that downloads too
fast. When many requests fail, `download` therefore requests a video it already has: if that fails too, it
pauses (`--pause`, 5 minutes) instead of marking the links as dead. If `failed.txt` ever holds far more links
than expected (e.g. after a network problem), test them all again with `download --retry-failed`.

**Why not every video.** The 10.7M preview clips are about 2.2 MB each, about 24 TB in total. So videos are
downloaded in order of how many videos share their caption (a stock clip uploaded several times usually
keeps its caption), at most `--per-caption` per caption. I3D then decides which of them really are
duplicates, also across different captions. The pool grows with a larger `--max-gb`.

Data in `data/webvid/` (`--root`):

| file | content |
|---|---|
| `train.parquet` | `videoid, contentUrl, duration` (s)`, page_dir, name` (caption) |
| `captions.parquet` | one row per unique caption (normalized: case and white space ignored): representative `videoid`, `name`, `group` = number of videos |
| `videos/<id % 1000>/<videoid>.mp4` | downloaded videos (folder = last 3 digits of the id); `videos/failed.txt` lists the dead links, `failed_before_*.txt` older lists moved aside by `--retry-failed` |
| `i3d/feats_*.npz` | `videoid`, `feats` (1024: the last Inception block of I3D, averaged over space and time; used for the similarity), `logits` (400 Kinetics classes), float16 |
| `duplicates.parquet` | per video: `n_dup` (videos with cosine ≥ `--threshold`), `max_sim`, the `--top` nearest `neighbors` and their `neighbor_sims` |
| `prompts.csv` | the sheet: `index` (GIF name: the caption's smallest downloaded videoid), `caption`, `n_dup`, `dup_videoid` (its most duplicated video), `caption_count`, `checked` (videos of the caption with features) |

| command | option | default | meaning |
|---|---|---|---|
| `download` | `--max-gb` | 500 | stop when `videos/` holds this many GB |
| | `--per-caption` | 20 | at most this many (working) videos per caption |
| | `--for-videos` | — | folder of generated GIFs: download only the training videos of their captions |
| | `--retry-failed` | off | test the links in `failed.txt` again |
| | `--pause` | 300 | seconds to wait when the server throttles |
| | `--min-group` | 1 | only captions shared by at least this many videos |
| | `--max-videos`, `--workers` | all, 16 | first this many videos of the order; parallel downloads |
| `i3d` | `--frames` | 16 | frames evenly spaced over the whole clip, resized to 224×224 |
| | `--batch-size`, `--workers` | 32, 16 | videos per I3D pass; decoding processes |
| | `--i3d-weights` | `data/webvid/models/i3d_rgb_imagenet.pt` | Kinetics-400 I3D weights ([piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) `models/rgb_imagenet.pt`), downloaded if missing |
| `duplicates` | `--threshold` | 0.95 | cosine similarity that counts as a duplicate |
| `prompts` | `--checked-only` | off | **use it**: only captions with at least one downloaded, I3D-checked video |
| | `--out` | `data/webvid/prompts.csv` | sheet to write |

Notes:
- The previews carry a Shutterstock watermark in the middle of every frame, which makes all videos a bit
  similar. `duplicates` prints the similarity of each video to its nearest neighbor; choose `--threshold`
  well above the median.
- The duplication ranking only sees the downloaded videos, so the counts are lower bounds.
- `batch --frames/--steps/--seed/...` keep their defaults unless given. AnimateDiff v1 was trained
  at 256×256 on WebVid; the default here is 512×512 (`--height/--width 256` is closer to training and faster).

#### Measuring memorization: GSSCD (`webvid.py gsscd`)

GSSCD (Chen et al., *Investigating Memorization in Video Diffusion Models*, Eq. 2) measures content
memorization. It is the largest SSCD similarity over **all pairs of frames** of the generated video and a
training video:

```
GSSCD = max over generated frames i and training frames j of  cos( SSCD(gen_i), SSCD(train_j) )
```

SSCD ([Pizzi et al., CVPR 2022](https://github.com/facebookresearch/sscd-copy-detection)) is an image
copy-detection descriptor. A video counts as memorized when GSSCD ≥ 0.4, the paper's threshold.

```bash
python webvid.py gsscd                                     # outputs/webvid/normal -> outputs/webvid/gsscd_normal.csv/.png
python webvid.py gsscd --videos outputs/webvid_guard/guard # any folder of <videoid>.gif, e.g. a mitigation
```

**Which training video.** The GIF `<videoid>.gif` was generated from that video's caption. It is compared with
**every downloaded training video with the same caption** (up to 20, see `download --per-caption`), and the best
match counts. `gsscd_index` is the score against the video `<videoid>` alone.

**Output**, next to the GIF folder (`--out` to choose another file):

| file | |
|---|---|
| `gsscd_<folder>.csv` | one row per generated video, the highest GSSCD first (columns below) |
| `gsscd_<folder>.png` | histogram of GSSCD, with the threshold and the median |
| `gsscd_<folder>_pairs/<videoid>.jpg` | for every memorized video: the generated frame and the training frame of its GSSCD pair, side by side |

| column | meaning |
|---|---|
| `videoid` | the generated video (`<videoid>.gif`) |
| `caption` | its prompt |
| `gsscd` | GSSCD against the best-matching training video |
| `gen_frame` | the generated frame (0–15) of the most similar pair |
| `gt_videoid` | the training video of that pair |
| `gt_frame`, `gt_time` | the training frame of that pair: frame number in the `.mp4` and time in seconds |
| `memorized` | `gsscd ≥ --threshold` |
| `frames_memorized` | how many generated frames have a training frame with SSCD ≥ threshold (one frame vs. the whole video copied) |
| `mean_frame_sscd` | mean over the generated frames of their best SSCD |
| `gt_memorized` | how many training videos of the caption the video copies (GSSCD ≥ threshold with each) |
| `gsscd_index` | GSSCD against the training video `<videoid>` itself (empty if it was not downloaded) |
| `n_gt`, `n_gt_frames` | training videos and frames compared |
| `n_dup`, `caption_count` | from `prompts.csv`: I3D duplicates and videos with the caption, to relate duplication to memorization |

| option | default | meaning |
|---|---|---|
| `--videos` | `outputs/webvid/normal` | folder of generated GIFs |
| `--threshold` | 0.4 | GSSCD that counts as memorized |
| `--gt-fps` | 8 | training frames compared per second of training video (the clips are 25–30 fps; every frame would be about 3× slower) |
| `--crop` | `center` | `center`: compare the center square of the training frames, which is what AnimateDiff was trained on (the previews are 16:9, the videos square); `none`: the whole frame, squeezed |
| `--size` | 288 | frames are resized to `size × size` (SSCD's evaluation size) |
| `--workers` | 16 | processes decoding the training videos |
| `--no-pairs` | off | do not save the image pairs |
| `--sscd-weights` | `data/webvid/models/sscd_disc_mixup.classy.pt` | SSCD weights, downloaded if missing |

Notes:
- **It continues where it stopped.** Videos already in the CSV are skipped, so run it again as more GIFs are
  generated. Videos with no downloaded training video (caption not downloaded, or all its links dead) are not
  listed; the count is printed, and they are tried again next time.
- **Speed.** Every training video is decoded and embedded once, and its descriptors are kept in
  `data/webvid/sscd/<crop>_<size>px_<fps>fps/`. Later runs with the same settings, including runs on other
  mitigation folders, only embed the 16 generated frames. The first run embeds about 5 training videos per second
  (20 per caption), so about 4 s per generated video. `--gt-fps 4` halves that.
- **The paper leaves open** how many frames are compared, the preprocessing and which training video is used.
  The choices above are this project's. `sscd_disc_mixup` is the checkpoint the SSCD authors recommend.
- **Look at the pairs near the threshold.** SSCD gives fairly high scores to similar textures (bokeh, particles,
  debris on black) without a copy; in a first test the two videos just above 0.4 were such textures.
- The model is in `sscd.py`: torchvision's ResNet-50 with GeM pooling and a 512-d projection, loading the
  official Classy Vision checkpoint as plain tensors.

### 9. Compare memorized and non-memorized prompts (`compare`)

The spike plots do **not** tell a memorized prompt from a normal one. The z-score flags tokens that stand
out *within* a prompt, so every prompt has spikes. GUARD uses them only to choose *which* tokens to
attenuate, assuming the prompt is memorized. What does differ is how strongly the prompt drives the
generation and how that changes over the steps. The `attention` command therefore also writes
`memorization_signals.png`. To overlay several runs in one figure:

```bash
python main.py attention --prompt "As Punisher Joins <i>Daredevil</i> Season Two, Who Will the New Villain Be?" --out outputs/attention-punisher
python main.py attention --prompt "A fluffy orange cat walking slowly through a green garden" --out outputs/attention-cat
python main.py compare outputs/attention-punisher outputs/attention-cat --out outputs/compare.png
```

`compare` only reads each folder's `memorization_signals.json`, so it takes seconds and loads no model.
Use the same `--frames --steps --height --width --seed --s` for every run you compare.

| panel | what it is | memorized prompts (in our tests) |
|---|---|---|
| text guidance strength | `‖ε_cond − ε_null‖` per frame at every step (Wen et al.). Line = max over frames, band = min..max | much larger in the **second half** of denoising (e.g. 11 vs 2–4 at step 12). Unlike SD, no gap at step 0 |
| `<eot>` + padding share, down / mid / up | the share of cross-attention on `<eot>` and the padding tokens (Ren et al.; GUARD Fig. 7), **mean** over the block's layers, frames, heads and query pixels | does not fade: at the last step 0.04–0.06 (down) vs 0.01, and 0.09–0.19 (mid) vs 0.04–0.07 |
| change between consecutive frames | mean `|pixel difference|` between frame f and f−1 | larger: the video flickers or breaks apart instead of moving smoothly |
| similarity to frame 0 | cosine similarity of each frame to the first one | drops lower |

These come from 2 memorized and 2 normal prompts, so treat them as hints, not as a validated detector.
The `<eot>` share and the frame change are the only averaged quantities in the project. A share of
attention is a sum over tokens, and it is only meaningful as an average over the pixels that attend.

### 10. Artificial memorization: fine-tune the model on duplicated videos (`finetune.py`)

The model's natural memorization is rare and its strength is unknown. Here its whole UNet is fine-tuned on a
dataset in which some videos are duplicated, so you know which videos should be memorized and how strongly.

#### The dataset

Every video, every caption and every download link appears only once in the table `finetune.csv`, across both roles;
the duplication is its `copies` column. These are the default numbers; all of them can be changed (see below).

| role | samples | copies | why |
|---|---|---|---|
| `background` | 20000 | 1 | random WebVid-10M videos: keep the model on the training distribution; also the "no duplicates" control |
| `memorized` | 200 | 2, 3, 4, 5, 6, ... 51 (4 videos each) | the spectrum of memorization: 50 levels, the first 50 natural numbers starting from 2 |

One epoch is 20000 + 4 × (2 + 3 + 4 + ... + 51) = **25300 rows**.
To try another spectrum, change `duplication_frequencies()` in `finetune.py`.

How the samples are chosen, and why:

- **Background: drawn at random from the metadata of all 10.7M videos, not from the videos already downloaded.**
  `webvid.py download` starts with the most repeated captions, so the downloaded videos are not representative:
  the caption of the median downloaded video is used by 44 videos, that of the median WebVid video by 1 (itself). The draw shuffles
  the videos and then keeps the first video of each caption, so frequent captions stay more likely than rare ones.
  `build` prints the duration and caption length of the background next to those of all of WebVid, to compare.
- **Memorized: only captions no other WebVid video has** (54% of the videos), so a caption points to exactly one video.
  Which video gets how many copies is random. Near-duplicates under another caption cannot be excluded, because
  that would need all 10.7M videos.

#### The fine-tuning

**The whole UNet trains; CLIP and the VAE stay frozen.** Normal AnimateDiff training only trains the motion modules;
here all 1277 M parameters of the UNet train:

| network | parameters | how it is trained |
|---|---|---|
| UNet: image layers and motion modules | 1277 M | the diffusion loss (predict the noise added to the video latents) |
| CLIP text encoder | 123 M | frozen |
| VAE: encoder and decoder | 84 M | frozen |

CLIP and the VAE were also frozen when Stable Diffusion and AnimateDiff were trained. The image layers learned from
LAION images, not WebVid-10M, so training them on WebVid can also change what the model generates for other prompts. To
measure that, compare the fine-tuned and original models on prompts that are not memorized.

Settings as in AnimateDiff's training ([config](https://github.com/guoyww/AnimateDiff/blob/main/configs/training/v1/training.yaml)):
16 frames, every 4th frame, 256×256 center crop, linear noise schedule, 10% empty prompts, gradient clipping at 1.
What differs, and why:

| | AnimateDiff | here | why |
|---|---|---|---|
| trained parameters | motion modules | the whole UNet (motion modules and image layers) | memorize more strongly than the motion modules alone, which get no caption directly |
| clip | random position, random flip | always the first 61 frames, no flip | the copies of a video must be the same sample |
| learning rate | 1e-4 | 1e-5 (`--lr`) | the usual range for fine-tuning a network that already works |
| optimizer | AdamW | 8-bit AdamW (`bitsandbytes`) | normal AdamW needs about 8 GB more (calculated): that would not fit in a 24 GB GPU |

#### What to do

Run everything from this folder. On the server, use the right environment (the home disk is full):

```bash
export HF_HOME=/storage/Rad/hf_cache
source /storage/Rad/venv_cu126/bin/activate
```

**1. Draw 25000 random candidates** (more than needed: about 7% of the links are dead), and **download them** (about 56 GB).
The download can be stopped and started again:

```bash
python finetune.py select
python webvid.py download --ids-from data/webvid/finetune/candidates.csv --per-caption 0
```

**2. Build the table.** This keeps the first videos that can be read and assigns the copies:

```bash
python finetune.py build
```

It writes, in `data/webvid/finetune/`:

| file | content |
|---|---|
| `finetune.csv` | what is trained on: `videoid, contentUrl, caption, copies, role` |
| `memorized_prompts.csv` | a sheet for `main.py batch` with the 200 memorized videos: `index, caption, copies` |

If it says there are too few samples, run `select --candidates 30000` (the first 25000 stay the same) and download again.

To choose the numbers:

```bash
python finetune.py select --candidates 30000
python finetune.py build --background 25000 --memorized 300 --per-level 4
```

| option | default | meaning |
|---|---|---|
| `select --candidates` | 25000 | videos to draw and download. Both roles come from these, so it must be more than `--background` + `--memorized`, plus about 7% for dead links |
| `build --background` | 20000 | samples with one copy |
| `build --memorized` | 200 | samples to memorize. If this is not a multiple of `--per-level`, the last level has fewer videos |
| `build --per-level` | 4 | memorized samples for each number of copies |

More memorized samples also means more levels, so a higher largest number of copies: `--memorized 300` gives 2 to 76.

**3. Try a few steps first**, to see that it runs (a few minutes, with loading and saving):

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 finetune.py train --max-steps 20 --out outputs/finetune_test
rm -r outputs/finetune_test
```

**4. Fine-tune.** It takes a long time, so run it in the background, with the logs in a file. It keeps going if you
close the terminal or the SSH connection drops:

```bash
nohup accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 \
    finetune.py train --out outputs/finetune --epochs 10 --seed 0 \
    > outputs/finetune.log 2>&1 &
```

`nohup` keeps it running after the terminal closes, `> outputs/finetune.log 2>&1` writes the output and the errors to the
file, and `&` gives you the terminal back.

```bash
tail -f outputs/finetune.log                     # watch the logs (Ctrl+C stops watching, not the training)
tr '\r' '\n' < outputs/finetune.log | tail -1     # only the last line of the progress bar
pgrep -af "finetune.py train"                    # is it running?
pkill -f "finetune.py train"                     # stop it: what came after the last saved epoch is lost
```

A new run refuses an `--out` folder that already has checkpoints, so it cannot overwrite them. The log is not protected:
`>` replaces the file, so use a new log name for each run, or `>>` to add to it (as in the `--resume` command below).

On one GPU, use `python finetune.py train ...` instead of `accelerate launch ... finetune.py train ...`. The progress bar shows the loss. With 4 GPUs (8 videos per step) and the default numbers, an epoch is about 3160 steps.
In a short test with the whole model trained, a step took about 4 s on 4 GPUs (training only the UNet should not be slower), so expect about 3.5 hours per epoch and about 36 hours for 10 epochs.
Check the real speed in step 3: decoding many more videos from the network drive can make it slower.

A checkpoint is saved after every epoch in `outputs/finetune/epoch_<n>/` (7.4 GB: the 5.1 GB model and 2.3 GB to continue
training; about 74 GB for 10 epochs) and the model at the end in `outputs/finetune/final/` (5.1 GB). `--save-every` saves
less often. **10 epochs is a starting point, not a measured value**: nobody has
measured yet how many epochs each level of copies needs, which is why every epoch is kept.

**If it stops, or to train further, continue from the latest checkpoint** with `--resume` and the same `--out`. It finds the
last complete checkpoint by itself and carries on where that epoch ended; what came after that epoch is trained again.
`--epochs` is the total of the run, so raise it to train further than planned:

```bash
nohup accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 \
    finetune.py train --out outputs/finetune --epochs 10 --seed 0 --resume \
    >> outputs/finetune.log 2>&1 &
```

It uses the table saved in `--out`, and refuses a different `--batch-size`, `--lr`, `--size`, `--null-prompt`, `--seed` or
number of GPUs, because that would not continue the same training.

To start a new training from a model instead, use `--model` and a new folder. The optimizer's statistics and the order of
the rows start over, and the epochs count from 1:

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 finetune.py train --model outputs/finetune/epoch_5 --out outputs/finetune_more --epochs 5
```

| option of `train` | default | meaning |
|---|---|---|
| `--epochs` | 10 | total passes over all the rows of the table (25300 by default), also with `--resume` |
| `--batch-size` | 2 | videos per GPU; 3 does not fit in 24 GB (peak with 2: 20.4 GB) |
| `--lr` | 1e-5 | learning rate of the UNet |
| `--size` | 256 | video size in training |
| `--null-prompt` | 0.1 | share of samples trained on the empty prompt |
| `--save-every` | 1 | save a checkpoint every this many epochs |
| `--resume` | off | continue the run in `--out` from its latest checkpoint |
| `--model` | original AnimateDiff | start a new training from a saved model |
| `--max-steps` | none | stop early, for a test |
| `--seed` | 0 | the same seed and settings give the same weights (see Reproducibility below) |

**5. Measure what was memorized** (`memorization_eval.py`). Each caption is generated once, for two groups: the memorized
samples, and the same number of random background samples.

```bash
python memorization_eval.py prompts          # writes outputs/finetune_eval/prompts.csv

for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python main.py --model outputs/finetune/final batch \
      --csv outputs/finetune_eval/prompts.csv --out outputs/finetune_eval --experiments normal \
      --height 256 --width 256 --shard $k/4 > outputs/finetune_eval_$k.log 2>&1 &
done

python memorization_eval.py metrics          # once all the videos are generated (it refuses otherwise)
```

`--model` comes before `batch`. **Generate at 256×256**, the size it was fine-tuned at: at the default 512×512 the memorized
videos may not come back. Generation can be stopped and started again; videos that exist are skipped.

Both metrics are from Chen et al., [*Investigating Memorization in Video Diffusion Models*](https://arxiv.org/abs/2410.21669).
Every generated video is compared with **the exact 16-frame clip it was fine-tuned on** (center crop, every 4th frame):

| metric | measures | how | memorized when |
|---|---|---|---|
| **GSSCD** | content | largest SSCD cosine similarity over all pairs of (generated frame, training frame) | ≥ 0.4 |
| **OFS-k** | motion | RAFT optical flow between consecutive frames, cosine similarity per pixel averaged over the frame, then the best average over `k` consecutive flows, over all alignments of the two videos | > 0.5 |

`--k` is 3 by default, the paper's best value. OFS-k is written twice, as `ofs_3` and `ofs_3_nmf`, because the paper reports it
without and with its natural motion filter, which ignores static frames and camera panning.

**Choices that are not from the paper.** The paper does not give the filter's details, so these are my choices and have not
been calibrated: a flow is static if its mean length is below 0.5 pixels (`--min-magnitude`), it is panning if the entropy of
its 36-bin direction histogram, scaled to 0–1, is below 0.5 (`--min-entropy`), the filter is applied to both videos, and a window
with an ignored flow does not count. The paper also does not say which RAFT weights it used (here: torchvision's `raft_large`,
downloaded the first time) or how many training frames it compares. `webvid.py gsscd` compares more frames, so its numbers are
not directly comparable.

It writes, in `outputs/finetune_eval/`:

| file | content |
|---|---|
| `metrics.csv` | one row per video: `videoid, role, copies, caption, gsscd, ofs_3, ofs_3_nmf`, and whether each counts as memorized |
| `summary_role.csv` | memorized against background: mean, median and share of memorized videos for each metric |
| `summary_copies.csv` | the same for each number of copies: both metrics should grow with the copies |

`ofs_nmf_videos` in the summaries is the number of videos with a window left after the filter; the others have an empty
`ofs_3_nmf` and count as not memorized.

**Figures for presenting.** `python memorization_eval.py plot` draws `metrics.csv` as PNG and PDF in `outputs/finetune_eval/plots/`
(add `--out <folder>` before `plot` for another run; `--groups` changes the groups of copies; `--title` writes a title above
every figure):

| figure | shows |
|---|---|
| `distributions` | how the scores of the memorized and background videos are spread |
| `score_vs_copies` | every memorized video against its number of copies, the trend, and the background median |
| `share_memorized` | the share of videos above the threshold: the background, then groups of copies |
| `content_vs_motion` | GSSCD against OFS-k for every video, with how many videos fall on each side of the two thresholds |

**6. The same for the original model, and the comparison.** The original AnimateDiff never saw the duplicates, so it shows
what the scores are without the fine-tuning. Use the **same `prompts.csv`**: both models then generate each caption from
the same starting noise (seed 0), and every video has a twin from the other model.

```bash
mkdir -p outputs/finetune_eval_original
cp outputs/finetune_eval/prompts.csv outputs/finetune_eval_original/

for k in 0 1 2 3; do   # no --model: the original model
  CUDA_VISIBLE_DEVICES=$k nohup python main.py batch \
      --csv outputs/finetune_eval_original/prompts.csv --out outputs/finetune_eval_original --experiments normal \
      --height 256 --width 256 --shard $k/4 > outputs/finetune_eval_original_$k.log 2>&1 &
done

python memorization_eval.py --out outputs/finetune_eval_original metrics --videos outputs/finetune_eval_original/normal
python memorization_eval.py --out outputs/finetune_eval_original plot --title "Original model (not fine-tuned)"
python memorization_eval.py --out outputs/finetune_eval plot --title "Fine-tuned model"
python memorization_eval.py compare     # writes outputs/finetune_compare/
```

In the original model's figures, "memorized" and "background" are only the names of the two groups of captions: for
that model, both groups are ordinary WebVid samples, so it should score them alike.

`compare` (options `--finetuned`, `--original`, `--save-to`, `--groups`, the thresholds) writes to `outputs/finetune_compare/`.
In its figures, **gray is the original model**, and the fine-tuned model keeps the colors of the other figures (blue memorized,
orange background):

| file | shows |
|---|---|
| `compare_share_memorized` | the share of videos above the threshold, per group, original bar next to fine-tuned bar |
| `compare_score_vs_copies` | the mean score against the copies for both models; the background is the diamond at 1 copy |
| `compare_paired` | every video: its original score (x) against its fine-tuned score (y); above the diagonal = the fine-tuning brought it closer to its training clip |
| `compare_distributions` | the scores of both models, memorized samples on top, background below |
| `paired_metrics.csv` | one row per video with the scores of both models (`gsscd_original`, `gsscd_finetuned`, ...) |
| `compare_summary.csv` | per group and metric: both means, the mean change, both shares above the threshold, how many twins went up, and the Wilcoxon signed-rank p-value of the change (is it larger than chance?) |

With the filter, a video is compared only where both models' videos have a window left, so those numbers cover fewer videos.

**7. The mitigation methods on the fine-tuned model.** Generate the same 400 captions (200 memorized, 200 background) with
each mitigation method of sections 2–6 (`guard`, `soften`, `clip`, `noise`, `sail`), using the fine-tuned model, then measure
GSSCD and OFS-k of every video against its training clip, as in step 5.

Generate into **the same folder as step 5**, `outputs/finetune_eval`. Its `normal/` videos are then the "no mitigation"
reference, and every method starts each caption from the same noise (seed 0) as its `normal` video. Each method writes to
its own subfolder (`outputs/finetune_eval/guard/`, ...). The other options must stay as in step 5 (`--height 256 --width 256`,
same `--model`): `batch` refuses a folder made with other settings. The `soften` videos of this step were first made with
`-T 2` (the default until 2026-11-17); see "Running one method again with other settings" below for how they were redone
with the new default `-T 1.2`.

```bash
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python main.py --model outputs/finetune/final batch \
      --csv outputs/finetune_eval/prompts.csv --out outputs/finetune_eval --experiments guard,soften,clip,noise,sail \
      --height 256 --width 256 --shard $k/4 > outputs/finetune_mitigation_$k.log 2>&1 &
done
tail -f outputs/finetune_mitigation_0.log      # progress: "[n/100] <method>: index ..."; Ctrl+C stops only the tail
```

Every GPU goes through its 100 captions and makes the five videos of each caption before the next. `noise` and `sail`
optimize the starting noise through the UNet, so they take much longer than the other three (I have not timed them at
256×256). A stopped run can be started again with the same command; videos that exist are skipped. To run the methods
one after another instead, give one method to `--experiments` at a time.

Then, once all the videos exist (it refuses otherwise):

```bash
python memorization_eval.py metrics --experiments guard,soften,clip,noise,sail       # about 5× as long as step 5
for m in guard soften clip noise sail; do python memorization_eval.py plot --experiment $m; done
python memorization_eval.py mitigation                                              # writes outputs/finetune_mitigation/
```

`metrics --experiments` reads `<out>/<method>/` and writes, next to the files of step 5, `metrics_<method>.csv`,
`summary_role_<method>.csv` and `summary_copies_<method>.csv` (`normal` keeps the names `metrics.csv`, ...). Each training
clip is read and measured once for all the methods. `plot --experiment <method>` draws the four figures of step 5 for one
method in `outputs/finetune_eval/plots_<method>/`.

`mitigation` puts all the methods together (options `--experiments`, `--original`, `--save-to`, `--groups`, the thresholds). It
draws the original model too, as the reference, if `outputs/finetune_eval_original/metrics.csv` exists (`--original ''` to leave
it out). A method works if the memorized samples are copied less than without mitigation, ideally as rarely as by the
original model, which never saw the duplicates.

| file | shows |
|---|---|
| `mitigation_share_memorized` | for every method, the share of the memorized samples (blue) and background samples (orange) above the threshold, one panel per metric |
| `mitigation_score_vs_copies` | one panel per method and metric (GSSCD, OFS-k): the mean score against the copies with the method (blue), without mitigation (dark gray) and from the original model (light gray). The lower the blue line under the dark gray one, the more the method reduces the copying |
| `mitigation_summary.csv` | per method, group and metric: the mean, the share above the threshold, the mean change against no mitigation and its Wilcoxon p-value (same caption and seed, so each video is compared with its own twin), `memorized_without_mitigation` (how many videos of the group are above the threshold without mitigation) and `still_memorized` (how many of those stay above it with the method) |

**Running one method again with other settings** (example: `soften` with `-T 1.2` instead of 2). The other methods and
`normal/` stay as they are; only the method's own files are moved to an archive folder, so the old results are kept and
`batch` accepts the new settings (it refuses while `outputs/finetune_eval/soften/` still holds videos of another `-T`).

```bash
# 1. keep the T = 2 results: the videos, their measurements, and the summaries that include them
mkdir outputs/finetune_eval_soften_T2
mv outputs/finetune_eval/{soften,plots_soften,metrics_soften.csv,summary_role_soften.csv,summary_copies_soften.csv,quality_soften.csv,i3d_features_soften.npy} \
   outputs/finetune_eval_soften_T2/
cp -r outputs/finetune_mitigation outputs/finetune_mitigation_soften_T2
cp -r outputs/finetune_quality outputs/finetune_quality_soften_T2

# 2. the 400 soften videos with T = 1.2 (now the default, so -T can be left out), same model, size and batch size as before
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python main.py --model outputs/finetune/final batch \
      --csv outputs/finetune_eval/prompts.csv --out outputs/finetune_eval --experiments soften -T 1.2 \
      --height 256 --width 256 --batch-size 5 --shard $k/4 > outputs/finetune_soften_T1.2_$k.log 2>&1 &
done

# 3. once all 400 exist: memorization (step 7) and quality (step 8) of the new videos, then the summaries of all methods
python memorization_eval.py metrics --experiments soften
python memorization_eval.py plot --experiment soften
python memorization_eval.py mitigation
python quality_eval.py metrics --experiments soften
python quality_eval.py summary
```

**The "only frames above the threshold" variants** (`noise_above`, `sail_above`, section 6a) are two more methods in
the same folder, compared with all the others. To compare them fairly with `noise/` and `sail/`, they must use **the same
thresholds** as those videos (`--target 0.2`, SAIL 3.4 = `α·√4096 + 0.2`) and the same batch size and shards (the
batch composition changes the float16 results slightly). Give the thresholds explicitly, since the defaults may have
changed since `noise/` and `sail/` were made. `batch` does not compare a variant's settings with its base method's:
check that they match yourself.

```bash
# 1. the videos (each also gets <index>.json: every frame's loss at every optimizer step)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python main.py --model outputs/finetune/final batch \
      --csv outputs/finetune_eval/prompts.csv --out outputs/finetune_eval --experiments noise_above,sail_above \
      --target 0.2 --sail-thres 3.4 \
      --height 256 --width 256 --batch-size 5 --shard $k/4 > outputs/finetune_above_$k.log 2>&1 &
done

# 2. once all 400 of each exist: memorization and quality, then the summaries with all eight
python memorization_eval.py metrics --experiments noise_above,sail_above
for m in noise_above sail_above; do python memorization_eval.py plot --experiment $m; done
python memorization_eval.py mitigation --experiments normal,guard,soften,clip,noise,noise_above,sail,sail_above
python quality_eval.py metrics --experiments noise_above,sail_above
python quality_eval.py summary --experiments normal,guard,soften,clip,noise,noise_above,sail,sail_above

# 3. optional: per-frame loss plots of the new videos (heatmaps of all 400 per folder, and a summary table)
python main.py plot-frameloss outputs/finetune_eval/noise_above --save-to outputs/finetune_eval/frameloss_noise_above
python main.py plot-frameloss outputs/finetune_eval/sail_above --save-to outputs/finetune_eval/frameloss_sail_above
```

Without `--experiments`, `mitigation` and `summary` show the six main experiments only. The old `noise/` and `sail/`
videos have no per-frame logs, so only the `_above` variants can be plotted per frame unless those are made again.

**Things to keep in mind**
- The methods run with their default settings (sections 2–6). Those were chosen on the original model at 512×512; the
  fine-tuned model at 256×256 may need other strengths. `sail`'s default stopping threshold follows the latent size, so it
  adapts to 256×256; `noise`'s `--target` does not. To try other settings, either run the method again in the same folder
  (above), or use another `--out` and copy `prompts.csv`, `normal/`, `metrics.csv`, `summary_role.csv` and
  `summary_copies.csv` of `outputs/finetune_eval` into it first, so that no mitigation is the same reference.
- A lower score is not the whole story: a method that turns every video into noise would also score low. Look at the
  videos of each method (`outputs/finetune_eval/<method>/`) to see whether they still show what the caption asks for.
- The background samples show the side effects: a method should change them little, since they were not memorized.
- Step 8 measures the quality of the videos, to see what each method costs.

**8. Video quality: FVD and ViCLIP similarity** (`quality_eval.py`). Less copying is only useful if the videos stay good.
Two scores, for the same 400 videos of every experiment (no mitigation and each method), and optionally the original model:

| score | measures | how | better |
|---|---|---|---|
| **FVD** (Unterthiner et al., 2018) | realism: do the videos look like real WebVid videos? | Fréchet distance between the I3D features (Kinetics-400, the 400 logits) of the generated videos and of real clips | lower |
| **ViCLIP similarity** (InternVid, Wang et al., 2023) | does the video show its caption? | cosine similarity of the ViCLIP-L/14 features of the video (8 frames, 224×224) and of the caption | higher |

```bash
python quality_eval.py check-viclip     # once: ViCLIP must give the result of its official demo (printed next to it)
python quality_eval.py reference        # once: the real clips (FVD reference and the real training clips)
python quality_eval.py metrics --experiments normal,guard,soften,clip,noise,sail
python quality_eval.py --out outputs/finetune_eval_original metrics --experiments normal     # optional: original model
python quality_eval.py summary          # table and figures in outputs/finetune_quality/
```

`metrics` refuses to start while videos are missing. It writes, next to the videos' other results in `--out`,
`quality.csv` (one row per video: `videoid, role, copies, caption, viclip`) and `i3d_features.npy` for no mitigation, and
`quality_<method>.csv`, `i3d_features_<method>.npy` for the methods. `summary` reads them, and `metrics*.csv` of
`memorization_eval.py` for the share of memorized samples still copied, and writes to `outputs/finetune_quality/`:

| file | shows |
|---|---|
| `quality_summary.csv` | per row (real training clips, original model, no mitigation, each method) and group (all, memorized samples, background samples): `fvd`, `viclip_mean`, `viclip_std`, the change of ViCLIP against no mitigation with its Wilcoxon p-value (each video against its twin), and `memorized_copied_%` (GSSCD ≥ 0.4) |
| `quality_scores` | FVD (top) and ViCLIP similarity (bottom) of every row, memorized samples (blue) and background samples (orange) |
| `quality_vs_memorization` | the trade-off, memorized samples only: how many are still copied (x) against their ViCLIP (left) and FVD (right). The best methods are on the left with the quality of no mitigation; the dashed line is the quality of the real training clips |

**What the real clips are.** `reference` reads two sets of real clips, prepared as for the fine-tuning (16 frames, every 4th,
center square, 256×256):
- **The FVD reference**: 2048 random background clips of the fine-tuning table (`--size`, `--seed`), but **not** the 400
  evaluated ones. Against the evaluated videos' own training clips, a copy would get a better FVD: a memorizing model
  would look the most realistic. Saved in `outputs/finetune_quality/fvd_reference.npz`.
- **The training clips of the 400 captions**, scored like generated videos (row "Real training clips"): the best FVD and
  ViCLIP that real videos get with the same number of videos.

**Choices and limits**
- **I3D:** the TorchScript I3D of StyleGAN-V (`data/webvid/models/fvd/i3d_torchscript.pt`, downloaded once and checked
  against its SHA-256), the one most video generation papers use, so the numbers can be compared with theirs. `i3d.py` is
  another port of I3D, used by `webvid.py` to find duplicated videos; it is not used for FVD.
- **FVD with few videos is noisy and biased upwards.** Papers use thousands of videos; here a group has 200 (all 400 in
  "All"). The real training clips of the memorized and of the background captions, 200 each, are about 590 apart. Compare rows of the same group with each other and with
  the real training clips, and do not compare these numbers with FVDs of papers.
- **ViCLIP:** `OpenGVLab/ViCLIP-L-14-hf`, fixed to one version. Its model code comes with the weights from Hugging Face and
  runs with `trust_remote_code`; the version is pinned so that the code that runs is the code that was read. It needs
  `einops`, `ftfy` and `timm` (`pip install einops ftfy timm`). ViCLIP reads at most 32 tokens, so the end of long captions
  is ignored. The raw cosine similarities are small (about 0.2–0.3 for matching pairs); only differences between rows matter.
- **ViCLIP is a caption score, not a memorization score**: a copied training video matches its caption well. Read it with
  the memorization: a good method keeps ViCLIP near no mitigation while fewer videos are copied.

#### Reproducibility

Running `train` again from the start with the same `--seed` gives **exactly the same weights**, bit for bit (checked by comparing every
saved tensor of two short runs on 4 GPUs, 20 steps each). Use another `--seed` for a different run of the same experiment:

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 finetune.py train --out outputs/finetune_seed1 --seed 1
```

The seed sets the order of the rows in each epoch, the empty prompts, the noise, the timesteps and the latents sampled from the VAE. The
number of decoding workers (`--workers`) and which GPUs of the server are used do not change the result.

The result does change with:

| what | why |
|---|---|
| another number of GPUs or `--batch-size` | the batches are different |
| another table | see below |
| any other option (`--lr`, `--size`, ...) | a different training |
| another version of the code or of the packages (`requirements.txt`), or another GPU model or driver | floating-point results can differ; only the same setup was tested |
| continuing with `--model` | the optimizer's statistics and the order of the rows start over. `--resume` restores them, but the result is not bit-for-bit the same as a run that was never stopped: see below |

**Continuing with `--resume`.** It is the same training, but the weights do not stay bit-for-bit those of a run that was never stopped. In a test on
a small table on 4 GPUs, the first step after resuming had the same inputs, noise, loss, weights and optimizer state, but
some gradients differed by rounding, and the difference grew quickly: 12 weights by about 1e-10 after 6 steps, most weights by about 1e-8 on average after 12. Two resumes from the same
checkpoint gave the same weights (checked after one step). The cause inside the gradient computation was not found. If a
result must be repeatable bit for bit, keep the run unstopped and repeat it unstopped.

**The table.** `select --seed` always draws the same candidates, but `build` keeps those that can be read, and links keep
dying, so a later download can give another table. To repeat a run, use the copy of the table saved with it. Every run
saves in its `--out` folder:

| file | content |
|---|---|
| `settings.json` | all the options of the run, and the number of GPUs |
| `finetune.csv` | the table it was trained on (use it with `--table`) |


## Code

| file | content |
|---|---|
| `sscd.py` | the SSCD copy-detection network (ResNet-50 + GeM), for the official `sscd_disc_mixup` weights; used by `webvid.py gsscd` |
| `i3d.py` | the I3D network (Inception-I3D, RGB), for the Kinetics-400 weights of DeepMind's model converted to PyTorch |
| `finetune.py` | artificial memorization (section 10): chooses the samples, assigns the copies, and fine-tunes the UNet (CLIP and the VAE stay frozen) |
| `memorization_eval.py` | measures the artificial memorization (section 10): the captions of the memorized samples and as many background samples, then GSSCD and OFS-k of the generated videos, figures of the results, the comparison with the original model, and the comparison of the mitigation methods |
| `quality_eval.py` | video quality of the same videos (section 10, step 8): FVD against real WebVid clips and ViCLIP similarity to the caption, for no mitigation, each mitigation method and the original model |
| `webvid.py` | WebVid-10M: metadata, video download, I3D features, duplication ranking, the prompt sheet and GSSCD of generated videos (section 8) |
| `main.py` | command line (`attention`, `guard`, `soften`, `clip`, `noise`, `sail`, `batch`, `compare`); `experiment()` builds the normal / guard / soften / clip / noise / sail setup that all commands share |
| `sampler.py` | loads AnimateDiff and runs the denoising loop for one prompt or a batch of prompts: normal CFG (2 streams) or GUARD / soften with repulsion (3 streams); `embed_prompts` builds the text embeddings, including the `clip` mitigation; `adjust_noise` is the `noise` experiment, `sail` the `sail` experiment, `guidance_vjp` the memory-saving gradient both use |
| `attention.py` | the attention processor put in every layer: `spike_scores` / `find_spikes` (detector), `Processor.attenuated` (GUARD), `Processor.softened` (temperature), `Processor.recorded` (stores maps for plots), `select_layers` (layer choice for `soften`) |
| `plots.py` | all figures and the text summaries; `memorization_signals` / `plot_memorization_signals` for section 9 |

## Notes

- Memory and speed (512×512, 16 frames, 25 steps, 16 GB GPU): normal sampling takes about
  1.25 s/step, GUARD about 1.9 s/step (3 streams). `guard` peaks at about 9 GB of GPU memory.
  `attention` needs more because of the explicit spatial attention maps. If you run out of memory, lower
  `--height/--width` (e.g. 384) or `--frames`. AnimateDiff v1 was trained at 256×256.
- With only 16 frames, a z-score cannot exceed `15/sqrt(16) = 3.75`. With `tau = 3`, spiky key frames
  in temporal attention are therefore rare; try `--tau 1` or `2` for those plots.
- The `attention` command computes the spatial self-attention explicitly (4096 × 4096 per head and
  frame at 512×512), so it is slower than `guard`. In RAM it keeps about 0.4 GB of maps per 25 steps.
- `attention` uses plain softmax attention in fp16, which differs from diffusers' fused kernel only by
  rounding (max difference ≈ 0.008 in the noise prediction). Without recording or attenuation, the
  processor is identical to diffusers' and the sampler reproduces `AnimateDiffPipeline` frame-for-frame.
