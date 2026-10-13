# GUARD for AnimateDiff

This project applies **CA-in-GUARD** to the text-to-video model AnimateDiff. CA-in-GUARD is the
memorization-mitigation method from *"You Don't Need All That Attention: Surgical Memorization
Mitigation in Text-to-Image Diffusion Models"* (Zhao et al., ICML 2026,
[code](https://github.com/kairanzhao/GUARD)). The project also plots every attention map of the
model and marks the "spiky" ones.

- Model: `CompVis/stable-diffusion-v1-4` + motion adapter `guoyww/animatediff-motion-adapter-v1-4`
- Two commands:
  - `attention` generates a video, then plots spatial, cross- and temporal attention for every layer, head, frame and diffusion step, with spikes marked.
  - `generate` generates a video with GUARD (and, optionally, the normal video for comparison).

## The method in short

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

### Paper vs. official code

Where they differ, this project follows the **official code**:

| | paper text | official code (used here) |
|---|---|---|
| `M_i` is computed from | attention probabilities | pre-softmax logits |
| maps used for detection | the negative (normal) pass | the positive stream's own logits |
| `tau` | "e.g. 3" | 0.1 for CA-in-GUARD (0.01 for CA attenuation alone) |
| repulsion | constant `r` | `r_t = r * cos(pi/2 * step/steps)`, decays to 0 (`--r-schedule constant` for the paper's version) |

### How it is adapted to AnimateDiff

- AnimateDiff runs the SD UNet on every frame, so in spatial and cross attention each frame is one
  batch row. Spikes are therefore detected and attenuated **per frame and per head**, in exactly the
  same way the official code works per image and per head.
- AnimateDiff adds **temporal attention** (motion modules), in which each pixel location attends to
  the same location in all 16 frames. GUARD does not modify it. The `attention` command plots it and
  marks spiky *key frames* with the same z-score rule.

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

Options shared by both commands:

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
| `--tau` | 3.0 | spike threshold on the z-score (3 marks strong outliers only; use 0.1 to see what `generate` attenuates) |
| `--plot-steps` | all | diffusion steps to plot, e.g. `0,12,24` (0 = first, noisiest step) |

Output in `outputs/attention/`:

```
video.gif                    the generated video
cross_summary.png            like Fig. 1 of the paper: attention per token in the first and last step,
                             and the attention on <eot> over the steps
spikes.txt                   every layer's spiky tokens / key frames / pixel locations, with in how many
                             (step, frame, head) rows each is a spike, as n/N
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

Plotting is the slow part: 72 figures per step, saved at `dpi=300` (set in `save()` in `plots.py`;
lower it for faster, smaller images). That is about a minute per step, so use `--plot-steps` to look at
a few steps first.

### 2. Generate a video with GUARD

```bash
python main.py generate --prompt "Living in the Light with Ann Graham Lotz" --baseline
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
| `--baseline` | off | also make the normal video |

The defaults are the official repo's CA-in-GUARD settings for SD v1.4. The paper's search ranges
(their Table 2) were `tau` 0.1–3, `alpha` 0.1–0.9 and `r` 0.1–6.

## Code

| file | content |
|---|---|
| `main.py` | command line (`attention`, `generate`) |
| `sampler.py` | loads AnimateDiff and runs the denoising loop: normal CFG (2 streams) or GUARD (3 streams) |
| `attention.py` | the attention processor put in every layer: `spike_scores` / `find_spikes` (detector), `Processor.attenuated` (GUARD), `Processor.recorded` (stores maps for plots) |
| `plots.py` | all figures and the text summaries |

## Notes

- Memory and speed (512×512, 16 frames, 25 steps, 16 GB GPU): normal sampling takes about
  1.25 s/step, GUARD about 1.9 s/step (3 streams). `generate` peaks at about 9 GB of GPU memory.
  `attention` needs more because of the explicit spatial attention maps. If you run out of memory, lower
  `--height/--width` (e.g. 384) or `--frames`. AnimateDiff v1 was trained at 256×256.
- With only 16 frames, a z-score cannot exceed `15/sqrt(16) = 3.75`. With `tau = 3`, spiky key frames
  in temporal attention are therefore rare; try `--tau 1` or `2` for those plots.
- The `attention` command computes the spatial self-attention explicitly (4096 × 4096 per head and
  frame at 512×512), so it is slower than `generate`. In RAM it keeps about 0.4 GB of maps per 25 steps.
- `attention` uses plain softmax attention in fp16, which differs from diffusers' fused kernel only by
  rounding (max difference ≈ 0.008 in the noise prediction). Without recording or attenuation, the
  processor is identical to diffusers' and the sampler reproduces `AnimateDiffPipeline` frame-for-frame.
