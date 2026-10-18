# GUARD for AnimateDiff

This project applies **CA-in-GUARD** to the text-to-video model AnimateDiff. CA-in-GUARD is the
memorization-mitigation method from *"You Don't Need All That Attention: Surgical Memorization
Mitigation in Text-to-Image Diffusion Models"* (Zhao et al., ICML 2026,
[code](https://github.com/kairanzhao/GUARD)). The project also plots every attention map of the
model and marks the "spiky" ones. A third experiment, `soften`, tests a simpler alternative to GUARD:
instead of detecting spiky tokens and attenuating only those, it softens all the attention of chosen
layers with a temperature.

- Model: `CompVis/stable-diffusion-v1-4` + motion adapter `guoyww/animatediff-motion-adapter-v1-4`
- Four commands:
  - `attention` generates a video, then plots spatial, cross- and temporal attention for every layer, head, frame and diffusion step, with spikes marked.
  - `generate` generates a video with GUARD (and, optionally, the normal video for comparison).
  - `soften` generates a video with temperature-softened attention in the layers you choose (and, optionally, the normal video).
  - `batch` runs any of these experiments (normal, GUARD, soften) on every prompt of a CSV sheet and saves the GIFs.

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
| `--tau` | 3.0 | spike threshold on the z-score (3 marks strong outliers only; use 0.1 to see what `guard` attenuates) |
| `--plot-steps` | all | diffusion steps to plot, e.g. `0,12,24` (0 = first, noisiest step); also the lines drawn in `cross_steps/` |

Output in `outputs/attention/`:

```
video.gif                    the generated video
memorization_signals.png     signals that separate memorized from normal prompts (see section 5)
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

### 2. Generate a video with GUARD

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
| `--tau` | 3.0 | spike threshold on the z-score |
| `--blocks` | `down,mid` | UNet parts whose cross-attention is attenuated (`down`, `mid`, `up`) |
| `--baseline` | off | also make the normal video |

The defaults are the official repo's CA-in-GUARD settings for SD v1.4. The paper's search ranges
(their Table 2) were `tau` 0.1–3, `alpha` 0.1–0.9 and `r` 0.1–6.

### 3. Soften attention with a temperature

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
# cross-attention of the down and mid blocks (the layers GUARD attenuates), T = 2, with the normal video for comparison
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
| `--temperature`, `-T` | 2.0 | logits are divided by `T` before the softmax |
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

### 4. Run the experiments on a CSV sheet of prompts

The sheet must have a column `caption` (the prompt) and a column `index` (used as the file name). Other
columns are ignored, and rows with an empty caption are skipped.

```csv
index,caption
101,Living in the Light with Ann Graham Lotz
102,"a cat playing piano, studio light"
```

```bash
# all three experiments on the whole sheet
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

**Settings are fixed per folder.** The first run writes `settings.json`. A later run into the same
folder with different generation settings (e.g. another `--alpha` or `-T`) stops with an error, so
videos made with different settings are never mixed. Use a new `--out` for each setting. `--start`,
`--end`, `--batch-size`, `--experiments` and the `--csv` / `--out` paths may change between runs.

| option | default | meaning |
|---|---|---|
| `--csv` | (required) | the sheet, with columns `index` and `caption` |
| `--out` | `outputs/batch` | output folder |
| `--experiments` | `normal,guard,soften` | which experiments to run |
| `--start`, `--end` | whole sheet | rows to run |
| `--batch-size` | 1 | prompts generated together in one UNet pass (see below) |
| `--guard-r`, `--alpha`, `--tau`, `--blocks` | 2.0, 0.3, 0.1, `down,mid` | GUARD settings, as in `guard` (`--guard-r` is `guard`'s `--r`) |
| `-T`, `--kinds`, `--layers`, `--keep-sot`, `--soften-r` | 2.0, `cross`, `down,mid`, off, 0 | soften settings, as in `soften` (`--soften-r` is `soften`'s `--r`) |
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

### 5. Compare memorized and non-memorized prompts

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

## Code

| file | content |
|---|---|
| `main.py` | command line (`attention`, `guard`, `soften`, `batch`, `compare`); `experiment()` builds the normal / guard / soften setup that all commands share |
| `sampler.py` | loads AnimateDiff and runs the denoising loop for one prompt or a batch of prompts: normal CFG (2 streams) or GUARD / soften with repulsion (3 streams) |
| `attention.py` | the attention processor put in every layer: `spike_scores` / `find_spikes` (detector), `Processor.attenuated` (GUARD), `Processor.softened` (temperature), `Processor.recorded` (stores maps for plots), `select_layers` (layer choice for `soften`) |
| `plots.py` | all figures and the text summaries; `memorization_signals` / `plot_memorization_signals` for section 5 |

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
