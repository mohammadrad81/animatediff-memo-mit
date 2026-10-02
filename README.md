# AnimateDiff attention analysis (towards GUARD-style memorization mitigation)

Record, plot and find spikes in the attention maps of AnimateDiff
(`CompVis/stable-diffusion-v1-4` + `guoyww/animatediff-motion-adapter-v1-4`), following the
spike definition of *"You Don't Need All That Attention"* (GUARD, Zhao et al., ICML 2026).

## Attention types

| type       | modules                          | what attends to what                                           | shape recorded |
|------------|----------------------------------|----------------------------------------------------------------|----------------|
| `spatial`  | `*.attentions.*.attn1`           | each location of a frame -> every location of the same frame    | `[frames, heads, h*w]` |
| `cross`    | `*.attentions.*.attn2`           | each location of a frame -> the 77 text tokens                  | `[frames, heads, 77]` |
| `temporal` | `*.motion_modules.*.attn1/attn2` | each location -> the same location in every frame              | `[heads, frames]`, `[heads, F, F]` |

## Spike definition (GUARD Sec. 5.2, and the official code)

For each attended position `i` (text token / key location / key frame):

```
M_i = max_q AttnScore[q, i]        Z_i = (M_i - mean(M)) / std(M)        spike  <=>  Z_i > tau
```

By default this matches `local_sd_pipeline.py` from the official GUARD repo:
- `--score max_logit`: AttnScore is the pre-softmax logit `QK^T/sqrt(d)`. Use `--score max_prob` to get the softmax probability the paper's text describes.
- Statistics are taken per (batch row, head). In AnimateDiff a cross-attention batch row is one frame, so this means per (frame, head). Use `--head-agg mean|max` and `--frame-agg mean|max` to aggregate before the z-score.
- For cross-attention, `<sot>` (position 0) is left out of mean/std and is never flagged.
- Only the conditional CFG branch is analysed (`--branch uncond` for the other one).

`tau` is applied at analysis time: the raw `M` values are saved, so you can re-run `spikes`/`plot`
with any `--tau` / `--score` without regenerating.

## Install

Your system Python has no `pip`/`venv` yet:

```bash
sudo apt install python3-venv python3-pip          # once
python3 -m venv .venv && source .venv/bin/activate
# RTX 50xx (Blackwell) needs CUDA >= 12.8 wheels:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

All commands: `python -m animatediff_attn <command> --help`.

### 1. See which layers exist

```bash
python -m animatediff_attn list-layers                         # all attention layers, with ids
python -m animatediff_attn list-layers --attn-types temporal --blocks up
```

Every layer gets an `id`; selections can use ids, types, blocks or glob patterns.

### 2. Generate + record + detect spikes + plot

```bash
python -m animatediff_attn run \
    --prompt "Foyer painted in KHAKI" \
    --seed 0 --num-frames 16 --num-inference-steps 25
```

This records all layers. Statistics for cross and temporal attention are kept at every step. Full maps
(and spatial self-attention) are kept only at the first and last step. Output goes to `outputs/<prompt>_seed0/`.

Selecting what to record (all filters are intersected):

```bash
python -m animatediff_attn run --prompt "..." \
    --attn-types cross,temporal --blocks down,mid --layers "*down_blocks.1*" --layer-ids 0:20 \
    --heads 0,3 --frames 0:16:4 \
    --record-steps all --map-steps 0,12,-1 --map-tokens valid \
    --query-points "0.5,0.5;0.2,0.8" \
    --tau 3 --score max_logit
```

| option | meaning |
|---|---|
| `--attn-types` | `spatial`, `cross`, `temporal` |
| `--blocks` | `down`, `mid`, `up`, or a specific block such as `down.1` / `up.3` |
| `--layers` | fnmatch globs on module names |
| `--layer-ids` | ids from `list-layers` (`3,7,11` or python slices like `0:20`) |
| `--heads` / `--frames` | heads / frames stored in full maps and shown in plots |
| `--record-steps` | steps with per-token statistics (cross & temporal) |
| `--map-steps` | steps with full maps, and the only steps where spatial self-attention is analysed |
| `--map-tokens` | cross-attention tokens stored in maps: `valid` (`<sot>`..`<eot>`), `all`, or `1,4,5` |
| `--query-points` | spatial self-attention query points, `y,x` in [0,1], separated by `;` |
| `--tau`, `--score`, `--head-agg`, `--frame-agg` | spike detection (see above) |

Batch of prompts (e.g. the Webster memorized-prompt list): `--prompts-file prompts.txt` (one per line).

Less VRAM: `--cpu-offload`, fewer `--num-frames`, `--height/--width 256`, or fewer `--map-steps`.

### 3. Re-plot / re-analyse a run without regenerating

```bash
python -m animatediff_attn plot   --run-dir outputs/<run> --attn-types cross --plots profile,heads \
                                   --plot-steps 0,-1 --score max_prob --tau 2.5 --heads 0,1
python -m animatediff_attn spikes --run-dir outputs/<run> --tau 3 --head-agg mean --frame-agg max
```

## Outputs (`outputs/<run>/`)

| file | content |
|---|---|
| `video.gif`, `frames/` | generated video |
| `record.pt` | everything recorded (`torch.load(..., weights_only=False)`) |
| `meta.json` | prompt, token labels, `<eot>` index, layers, resolutions, timesteps |
| `spikes_<score>_tau<τ>_h-<agg>_f-<agg>.csv` | one row per spike: layer, step, timestep, frame, head, position, label, M, z |
| `spikes_summary_*.json` | how often each token / key frame / location is flagged |
| `plots/profile/` | M_i and z_i over tokens per cross layer + mean of all layers (paper Fig. 1a/b) |
| `plots/heads/` | z-score heatmaps heads x tokens and frames x tokens, spikes boxed |
| `plots/steps/` | token score over denoising steps (Fig. 1c/d) and per U-Net block group (Fig. 7) |
| `plots/crossmaps/` | per-token cross-attention maps over each frame (`--map-head mean` or an index) |
| `plots/selfmaps/` | spatial self-attention of the query points per frame/head; key-wise spike maps |
| `plots/temporal/` | frame x frame attention per head, key-frame z-scores, per-location maps (`--query-frame`) |
| `plots/summary/` | layers x steps heatmaps of max z and spike counts, per attention type |

## Code layout

```
animatediff_attn/
  selection.py  enumerate / select attention layers, index-list parsing
  recorder.py   attention-processor wrapper that records stats & maps (output of the model is unchanged)
  model.py      pipeline loading, token labels, generation
  spikes.py     GUARD z-score spike detection, CSV / summary
  plotting.py   all plots
  cli.py        command line interface
```
