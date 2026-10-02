"""Plots of recorded attention. Every function writes PNG files and returns their paths.

Plot kinds (``--plots``):
  profile    per cross-attn layer (+ all-layer aggregate): M_i over text tokens, spikes marked (cf. paper Fig. 1a/b)
  heads      per cross-attn layer & step: z-score heatmaps heads x tokens and frames x tokens
  steps      score of selected tokens over denoising steps (Fig. 1c/d) and per U-Net block group (Fig. 7)
  crossmaps  per cross-attn layer & map step: spatial maps of each token, per frame, over the frame image
  selfmaps   per spatial self-attn layer & map step: maps of the query points + key-wise spike maps
  temporal   per temporal layer & step: frame x frame attention per head, key-frame spikes, spatial maps
  summary    per attention type: layers x steps heatmaps of max z-score and spike counts
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .spikes import detect, exclude_first_for, position_scores  # noqa: E402

PLOT_KINDS = ("profile", "heads", "steps", "crossmaps", "selfmaps", "temporal", "summary")

# Palette: fixed-order categorical hues, a single-hue blue ramp for magnitude and a
# blue <-> red diverging map with a neutral gray midpoint for z-scores.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb"] + BLUE_RAMP)
OVERLAY = LinearSegmentedColormap.from_list("overlay_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ee8f8e", "#b52f2e"])
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3de"
SPIKE_STYLE = dict(marker="o", s=70, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
MAX_GRID_ROWS = 8

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "lines.linewidth": 2,
})


def ordinal_colors(n: int) -> list[str]:
    """Steps are ordered: light (early) -> dark (late) on the blue ramp."""
    if n == 1:
        return [BLUE_RAMP[4]]
    idx = np.linspace(1, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def _names_of(record, names, attn_type):
    return [n for n in names if record["layers"][n]["attn_type"] == attn_type]


def _short(record, name):
    info = record["layers"][name]
    return f"L{info['idx']:02d}_{info['attn_type']}_{info['block']}_{name.split('.')[-1]}"


def _title(record, name):
    info = record["layers"][name]
    return f"[{info['idx']}] {name}  ({info['attn_type']}, {info['block']})"


def _thin(seq, k=MAX_GRID_ROWS):
    if len(seq) <= k:
        return list(range(len(seq)))
    return sorted(set(np.linspace(0, len(seq) - 1, k).round().astype(int).tolist()))


def _token_ticks(ax, labels, eot, positions):
    ticks, texts = [], []
    for p in positions:
        if p <= eot or p % 10 == 0:
            ticks.append(p)
            texts.append(f"{labels[p]} ({p})" if p <= eot else str(p))
    ax.set_xticks(ticks)
    ax.set_xticklabels(texts, rotation=90, fontsize=7)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _upsample(m: torch.Tensor, size) -> np.ndarray:
    return F.interpolate(m.float()[None, None], size=size, mode="bilinear", align_corners=False)[0, 0].numpy()


def _spike_legend(ax, tau):
    ax.scatter([], [], label=f"spike (z > {tau:g})", **SPIKE_STYLE)
    ax.legend(loc="upper right", fontsize=7, frameon=False)


# --------------------------------------------------------------------------- profile
def plot_profile(record, names, steps, out, score="max_logit", tau=3.0, heads=None, frames=None, hide_first=True):
    labels, eot = record["meta"]["token_labels"], record["meta"]["eot_index"]
    cross = _names_of(record, names, "cross")
    paths = []
    targets = [(n, [n]) for n in cross] + ([("ALL", cross)] if len(cross) > 1 else [])
    for tag, group in targets:
        prof = {}
        for s in steps:
            ms = [position_scores(record["stats"][n][s], "cross", score, heads, frames, "mean", "mean")[0, 0]
                  for n in group if s in record["stats"][n]]
            if ms:
                prof[s] = torch.stack(ms).mean(0)
        if not prof:
            continue
        start = 1 if hide_first else 0
        pos = np.arange(start, len(labels))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for color, (s, M) in zip(ordinal_colors(len(prof)), prof.items()):
            z, spk = detect(M, tau, exclude_first=True)
            lbl = f"step {s} (t={record['timesteps'][s]:.0f})"
            a1.plot(pos, M[start:].numpy(), color=color, label=lbl)
            a2.plot(pos, z[start:].numpy(), color=color, label=lbl)
            sp = spk.nonzero().flatten().numpy()
            a1.scatter(sp, M[sp].numpy(), **SPIKE_STYLE)
            a2.scatter(sp, z[sp].numpy(), **SPIKE_STYLE)
        a1.scatter([], [], label=f"spike (z > {tau:g})", **SPIKE_STYLE)
        a1.axvline(eot, color=MUTED, ls=":", lw=1)
        a2.axvline(eot, color=MUTED, ls=":", lw=1)
        a2.axhline(tau, color=MUTED, ls="--", lw=1, label=f"tau = {tau:g}")
        a1.set_ylabel(f"{score} (mean over heads/frames)")
        a2.set_ylabel("z-score")
        a1.legend(fontsize=7, frameon=False, ncol=2)
        for a in (a1, a2):
            a.grid(axis="y")
        _token_ticks(a2, labels, eot, pos)
        title = f"Cross-attention token profile, mean of {len(group)} layers" if tag == "ALL" else _title(record, tag)
        a1.set_title(f"{title}\nprompt: {record['meta']['prompt']}", loc="left")
        fname = "ALL" if tag == "ALL" else _short(record, tag)
        paths.append(_save(fig, out / "profile" / f"{fname}_{score}.png"))
    return paths


# ----------------------------------------------------------------------------- heads
def plot_heads(record, names, steps, out, score="max_logit", tau=3.0, heads=None, frames=None):
    labels, eot = record["meta"]["token_labels"], record["meta"]["eot_index"]
    paths = []
    for n in _names_of(record, names, "cross"):
        for s in steps:
            stats = record["stats"][n].get(s)
            if stats is None:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(16, 8))
            for ax, (h_agg, f_agg, ylab) in zip(axes, [("none", "mean", "head"), ("mean", "none", "frame")]):
                M = position_scores(stats, "cross", score, heads, frames, h_agg, f_agg)
                z, spk = detect(M, tau, exclude_first=True)
                z2d = (z[0] if f_agg == "mean" else z[:, 0]).numpy()
                spk2d = (spk[0] if f_agg == "mean" else spk[:, 0]).numpy()
                z2d[:, 0] = np.nan  # <sot> is excluded from the statistics
                vmax = max(np.nanmax(np.abs(z2d)), tau + 0.1)
                im = ax.imshow(z2d, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest")
                ys, xs = np.nonzero(spk2d)
                ax.scatter(xs, ys, **{**SPIKE_STYLE, "s": 40, "marker": "s"})
                ax.scatter([], [], label=f"spike (z > {tau:g})", **{**SPIKE_STYLE, "s": 40, "marker": "s"})
                ax.legend(loc="upper right", fontsize=7, frameon=False)
                ids = heads if ylab == "head" else frames
                ticks = list(range(z2d.shape[0]))
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(ids[i]) if ids else str(i) for i in ticks], fontsize=7)
                ax.set_ylabel(ylab + (" (frames averaged)" if ylab == "head" else " (heads averaged)"))
                _token_ticks(ax, labels, eot, range(z2d.shape[1]))
                fig.colorbar(im, ax=ax, label="z-score", fraction=0.02)
            axes[0].set_title(f"{_title(record, n)} - step {s}, z-score of {score}", loc="left")
            paths.append(_save(fig, out / "heads" / f"{_short(record, n)}_step{s:03d}.png"))
    return paths


# ----------------------------------------------------------------------------- steps
def plot_steps(record, names, out, score="mass", tau=3.0, tokens=None, heads=None, frames=None):
    labels, eot = record["meta"]["token_labels"], record["meta"]["eot_index"]
    cross = _names_of(record, names, "cross")
    if not cross:
        return []
    steps = sorted({s for n in cross for s in record["stats"][n]})
    agg = {}
    for s in steps:
        ms = [position_scores(record["stats"][n][s], "cross", score, heads, frames, "mean", "mean")[0, 0]
              for n in cross if s in record["stats"][n]]
        agg[s] = torch.stack(ms).mean(0)
    if tokens is None:  # <eot> plus every token that is a spike of the aggregate profile at some step
        tokens = [eot]
        for s in steps:
            _, spk = detect(agg[s], tau, exclude_first=True)
            tokens += [p for p in spk.nonzero().flatten().tolist() if p not in tokens]
    tokens = tokens[: len(CATEGORICAL)]
    paths = []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for color, p in zip(CATEGORICAL, tokens):
        ys = [agg[s][p].item() for s in steps]
        ax.plot(steps, ys, color=color, label=f"{labels[p]} ({p})")
        ax.annotate(f"{labels[p]}", (steps[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=INK, va="center")
    ax.set_xlabel("inference step")
    ax.set_ylabel(f"{score} (mean over {len(cross)} layers, heads, frames)")
    ax.grid(axis="y")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Cross-attention {score} per token over denoising\nprompt: {record['meta']['prompt']}", loc="left")
    paths.append(_save(fig, out / "steps" / f"tokens_over_steps_{score}.png"))

    groups = {}
    for n in cross:
        groups.setdefault(record["layers"][n]["block"].split(".")[0], []).append(n)
    fig, axes = plt.subplots(1, len(tokens), figsize=(4.5 * len(tokens), 3.8), squeeze=False, sharey=False)
    for ax, p in zip(axes[0], tokens):
        for color, (g, members) in zip(CATEGORICAL, sorted(groups.items())):
            ys = []
            for s in steps:
                vals = [position_scores(record["stats"][n][s], "cross", score, heads, frames, "mean", "mean")[0, 0, p]
                        for n in members if s in record["stats"][n]]
                ys.append(torch.stack(vals).mean().item())
            ax.plot(steps, ys, color=color, label=f"{g} blocks")
        ax.set_title(f"token {labels[p]} ({p})", loc="left")
        ax.set_xlabel("inference step")
        ax.grid(axis="y")
    axes[0][0].set_ylabel(f"{score}")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Cross-attention per U-Net block group (cf. paper Fig. 7)", x=0.01, ha="left")
    paths.append(_save(fig, out / "steps" / f"block_groups_{score}.png"))
    return paths


# ------------------------------------------------------------------------- crossmaps
def plot_crossmaps(record, names, steps, out, frame_images=None, head="mean"):
    labels = record["meta"]["token_labels"]
    cfg = record["config"]
    tokens = cfg["map_tokens"] or list(range(len(labels)))
    paths = []
    for n in _names_of(record, names, "cross"):
        for s in steps:
            maps = record["maps"][n].get(s)
            if maps is None:
                continue
            probs = maps["probs"].float()  # [F', H', h, w, T]
            stored_heads = cfg["heads"] or list(range(probs.shape[1]))
            if head == "mean":
                m = probs.mean(1)
            else:
                m = probs[:, stored_heads.index(int(head))]
            frames = cfg["frames"] or list(range(probs.shape[0]))
            rows = _thin(frames)
            cols = len(tokens)
            fig, axes = plt.subplots(len(rows), cols + 1, figsize=(1.6 * (cols + 1), 1.7 * len(rows)), squeeze=False)
            vmax = m.amax().item()
            for r, fi in enumerate(rows):
                img = frame_images[frames[fi]] if frame_images else None
                ax = axes[r][0]
                if img is not None:
                    ax.imshow(img)
                ax.set_ylabel(f"frame {frames[fi]}", fontsize=7)
                for c, tok in enumerate(tokens):
                    ax = axes[r][c + 1]
                    size = (img.shape[0], img.shape[1]) if img is not None else m.shape[1:3]
                    heat = _upsample(m[fi, :, :, c], size)
                    if img is not None:
                        ax.imshow(img)
                        ax.imshow(heat, cmap=OVERLAY, alpha=0.65, vmin=0, vmax=vmax)
                    else:
                        ax.imshow(heat, cmap=SEQ, vmin=0, vmax=vmax)
                    if r == 0:
                        ax.set_title(f"{labels[tok]} ({tok})", fontsize=7)
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{_title(record, n)} - step {s}, head {head}, attention prob per token", x=0.01, ha="left")
            paths.append(_save(fig, out / "crossmaps" / f"{_short(record, n)}_step{s:03d}_head-{head}.png"))
    return paths


# -------------------------------------------------------------------------- selfmaps
def plot_selfmaps(record, names, steps, out, score="max_prob", tau=3.0, frame_images=None):
    cfg = record["config"]
    paths = []
    for n in _names_of(record, names, "spatial"):
        h, w = record["resolutions"][n]
        for s in steps:
            maps, stats = record["maps"][n].get(s), record["stats"][n].get(s)
            if maps is None or stats is None:
                continue
            qm = maps["query_maps"].float()  # [F', H', P, h, w]
            frames = cfg["frames"] or list(range(qm.shape[0]))
            heads = cfg["heads"] or list(range(qm.shape[1]))
            rows = _thin(frames)
            for pi, (qy, qx) in enumerate(cfg["query_points"]):
                qidx = int(maps["query_index"][pi])
                fig, axes = plt.subplots(len(rows), len(heads) + 2, figsize=(1.7 * (len(heads) + 2), 1.7 * len(rows)),
                                         squeeze=False)
                for r, fi in enumerate(rows):
                    img = frame_images[frames[fi]] if frame_images else None
                    if img is not None:
                        axes[r][0].imshow(img)
                        axes[r][0].scatter([qx * (img.shape[1] - 1)], [qy * (img.shape[0] - 1)], marker="+", s=80,
                                           c=CATEGORICAL[1], linewidths=2)
                    axes[r][0].set_ylabel(f"frame {frames[fi]}", fontsize=7)
                    panels = [qm[fi, hi, pi] for hi in range(len(heads))] + [qm[fi, :, pi].mean(0)]
                    for c, panel in enumerate(panels):
                        ax = axes[r][c + 1]
                        ax.imshow(panel.numpy(), cmap=SEQ, interpolation="nearest")
                        ax.scatter([qidx % w], [qidx // w], marker="+", s=50, c=CATEGORICAL[1], linewidths=1.5)
                        if r == 0:
                            ax.set_title(f"head {heads[c]}" if c < len(heads) else "head mean", fontsize=7)
                for ax in axes.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.suptitle(f"{_title(record, n)} - step {s}, query point (y={qy:g}, x={qx:g}) +", x=0.01, ha="left")
                paths.append(_save(fig, out / "selfmaps" / f"{_short(record, n)}_step{s:03d}_query{pi}.png"))

            # key-wise spike maps: M_k = max_q score[q, k], z over keys, per (frame, head)
            M = position_scores(stats, "spatial", score)  # [F, H, N]
            z, spk = detect(M, tau, exclude_first=False)
            all_frames = list(range(M.shape[0]))
            rows = _thin(all_frames)
            n_heads = M.shape[1]
            fig, axes = plt.subplots(len(rows), n_heads, figsize=(1.7 * n_heads, 1.7 * len(rows)), squeeze=False)
            for r, f in enumerate(rows):
                for hi in range(n_heads):
                    ax = axes[r][hi]
                    ax.imshow(M[f, hi].reshape(h, w).numpy(), cmap=SEQ, interpolation="nearest")
                    ys, xs = np.divmod(spk[f, hi].nonzero().flatten().numpy(), w)
                    ax.scatter(xs, ys, **{**SPIKE_STYLE, "s": 18, "linewidths": 1})
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r == 0:
                        ax.set_title(f"head {hi}", fontsize=7)
                    if hi == 0:
                        ax.set_ylabel(f"frame {f}", fontsize=7)
            fig.suptitle(f"{_title(record, n)} - step {s}, key-wise {score} (max over queries); "
                         f"o = spike (z > {tau:g})", x=0.01, ha="left")
            paths.append(_save(fig, out / "selfmaps" / f"{_short(record, n)}_step{s:03d}_keyspikes_{score}.png"))
    return paths


# -------------------------------------------------------------------------- temporal
def plot_temporal(record, names, steps, out, score="max_logit", tau=3.0, query_frame=None):
    cfg = record["config"]
    paths = []
    for n in _names_of(record, names, "temporal"):
        h, w = record["resolutions"].get(n, (1, 1))
        for s in steps:
            stats = record["stats"][n].get(s)
            if stats is None:
                continue
            mean_map = stats["mean_map"]  # [H, F, F]
            n_heads, n_frames = mean_map.shape[:2]
            panels = [mean_map[i] for i in range(n_heads)] + [mean_map.mean(0)]
            ncols = min(5, len(panels))
            nrows = int(np.ceil(len(panels) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols + 1, 2.4 * nrows + 0.6), squeeze=False,
                                     layout="constrained")
            vmax = mean_map.amax().item()
            for i, ax in enumerate(axes.flat):
                if i >= len(panels):
                    ax.axis("off")
                    continue
                im = ax.imshow(panels[i].numpy(), cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_title(f"head {i}" if i < n_heads else "head mean", fontsize=8)
                ax.set_xlabel("key frame", fontsize=7)
                if i % ncols == 0:
                    ax.set_ylabel("query frame", fontsize=7)
            fig.colorbar(im, ax=axes.ravel().tolist(), label="attention prob", shrink=0.8)
            fig.suptitle(f"{_title(record, n)} - step {s}, frame-to-frame attention (mean over locations)",
                         x=0.01, ha="left")
            paths.append(_save(fig, out / "temporal" / f"{_short(record, n)}_step{s:03d}_matrices.png"))

            # key-frame spikes: M_j = max over (locations, query frames), z over key frames, per head
            M = position_scores(stats, "temporal", score)[0]  # [H, F]
            z, spk = detect(M, tau, exclude_first=False)
            fig, ax = plt.subplots(figsize=(8, 0.4 * n_heads + 1.5))
            vmax = max(z.abs().max().item(), tau + 0.1)
            im = ax.imshow(z.numpy(), cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto", interpolation="nearest")
            ys, xs = np.nonzero(spk.numpy())
            ax.scatter(xs, ys, **{**SPIKE_STYLE, "s": 40, "marker": "s"})
            _spike_legend(ax, tau)
            ax.set_xlabel("key frame")
            ax.set_ylabel("head")
            fig.colorbar(im, ax=ax, label="z-score", fraction=0.03)
            ax.set_title(f"{_title(record, n)} - step {s}, z-score of key-frame {score}", loc="left")
            paths.append(_save(fig, out / "temporal" / f"{_short(record, n)}_step{s:03d}_keyframe_z.png"))

            maps = record["maps"][n].get(s)
            if maps is None:
                continue
            probs = maps["probs"].float().mean(0)  # [h, w, Fq, Fk], head mean
            qf = n_frames // 2 if query_frame is None else query_frame
            cols = _thin(list(range(n_frames)), 16)
            fig, axes = plt.subplots(2, len(cols), figsize=(1.6 * len(cols), 3.6), squeeze=False)
            vmax = probs[:, :, qf].amax().item()
            for c, kf in enumerate(cols):
                axes[0][c].imshow(probs[:, :, qf, kf].numpy(), cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
                axes[0][c].set_title(f"key {kf}", fontsize=7)
                axes[1][c].imshow(probs[:, :, kf, kf].numpy(), cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
                axes[1][c].set_title(f"self {kf}->{kf}", fontsize=7)
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0][0].set_ylabel(f"query frame {qf}", fontsize=7)
            axes[1][0].set_ylabel("own-frame weight", fontsize=7)
            fig.suptitle(f"{_title(record, n)} - step {s}, per-location temporal attention (head mean, {h}x{w})",
                         x=0.01, ha="left")
            paths.append(_save(fig, out / "temporal" / f"{_short(record, n)}_step{s:03d}_spatial.png"))
    return paths


# --------------------------------------------------------------------------- summary
def plot_summary(record, names, out, score="max_logit", tau=3.0, heads=None, frames=None):
    paths = []
    for attn_type in ("cross", "temporal", "spatial"):
        group = _names_of(record, names, attn_type)
        if not group:
            continue
        steps = sorted({s for n in group for s in record["stats"][n]})
        zmax = np.full((len(group), len(steps)), np.nan)
        frac = np.full_like(zmax, np.nan)
        for i, n in enumerate(group):
            for j, s in enumerate(steps):
                stats = record["stats"][n].get(s)
                if stats is None:
                    continue
                M = position_scores(stats, attn_type, score, heads, frames)
                z, spk = detect(M, tau, exclude_first_for(attn_type))
                if exclude_first_for(attn_type):
                    z = z[..., 1:]
                zmax[i, j] = z.max().item()
                frac[i, j] = spk.float().sum(-1).mean().item()  # spikes per (frame, head)
        fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * len(group) + 2))
        for ax, data, label in [(axes[0], zmax, "max z-score"), (axes[1], frac, "spikes per (frame, head)")]:
            im = ax.imshow(data, aspect="auto", cmap=SEQ, interpolation="nearest")
            ax.set_yticks(range(len(group)))
            ax.set_yticklabels([f"[{record['layers'][n]['idx']}] {record['layers'][n]['block']}" for n in group], fontsize=7)
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(steps, fontsize=6)
            ax.set_xlabel("inference step")
            ax.set_title(label, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.suptitle(f"{attn_type} attention - {score}, tau = {tau:g}", x=0.01, ha="left")
        paths.append(_save(fig, out / "summary" / f"{attn_type}_{score}.png"))
    return paths


def load_frame_images(run_dir: Path):
    files = sorted((run_dir / "frames").glob("frame_*.png"))
    if not files:
        return None
    from PIL import Image

    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def make_plots(record, run_dir: Path, names, kinds, steps=None, score="max_logit", tau=3.0, heads=None, frames=None,
               tokens=None, map_head="mean", query_frame=None):
    out = run_dir / "plots"
    imgs = load_frame_images(run_dir)
    map_steps = sorted({s for n in names for s in record["maps"][n]})
    plot_steps_ = steps if steps is not None else map_steps
    paths = []
    if "profile" in kinds:
        paths += plot_profile(record, names, plot_steps_, out, score, tau, heads, frames)
    if "heads" in kinds:
        paths += plot_heads(record, names, plot_steps_, out, score, tau, heads, frames)
    if "steps" in kinds:
        paths += plot_steps(record, names, out, "mass", tau, tokens, heads, frames)
        paths += plot_steps(record, names, out, score, tau, tokens, heads, frames)
    if "crossmaps" in kinds:
        paths += plot_crossmaps(record, names, plot_steps_, out, imgs, map_head)
    if "selfmaps" in kinds:
        paths += plot_selfmaps(record, names, plot_steps_, out, "max_prob", tau, imgs)
    if "temporal" in kinds:
        paths += plot_temporal(record, names, plot_steps_, out, score, tau, query_frame)
    if "summary" in kinds:
        paths += plot_summary(record, names, out, score, tau, heads, frames)
    return paths
