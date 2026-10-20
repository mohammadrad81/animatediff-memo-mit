"""Plots and text summaries of recorded attention maps and spikes.

Each layer's record is a list over diffusion steps of (attn, peak):
  spatial / cross: attn, peak = [frames, heads, keys]   (keys = pixels / 77 tokens)
  temporal:        attn = [heads, frames, frames] (query frame x key frame), peak = [heads, key frames]
attn = max over queries of the attention probability a key receives (what the attention maps show).
peak = max over queries of the logit; the spike score Z is computed from it, separately for every
       (step, frame, head) row, exactly as GUARD's detector does.
Every diffusion step gets its own figures (out_dir/step_XX/); steps are never collapsed.
Nothing is averaged: every dimension that is collapsed (queries, pixels, frames, heads) is collapsed with the max.

Every figure shows two things side by side:
  attention panels (viridis):  attn, max over the dims that are collapsed. No markers.
  spike panels (blue -> red):  Z, max over the same dims. The colour scale is centred on tau, so red cells
                               are spikes, and they are also marked with a black x (x <=> Z > tau).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from attention import layer_kind, short, spike_scores


def token_labels(tokenizer, prompt):
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True).input_ids
    eot = ids.index(tokenizer.eos_token_id)
    labels = []
    for i, tok in enumerate(tokenizer.convert_ids_to_tokens(ids)):
        tok = "<sot>" if i == 0 else "<eot>" if i == eot else "<pad>" if i > eot else tok.replace("</w>", "")
        labels.append(f"{i}:{tok}")
    return labels, eot


def heatmap(ax, values, ylabel, xlabels=None, ylabels=None, title="", aspect="auto"):
    """Attention values (no spike markers)."""
    im = ax.imshow(values, aspect=aspect, cmap="viridis", interpolation="nearest")
    decorate(ax, im, ylabel, xlabels, ylabels, title)


def zmap(ax, z, tau, ylabel, xlabels=None, ylabels=None, title="", aspect="auto", marker_size=14):
    """Spike scores Z: blue below tau, red above tau, black x on every cell with Z > tau."""
    norm = TwoSlopeNorm(vcenter=tau, vmin=min(z.min(), tau - 1e-3), vmax=max(z.max(), tau + 1e-3))
    im = ax.imshow(z, aspect=aspect, cmap="coolwarm", norm=norm, interpolation="nearest")
    ys, xs = np.nonzero(z > tau)
    ax.scatter(xs, ys, marker="x", color="black", s=marker_size, linewidths=0.8)
    decorate(ax, im, ylabel, xlabels, ylabels, title)


def decorate(ax, im, ylabel, xlabels, ylabels, title):
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=90, fontsize=6)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)


def save(fig, path):
    if fig.get_layout_engine() is None:  # figures made with layout="constrained" arrange themselves
        fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_cross(title, attn, z, tau, labels, path):
    """One step. attn, z: [frames, heads, 77]. Token 0 (<sot>) is left out: it always dominates."""
    fig, axes = plt.subplots(2, 2, figsize=(30, 10))
    for row, (ylabel, dim) in zip(axes, [("head", 0), ("frame", 1)]):  # max over frames / over heads
        other = "frames" if dim == 0 else "heads"
        heatmap(row[0], attn.max(dim)[:, 1:], ylabel, labels[1:], title=f"attention probability, max over {other}")
        zmap(row[1], z.max(dim)[:, 1:], tau, ylabel, labels[1:], title=f"spike score Z, max over {other} (x: Z > {tau})")
    fig.suptitle(title)
    save(fig, path)


def plot_cross_steps(title, attn, labels, steps, path):
    """attn: [steps, heads, 77] (one frame, or max over frames). One panel per head: for every token (x axis),
    its max attention over the query pixels (y axis), one line per diffusion step (colour = step).
    Token 0 (<sot>) is left out: it always dominates."""
    a = attn[:, :, 1:]  # [steps, heads, 76]
    heads, x = a.shape[1], np.arange(a.shape[2])
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(min(steps), max(steps))
    rows = (heads + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(26, 3.8 * rows), squeeze=False, layout="constrained")
    for hd, ax in enumerate(axes.flat):
        if hd >= heads:
            ax.axis("off")
            continue
        for t in steps:
            ax.plot(x, a[t, hd], color=cmap(norm(t)), lw=0.5, marker="o", ms=1)
        ax.set_title(f"head {hd}", fontsize=9)
        ax.set_ylabel("max attention probability")
        ax.set_xticks(x, labels[1:], rotation=90, fontsize=6)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.grid(alpha=0.3)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.02, pad=0.01,
                 label="diffusion step (0 = first, noisiest)")
    fig.suptitle(f"{title}: max cross-attention per token, one line per step")
    save(fig, path)


def mosaic(a, frames, heads, h, w):
    """[frames, heads, h*w] -> one image with a (h x w) cell per (head = row, frame = column)."""
    return a.reshape(frames, heads, h, w).transpose(1, 2, 0, 3).reshape(heads * h, frames * w)


def plot_spatial(title, attn, z, tau, latent_hw, path):
    """One step. attn, z: [frames, heads, pixels]. Top: attention each pixel receives (each cell normalised
    to [0, 1]). Bottom: spike score of each pixel. Nothing is collapsed."""
    frames, heads, pixels = attn.shape
    ratio = (latent_hw[0] * latent_hw[1] / pixels) ** 0.5
    h = round(latent_hw[0] / ratio)
    w = pixels // h
    m = attn / attn.max(-1, keepdims=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(frames * 1.2 + 2, heads * 2.4 + 2))
    heatmap(a, mosaic(m, frames, heads, h, w), "", title=f"{title}: attention received per pixel ({h}x{w})", aspect="equal")
    zmap(b, mosaic(z, frames, heads, h, w), tau, "", title=f"{title}: spike score Z (x: Z > {tau})",
         aspect="equal", marker_size=4)
    for ax in (a, b):
        ax.set_xticks(np.arange(frames) * w + w / 2, [f"frame {i}" for i in range(frames)], fontsize=7)
        ax.set_yticks(np.arange(heads) * h + h / 2, [f"head {i}" for i in range(heads)], fontsize=7)
        for i in range(1, frames):
            ax.axvline(i * w - 0.5, color="white", lw=0.5)
        for i in range(1, heads):
            ax.axhline(i * h - 0.5, color="white", lw=0.5)
    save(fig, path)


def plot_temporal(title, attn, z, tau, path):
    """One step. attn: [heads, F, F] (query frame x key frame), z: [heads, F] (per key frame)."""
    heads = attn.shape[0]
    fig, axes = plt.subplots(1, heads + 1, figsize=(3.4 * (heads + 1), 3.8))
    for hd in range(heads):
        heatmap(axes[hd], attn[hd], "query frame", title=f"head {hd}: attention")
        axes[hd].set_xlabel("key frame", fontsize=8)
    zmap(axes[-1], z, tau, "head", title="spike score Z")
    axes[-1].set_xlabel("key frame", fontsize=8)
    fig.suptitle(f"{title}: frame-to-frame attention (max over pixel locations); x: Z > {tau}")
    save(fig, path)


def plot_all(maps, labels, eot, tau, latent_hw, out_dir, steps=None):
    """Make the figures of every diffusion step (or only `steps`) in out_dir/step_XX/, the token-by-step line
    charts in out_dir/cross_steps/ (max over frames) and out_dir/cross_steps_frames/<layer>/ (one per frame),
    plus spikes.txt and cross_summary.png in out_dir."""
    num_steps = len(next(iter(maps.values())))
    steps = range(num_steps) if steps is None else [t for t in steps if 0 <= t < num_steps]
    step_dir = {t: os.path.join(out_dir, f"step_{t:02d}") for t in steps}
    for d in step_dir.values():
        for kind in ["cross", "spatial", "temporal"]:
            os.makedirs(os.path.join(d, kind), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cross_steps"), exist_ok=True)
    # per step and kind: layer names, attention rows, Z rows (for the all-layers overview)
    overview = {t: {"cross": ([], [], []), "temporal": ([], [], [])} for t in steps}
    lines = [f"Spike: Z > tau={tau} in at least one (step, frame, head) row; (n/N) = spike in n of the N rows\n"]

    for name, record in maps.items():
        kind = layer_kind(name)
        attn = torch.stack([a for a, _ in record]).float().numpy()  # [steps, ...]
        peak = torch.stack([p for _, p in record])
        z = spike_scores(peak, skip_first=(kind == "cross")).numpy()
        print(f"plotting {short(name)} for {len(steps)} steps")
        if kind == "cross":  # all steps in one figure: how each token's attention changes over the steps
            plot_cross_steps(f"{name} (max over frames and query pixels)", attn.max(1), labels, steps,
                             os.path.join(out_dir, "cross_steps", short(name) + ".png"))
            frame_dir = os.path.join(out_dir, "cross_steps_frames", short(name))
            os.makedirs(frame_dir, exist_ok=True)
            for f in range(attn.shape[1]):  # the same chart for every frame
                plot_cross_steps(f"{name}, frame {f} (max over query pixels)", attn[:, f], labels, steps,
                                 os.path.join(frame_dir, f"frame_{f:02d}.png"))

        for t in steps:
            title = f"{name}, step {t}"
            path = os.path.join(step_dir[t], kind, short(name) + ".png")
            if kind == "cross":
                plot_cross(title, attn[t], z[t], tau, labels, path)
            elif kind == "spatial":
                plot_spatial(title, attn[t], z[t], tau, latent_hw, path)
            else:
                plot_temporal(title, attn[t], z[t], tau, path)
            if kind in ("cross", "temporal"):  # one overview row: max over frames and heads (cross)
                names, a, zs = overview[t][kind]  #                or heads and query frames (temporal)
                names.append(short(name))
                a.append(attn[t].max((0, 1)))
                zs.append(z[t].max(0) if kind == "temporal" else z[t].max((0, 1)))

        # Text summary over all steps: count rows, nothing is averaged.
        rows = (0, 1) if kind == "temporal" else (0, 1, 2)  # (steps, heads) or (steps, frames, heads)
        count = (z > tau).sum(rows)
        total = int(np.prod([z.shape[d] for d in rows]))
        if kind == "spatial":
            found = [f"{int((count > 0).sum())} of {count.size} pixel locations"]
        else:
            key = labels if kind == "cross" else [f"frame {i}" for i in range(count.size)]
            found = [f"{key[i]} ({count[i]}/{total})" for i in np.nonzero(count)[0]]
        lines.append(f"[{kind:8s}] {name}: " + (", ".join(found) or "-"))

    # Per step, one figure per kind with all layers (rows = layers).
    for t in steps:
        for kind, (names, m, zs) in overview[t].items():
            if not names:
                continue
            m, zs = np.stack(m), np.stack(zs)
            xl = labels[1:] if kind == "cross" else [f"f{i}" for i in range(m.shape[1])]
            if kind == "cross":
                m, zs = m[:, 1:], zs[:, 1:]
            m = m / m.max(1, keepdims=True)  # layers have very different ranges: normalise each row
            other = "frames/heads" if kind == "cross" else "heads/query frames"
            fig, (a, b) = plt.subplots(2, 1, figsize=(16 if kind == "cross" else 9, 0.7 * len(names) + 4))
            heatmap(a, m, "layer", xl, names, title=f"step {t}, {kind}: attention per key, max over {other} (each layer / its max)")
            zmap(b, zs, tau, "layer", xl, names, title=f"step {t}, {kind}: spike score Z, max over {other} (x: Z > {tau})")
            save(fig, os.path.join(step_dir[t], f"{kind}_overview.png"))

    if any(layer_kind(n) == "cross" for n in maps):
        plot_cross_summary(maps, labels, eot, os.path.join(out_dir, "cross_summary.png"))
    with open(os.path.join(out_dir, "spikes.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"spike list written to {os.path.join(out_dir, 'spikes.txt')}")


def plot_cross_summary(maps, labels, eot, path):
    """Like Figure 1 of the paper: cross-attention per token in the first and last step,
    and the attention on <eot> over the diffusion steps (max over layers, frames, heads, queries)."""
    attn = np.stack([np.stack([a.float().numpy().max((0, 1)) for a, _ in rec])
                     for name, rec in maps.items() if layer_kind(name) == "cross"]).max(0)  # [steps, 77]
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(1, attn.shape[1])
    a.plot(x, attn[0, 1:], label="first step")
    a.plot(x, attn[-1, 1:], label="last step")
    a.axvline(eot, ls="--", color="gray", label="<eot>")
    a.set_xticks(x, labels[1:], rotation=90, fontsize=6)
    a.set_ylabel("max cross-attention probability")
    a.legend()
    b.plot(attn[:, eot])
    b.set_xlabel("diffusion step")
    b.set_title("max attention on <eot>")
    save(fig, path)


def write_guard_summary(spikes, labels, path):
    """spikes: layer -> list over steps of ([77] count of attenuated (frame, head) rows, number of rows)."""
    lines = ["In how many (step, frame, head) rows GUARD attenuated each token: n/N\n"]
    all_counts, all_rows = 0, 0
    for name, record in spikes.items():
        counts = torch.stack([c for c, _ in record]).sum(0).numpy()
        rows = sum(n for _, n in record)
        all_counts, all_rows = all_counts + counts, all_rows + rows
        top = [f"{labels[i]} {counts[i]}/{rows}" for i in np.argsort(-counts)[:8] if counts[i] > 0]
        lines.append(f"{name}: " + (", ".join(top) or "-"))
    if spikes:
        lines.append("\nAll attenuated layers together:")
        lines += [f"  {labels[i]:24s} {all_counts[i]}/{all_rows}" for i in range(len(labels)) if all_counts[i] > 0]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_soften_summary(stats, layers, temperature, keep_sot, path):
    """stats: layer -> list over steps of {"entropy"/"top"/"sot": (before, after)}, each a mean over
    (frames or pixel locations, heads, queries). Values are averaged over the steps here."""
    lines = [f"Softened layers (temperature T = {temperature}{', <sot> kept' if keep_sot else ''}), stream 1 only.",
             "Mean over steps, frames/locations, heads and queries, before -> after softening:",
             "  entropy = entropy of the attention distribution (nats)",
             "  top     = largest attention probability of a key (cross: <sot> excluded)",
             "  sot     = attention on <sot> (cross only)\n"]
    for name in layers:
        if name not in stats:
            lines.append(f"{short(name):28s} (spatial: fused kernel, no statistics)")
            continue
        mean = {key: np.mean([s[key] for s in stats[name]], axis=0) for key in stats[name][0]}
        cols = [f"{key} {b:.3f} -> {a:.3f}" for key, (b, a) in mean.items() if not np.isnan(b)]
        lines.append(f"{short(name):28s} " + "   ".join(cols))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def memorization_signals(ctrl, frames, prompt):
    """The numbers used to compare memorized and non-memorized prompts, as a JSON-friendly dict.
    guidance:    [steps][frames] ||eps_cond - eps_null|| (Wen et al.: large for memorized prompts)
    eot_share:   block -> [steps] share of cross-attention on <eot> + padding tokens, mean over the layers of the
                 block, frames, heads and query pixels (our extension of Kim & No, CVPR 2026 Findings, and
                 Chen et al., ICLR 2025, who link the <eot> / padding tokens to memorization)
    frame_change:     [frames-1] mean |pixel difference| between consecutive frames (pixels in 0..1)
    frame_similarity: [frames] cosine similarity of every frame to frame 0"""
    eot = {}
    for name, record in ctrl.eot_share.items():
        eot.setdefault(name.split("_")[0], []).append(torch.stack(record))  # [steps, frames, heads]
    video = np.stack([np.asarray(f, dtype=np.float32) / 255 for f in frames]).reshape(len(frames), -1)
    unit = video / np.linalg.norm(video, axis=1, keepdims=True)
    return {"prompt": prompt,
            "guidance": torch.stack(ctrl.guidance)[:, 0].tolist(),
            "eot_share": {b: torch.stack(v).mean((0, 2, 3)).tolist() for b, v in eot.items()},
            "frame_change": np.abs(np.diff(video, axis=0)).mean(1).tolist(),
            "frame_similarity": (unit @ unit[0]).tolist()}


def plot_memorization_signals(runs, path):
    """runs: list of (label, signals) from memorization_signals(). One line per run in every panel."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), layout="constrained")
    colors = plt.get_cmap("tab10")
    for c, (label, sig) in enumerate(runs):
        color = colors(c % 10)
        g = np.array(sig["guidance"])  # [steps, frames]: line = max over frames, band = min..max over frames
        axes[0, 0].plot(g.max(1), color=color, label=label)
        axes[0, 0].fill_between(range(len(g)), g.min(1), g.max(1), color=color, alpha=0.15)
        for ax, block in zip([axes[0, 1], axes[0, 2], axes[1, 0]], ["down", "mid", "up"]):
            if block in sig["eot_share"]:
                ax.plot(sig["eot_share"][block], color=color, label=label)
        axes[1, 1].plot(range(1, len(sig["frame_change"]) + 1), sig["frame_change"], color=color, marker="o", ms=3, label=label)
        axes[1, 2].plot(sig["frame_similarity"], color=color, marker="o", ms=3, label=label)
    titles = [
        ("text guidance strength ||eps_cond - eps_null||\nline = max over frames, band = min..max (high = memorized)", "diffusion step"),
        ("share of cross-attention on <eot> + padding, down blocks\n(mean over layers, frames, heads, pixels; stays high = memorized)", "diffusion step"),
        ("share of cross-attention on <eot> + padding, mid block", "diffusion step"),
        ("share of cross-attention on <eot> + padding, up blocks", "diffusion step"),
        ("change between consecutive frames (mean |pixel difference|)", "frame f vs f-1"),
        ("similarity of each frame to frame 0 (cosine)", "frame"),
    ]
    for ax, (title, xlabel) in zip(axes.flat, titles):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    save(fig, path)
