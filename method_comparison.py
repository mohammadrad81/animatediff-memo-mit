"""Compare the mitigation methods on memorization and video quality together: spider plots, 2D trade-off plots, one
summary scatter, content against motion, and a scorecard. See section 10 of README.md (step 9).

It reads only the results of the earlier steps (no GPU, no videos):
  <out>/metrics[_<method>].csv                memorization_eval.py metrics   (GSSCD, OFS-k, OFS-k with the filter)
  <out>/quality[_<method>].csv, i3d_features[_<method>].npy     quality_eval.py metrics   (ViCLIP, I3D for FVD)
  <original>/metrics.csv, quality.csv, i3d_features.npy         the original model: the memorization reference
  <quality>/fvd_reference.npz, i3d_training.npy, quality_training.csv     quality_eval.py reference

  python method_comparison.py            figures and tradeoff_summary.csv in outputs/finetune_tradeoff

Every figure is drawn twice, with two ways of measuring the memorization of the memorized samples:
  *_threshold   the percentage of memorized samples above the paper's threshold (GSSCD >= 0.4, OFS > 0.5)
  *_mean        the mean score (GSSCD, OFS) of the memorized samples

The seven axes, all scaled so that higher is better:
  memorization (memorized samples): (no mitigation - method) / (no mitigation - original model)
      0 = copies as much as without mitigation, 1 = as little as the original model, which never saw the duplicates
      (above 1: less than the original model, which usually means the video lost its content)
  quality: the fraction of the quality without mitigation that the method keeps, 1 = nothing lost
      ViCLIP: method / no mitigation;  FVD: no mitigation / method (FVD is lower-is-better)
      for the memorized samples and, as the side effect, for the background samples
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd

from memorization_eval import (EXPERIMENTS, GRID, INK, MEMORIZED_COLOR, MUTED, NO_MITIGATION_COLOR, ORIGINAL_COLOR,
                               ORIGINAL_DEFAULT, pyplot, read_metrics, result_file, save_figures)
from quality_eval import frechet_distance

METHOD_COLOR = MEMORIZED_COLOR  # a method, as in mitigation_score_vs_copies
# Methods drawn together on one spider plot (any number). Each family of methods (a method and its variants, e.g. noise
# and noise_above) gets one color, in this fixed order: the reference categorical palette, whose first three colors stay
# apart for color-blind readers. Within a family, the variants differ by line style (the base method solid, the others
# dashed, dotted, ... with hollow markers). After eight families the colors repeat with another marker shape.
OVERLAY_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
OVERLAY_LINES = ["-", "--", ":", "-."]
OVERLAY_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DEFAULT_OVERLAY = "guard,noise,noise_above,sail,sail_above"
MEASURES = {"threshold": "% of memorized samples above the threshold", "mean": "mean score of the memorized samples"}
QUALITY = [("viclip_memorized", "Caption match\n(memorized)"), ("fvd_memorized", "Realism\n(memorized)"),
           ("viclip_background", "Caption match\n(background)"), ("fvd_background", "Realism\n(background)")]
COMBINED = [("memorization_removed", "Memorization\nremoved"), ("quality_kept", "Quality\nkept")]
REFERENCES = ("normal", "original")


# ------------------------------------------------------------------------------------------------------------------
# Part 1: the scores
# ------------------------------------------------------------------------------------------------------------------

def memorization_metrics(table, args):
    """[(column, title, threshold)] of GSSCD, OFS-k and OFS-k with the natural motion filter."""
    ofs = next(c for c in table.columns if re.fullmatch(r"ofs_\d+", c))
    k = ofs.split("_")[1]
    return [("gsscd", "GSSCD (content)", args.gsscd_threshold), (ofs, f"OFS-{k} (motion)", args.ofs_threshold),
            (f"{ofs}_nmf", f"OFS-{k} with natural motion filter", args.ofs_threshold)]


def above(column, values, threshold):
    """Counted as memorized, as in memorization_eval.py: GSSCD >= threshold, OFS > threshold; no score = False."""
    with np.errstate(invalid="ignore"):
        return values >= threshold if column == "gsscd" else values > threshold


def aligned(table, order, what):
    if set(table["videoid"]) != set(order):
        raise SystemExit(f"{what} has other videos than no mitigation: generate every experiment from the same prompts.csv")
    return pd.Series(range(len(table)), index=table["videoid"]).loc[order].to_numpy()


def selected_methods(args):
    """The mitigation methods of --methods, in the given order (all of them if not given). No mitigation and the
    original model are references, not methods: they are always added."""
    available = [e for e in EXPERIMENTS if e != "normal"]
    if not args.methods or args.methods == "all":
        return available
    methods = list(dict.fromkeys(m for m in args.methods.split(",") if m and m != "normal"))
    unknown = [m for m in methods if m not in available]
    if unknown:
        raise SystemExit(f"--methods: unknown {', '.join(unknown)}; use any of {','.join(available)}")
    if not methods:
        raise SystemExit("--methods: give at least one mitigation method")
    return methods


def load_rows(args):
    """{name: (metrics in the order of normal's videos, ViCLIP per video, I3D features per video)} for the experiments
    and the original model; the real training clips (ViCLIP, I3D); the FVD reference; the roles of the videos."""
    names = ["normal"] + selected_methods(args)  # no mitigation is always there: the reference of every comparison
    order = read_metrics(args.out)["videoid"].tolist()
    sources = [(e, args.out, e) for e in names] + [("original", args.original, "normal")]
    if not args.original or not os.path.exists(os.path.join(args.original, "metrics.csv")):
        raise SystemExit(f"the original model's metrics.csv is missing in '{args.original}': it is the reference of the memorization axes")
    rows = {}
    for name, folder, experiment in sources:
        metrics = read_metrics(folder, experiment)
        metrics = metrics.iloc[aligned(metrics, order, f"{folder}/{experiment}")].reset_index(drop=True)
        q_path, f_path = result_file(folder, "quality.csv", experiment), result_file(folder, "i3d_features.npy", experiment)
        for path in (q_path, f_path):
            if not os.path.exists(path):
                raise SystemExit(f"{path} is missing: run `quality_eval.py --out {folder} metrics --experiments {experiment}` first")
        quality, features = pd.read_csv(q_path, keep_default_na=False), np.load(f_path)
        if len(features) != len(quality):
            raise SystemExit(f"{f_path} and {q_path} have different numbers of videos")
        position = aligned(quality, order, q_path)
        rows[name] = (metrics, quality["viclip"].to_numpy()[position], features[position])

    training = pd.read_csv(os.path.join(args.quality, "quality_training.csv"), keep_default_na=False)
    position = aligned(training, order, "quality_training.csv")
    training = (training["viclip"].to_numpy()[position], np.load(os.path.join(args.quality, "i3d_training.npy"))[position])
    reference = np.load(os.path.join(args.quality, "fvd_reference.npz"))["features"]
    return rows, training, reference, rows["normal"][0]["role"].to_numpy()


def raw_columns(metrics):
    return [f"{column}_{measure}" for column, _, _ in metrics for measure in MEASURES] + [c for c, _ in QUALITY]


def raw_scores(row, metrics, roles, reference):
    """The raw values of one row (raw_columns): for each memorization metric, the % of memorized samples above the
    threshold and their mean score; then ViCLIP and FVD of the memorized and of the background samples."""
    table, viclip, features = row
    memorized, background = roles == "memorized", roles == "background"
    out = []
    for column, _, threshold in metrics:
        values = table[column].to_numpy(dtype=float)[memorized]
        out += [100 * above(column, values, threshold).mean(), np.nanmean(values)]
    return out + [viclip[memorized].mean(), frechet_distance(features[memorized], reference),
                  viclip[background].mean(), frechet_distance(features[background], reference)]


def normalized(raw, names, columns, metrics, measure):
    """[rows, 7 axes + 2 combined]: the axes scaled to higher = better (see the top of the file)."""
    col = {c: k for k, c in enumerate(columns)}
    normal, original = raw[names.index("normal")], raw[names.index("original")]
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for column, _, _ in metrics:
            k = col[f"{column}_{measure}"]
            out.append((normal[k] - raw[:, k]) / (normal[k] - original[k]))
        for column, _ in QUALITY:
            k = col[column]
            out.append(raw[:, k] / normal[k] if column.startswith("viclip") else normal[k] / raw[:, k])
    out = np.stack(out, axis=-1)
    n = len(metrics)
    return np.concatenate([out, out[:, :n].mean(-1, keepdims=True), out[:, n:].mean(-1, keepdims=True)], axis=-1)


def summary_table(raw, scaled, names, columns, metrics, training, reference, roles):
    """One row per measure, row (method, references, real clips) and axis: the raw value and the scaled value."""
    axes = [c for c, _, _ in metrics] + [c for c, _ in QUALITY] + [c for c, _ in COMBINED]
    rows = []
    for measure in MEASURES:
        for r, name in enumerate(names):
            for a, axis in enumerate(axes):
                column = f"{axis}_{measure}" if a < len(metrics) else axis
                rows.append({"measure": measure, "name": name, "axis": axis,
                             "raw": raw[r, columns.index(column)] if column in columns else np.nan,
                             "scaled": scaled[measure][r, a]})
        viclip, features = training
        for group in ("memorized", "background"):
            members = roles == group
            for axis, value in [(f"viclip_{group}", viclip[members].mean()),
                                (f"fvd_{group}", frechet_distance(features[members], reference))]:
                rows.append({"measure": measure, "name": "training", "axis": axis, "raw": value})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------------
# Part 2: the figures
# ------------------------------------------------------------------------------------------------------------------

def name_of(name):
    return {"original": "Original model", "normal": "No mitigation", "training": "Real training clips"}.get(name, EXPERIMENTS.get(name, name))


def color_of(name):
    return {"original": ORIGINAL_COLOR, "normal": NO_MITIGATION_COLOR}.get(name, METHOD_COLOR)


def place_labels(ax, points, fontsize=10.5, obstacles=()):
    """Name every point [(x, y, text)] (data coordinates) at the first spot around it where the label covers no other
    label, no point, no obstacle [(x, y)] (e.g. points along an arrow), stays inside the axes, and is nearer to its own
    point than to any other; a thin line joins a label that had to move further away."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    centers = [ax.transData.transform((x, y)) for x, y, _ in points]
    blocked = [ax.transData.transform(p) for p in obstacles]
    frame = ax.get_window_extent(renderer)
    directions = [(1, 0.5), (1, -0.5), (-1, 0.5), (-1, -0.5), (0, 1), (0, -1), (1, 0), (-1, 0)]
    placed = []

    def gap(box, p):  # distance from a point to the label's box, in pixels
        return math.hypot(max(box.x0 - p[0], 0, p[0] - box.x1), max(box.y0 - p[1], 0, p[1] - box.y1))

    def cost(box, own):
        c = 5 * sum(max(0, min(box.x1, p.x1) - max(box.x0, p.x0)) * max(0, min(box.y1, p.y1) - max(box.y0, p.y0))
                    for p in placed)
        c += sum(400 for i, p in enumerate(centers) if i != own and gap(box, p) < 6)
        c += sum(60 for p in blocked if gap(box, p) < 2)
        mine = gap(box, centers[own])
        c += sum(300 for i, p in enumerate(centers) if i != own and gap(box, p) < mine)  # it would name the wrong point
        c += 2 * (max(0, frame.x0 - box.x0) + max(0, box.x1 - frame.x1)) * box.height
        c += 2 * (max(0, frame.y0 - box.y0) + max(0, box.y1 - frame.y1)) * box.width
        return c

    for i, (x, y, text) in enumerate(points):
        best = None
        for distance in (7, 16, 28, 42, 60):
            for dx, dy in directions:
                ha = "left" if dx > 0 else "right" if dx < 0 else "center"
                va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
                offset = (dx * distance, dy * distance)
                trial = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=fontsize)
                box = trial.get_window_extent(renderer)
                trial.remove()
                c = cost(box, i)
                if best is None or c < best[0]:
                    best = (c, offset, ha, va, box, distance)
                if c == 0:
                    break
            if best[0] == 0:
                break
        _, offset, ha, va, box, distance = best
        arrow = dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=0, shrinkB=5) if distance > 7 else None
        ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                    color=INK, arrowprops=arrow, zorder=5)
        placed.append(box)


# The spider plots: five metrics of the memorized samples, each with its own meaning and direction (no flipping).
RADAR_SCALINGS = {"raw": "Not scaled (FVD divided by its maximum over all rows)",
                  "max": "Each axis divided by its maximum over all rows"}


def radar_values(raw, columns, metrics, measure, scaling):
    """[rows, 5]: ViCLIP, GSSCD, OFS-k, OFS-k NMF and FVD of the memorized samples. GSSCD and OFS are the mean score
    (measure "mean") or the fraction of videos above the threshold (measure "threshold", the % as a fraction). FVD is
    divided by its maximum over the rows (all methods, no mitigation and the original model) to share the axis range;
    with scaling "max", every axis is divided by its maximum over the rows."""
    keys = ["viclip_memorized"] + [f"{column}_{measure}" for column, _, _ in metrics] + ["fvd_memorized"]
    values = raw[:, [columns.index(k) for k in keys]].astype(float)
    if measure == "threshold":
        values[:, 1:-1] /= 100
    # (the maximum, the row it comes from) of every axis that is divided by its maximum, else None
    maxima = [None] * values.shape[1]
    for a in range(values.shape[1]) if scaling == "max" else [values.shape[1] - 1]:
        row = int(np.nanargmax(values[:, a]))
        maxima[a] = (values[row, a], row)
        values[:, a] /= values[row, a]
    return values, maxima


def radar_labels(metrics, measure, maxima, names):
    """The axis labels: the metric, what is shown, its direction, and the maximum it is divided by (and whose it is)."""
    k = metrics[1][0].split("_")[1]
    shown = ["mean"] + [("mean" if measure == "mean" else f"fraction {'≥' if i == 0 else '>'} {metrics[i][2]:g}")
                        for i in range(len(metrics))] + [""]
    metric = ["ViCLIP", "GSSCD", f"OFS-{k}", f"OFS-{k} NMF", "FVD"]
    direction = ["higher is better"] + ["lower = less copying"] * len(metrics) + ["lower is better"]
    labels = []
    for a, name in enumerate(metric):
        lines = [name + (" / max" if maxima[a] else "")] + ([shown[a]] if shown[a] else []) + [direction[a]]
        if maxima[a]:
            value, row = maxima[a]
            lines.append(f"max {value:.0f} ({name_of(names[row])})" if name == "FVD"
                         else f"max {value:.3g} ({name_of(names[row])})")
        labels.append("\n".join(lines))
    return labels


def radar_frame(ax, labels):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, labels, fontsize=9.5, color=INK)
    ax.tick_params(axis="x", pad=10)
    for label, angle in zip(ax.get_xticklabels(), angles):  # text grows away from the circle, not over it
        side = np.sin(angle)
        label.set_horizontalalignment("left" if side > 0.1 else "right" if side < -0.1 else "center")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"], fontsize=8.5, color=MUTED)
    ax.set_rlabel_position(180 / len(labels))
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_visible(False)
    return angles


def polygon(ax, angles, values, color, **style):
    closed = np.append(values, values[0])
    angles = np.append(angles, angles[0])
    fill = style.pop("fill", None)
    ax.plot(angles, closed, color=color, **style)
    if fill:
        ax.fill(angles, closed, color=color, alpha=fill, linewidth=0)


def radar_title(measure, scaling):
    return f"Memorized samples. {RADAR_SCALINGS[scaling]}.\nGSSCD and OFS: {MEASURES[measure]}"


def plot_radar_grid(values, names, labels, measure, scaling, plt):
    """One spider plot per method, over no mitigation (dark gray) and the original model (dashed light gray)."""
    methods = [n for n in names if n not in REFERENCES]
    columns = min(3, len(methods))
    lines = math.ceil(len(methods) / columns)
    spare = columns * lines - len(methods)  # an empty cell holds the legend; without one, it goes below the panels
    below = 1.6 if not spare else 0  # inches under the panels for the legend
    fig, axes = plt.subplots(lines, columns, figsize=(8.2 * columns, 6.6 * lines + below),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    axes = axes.ravel()
    for ax, method in zip(axes, methods):
        angles = radar_frame(ax, labels)
        polygon(ax, angles, values[names.index("normal")], NO_MITIGATION_COLOR, linewidth=1.5, fill=0.06, label="No mitigation")
        polygon(ax, angles, values[names.index("original")], ORIGINAL_COLOR, linewidth=1.6, linestyle="--",
                label="Original model (never saw the duplicates)")
        polygon(ax, angles, values[names.index(method)], METHOD_COLOR, linewidth=2.2, fill=0.15, marker="o",
                markersize=4, label="The method of the panel")
        ax.set_title(name_of(method), fontsize=12.5, pad=30, color=INK)
    for ax in axes[len(methods):]:
        ax.set_axis_off()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    title = radar_title(measure, scaling).replace(". ", ".\n")
    height = fig.get_figheight()
    if spare:
        axes[len(methods)].legend(handles, legend_labels, frameon=False, loc="center", fontsize=10.5, title=title,
                                  title_fontsize=10.5)
    else:
        fig.legend(handles, legend_labels, frameon=False, loc="lower center", ncol=len(handles), fontsize=10.5,
                   title=title, title_fontsize=10.5)
    # tight_layout does not see the side-aligned polar labels: room for them is set by hand (in inches, so that it
    # does not depend on the number of rows)
    fig.subplots_adjust(left=0.09, right=0.91, top=1 - 0.35 / height, bottom=(0.35 + below) / height,
                        wspace=0.95, hspace=4.0 / 6.6)
    return fig


def family_of(name):
    """The method a variant belongs to: the longest leading part of the name, cut at "_", that is an experiment of its
    own (noise_above -> noise, sail_above -> sail); a method that is no variant is its own family."""
    parts = name.split("_")
    for end in range(len(parts) - 1, 0, -1):
        if "_".join(parts[:end]) in EXPERIMENTS:
            return "_".join(parts[:end])
    return name


def overlay_styles(methods):
    """{method: line style} for any number of methods: one color per family, in the order the families first appear
    in `methods`; in a family the base method comes first (solid line, filled markers), its variants next (dashed,
    dotted, ... with hollow markers)."""
    families = list(dict.fromkeys(family_of(m) for m in methods))
    styles = {}
    for f, family in enumerate(families):
        members = sorted([m for m in methods if family_of(m) == family], key=lambda m: (m != family, methods.index(m)))
        color = OVERLAY_COLORS[f % len(OVERLAY_COLORS)]
        marker = OVERLAY_MARKERS[(f // len(OVERLAY_COLORS)) % len(OVERLAY_MARKERS)]
        for v, method in enumerate(members):
            styles[method] = dict(color=color, linestyle=OVERLAY_LINES[v % len(OVERLAY_LINES)], linewidth=2.2,
                                  marker=marker, markersize=5.5, markeredgecolor=color, markeredgewidth=1.5,
                                  markerfacecolor=color if v == 0 else "white")
    return styles


def plot_radar_overlay(values, names, labels, methods, measure, scaling, plt):
    """Any number of methods on one spider plot, with no mitigation and the original model (see overlay_styles)."""
    fig, ax = plt.subplots(figsize=(11.5, 7.0), subplot_kw={"projection": "polar"})
    angles = radar_frame(ax, labels)
    polygon(ax, angles, values[names.index("normal")], NO_MITIGATION_COLOR, linewidth=1.5, fill=0.06, label="No mitigation")
    polygon(ax, angles, values[names.index("original")], ORIGINAL_COLOR, linewidth=1.6, linestyle="--", label="Original model")
    for method, style in overlay_styles(methods).items():
        style = dict(style)
        # no fill: overlapping fills mix into colors of no method
        polygon(ax, angles, values[names.index(method)], style.pop("color"), label=name_of(method), **style)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.32, 1.0), fontsize=10.5,
              title=radar_title(measure, scaling).replace(". ", ".\n"), title_fontsize=10.5, alignment="left")
    fig.tight_layout()
    return fig


def plot_tradeoff_grid(raw, names, columns, metrics, training, measure, plt):
    """Memorization (x, one column per metric) against quality (y: ViCLIP and FVD) of the memorized samples, one point
    per row (methods, no mitigation, original model)."""
    from matplotlib.lines import Line2D
    point = raw
    fig, axes = plt.subplots(2, len(metrics), figsize=(5.6 * len(metrics), 10), squeeze=False)
    quality = [("viclip_memorized", "Mean ViCLIP similarity\n(higher is better)", True),
               ("fvd_memorized", "FVD\n(lower is better: axis flipped)", False)]
    for c, (column, title, threshold) in enumerate(metrics):
        kx = columns.index(f"{column}_{measure}")
        for r, (y_column, y_title, higher) in enumerate(quality):
            ax = axes[r][c]
            ky = columns.index(y_column)
            x, y = point[:, kx], point[:, ky]
            for i, name in enumerate(names):
                ax.scatter(x[i], y[i], s=85, color=color_of(name), edgecolors="white", linewidths=1.5,
                           zorder=4 if name in REFERENCES else 3)
            reference = training[y_column]
            ax.axhline(reference, color=MUTED, linestyle="--", linewidth=1.1)
            if measure == "mean":
                ax.axvline(threshold, color=MUTED, linestyle=(0, (1, 2)), linewidth=1.1)
            if not higher:
                ax.invert_yaxis()
            ax.margins(x=0.12, y=0.12)
            if r == 0:
                ax.set_title(title, loc="left")
            if c == 0:
                ax.set_ylabel(f"Memorized samples: {y_title}")
            ax.set_xlabel("Memorized samples above the threshold (%)" if measure == "threshold"
                          else f"Memorized samples: mean {title.split(' (')[0].replace(' with natural motion filter', ' NMF')}")
    handles = [Line2D([], [], marker="o", linestyle="", color=METHOD_COLOR, markersize=9, label="Mitigation method"),
               Line2D([], [], marker="o", linestyle="", color=NO_MITIGATION_COLOR, markersize=9, label="No mitigation"),
               Line2D([], [], marker="o", linestyle="", color=ORIGINAL_COLOR, markersize=9, label="Original model"),
               Line2D([], [], linestyle="--", color=MUTED, label="Real training clips")]
    if measure == "mean":
        handles.append(Line2D([], [], linestyle=(0, (1, 2)), color=MUTED, label="Threshold of the paper"))
    fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="lower center", fontsize=10.5)
    fig.text(0.5, 0.035, "Better = left (less copying) and up (better quality).", ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for c, (column, _, _) in enumerate(metrics):  # after the layout: the labels are placed in screen space
        kx = columns.index(f"{column}_{measure}")
        for r, (y_column, _, _) in enumerate(quality):
            ky = columns.index(y_column)
            place_labels(axes[r][c], [(point[i, kx], point[i, ky], name_of(n)) for i, n in enumerate(names)], fontsize=9.5)
    return fig


def plot_summary(scaled, names, measure, plt):
    """Each method as one point: memorization removed (mean of the three memorization axes) against quality kept (mean
    of the four quality axes). The goal is the top right: (1, 1)."""
    point = scaled
    a, b = point.shape[1] - 2, point.shape[1] - 1
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.axvline(1, color=MUTED, linestyle="--", linewidth=1)
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1, 1.0, " goal ", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, color=MUTED)
    for i, name in enumerate(names):
        ax.scatter(point[i, a], point[i, b], s=95, color=color_of(name), edgecolors="white", linewidths=1.5,
                   zorder=4 if name in REFERENCES else 3)
    ax.margins(0.12)
    ax.set_xlabel("Memorization removed (mean of GSSCD, OFS, OFS NMF axes)\n0 = as without mitigation, 1 = as the original model")
    ax.set_ylabel("Quality kept (mean of the 4 quality axes)\n1 = as without mitigation")
    ax.set_title(f"Memorization: {MEASURES[measure]}", loc="left")
    fig.tight_layout()
    place_labels(ax, [(point[i, a], point[i, b], name_of(n)) for i, n in enumerate(names)])
    return fig


def plot_content_vs_motion(raw, names, columns, metrics, measure, plt):
    """Content memorization (GSSCD, x) against motion memorization (OFS, y), with an arrow from no mitigation to every
    method: which one each method lowers."""
    point = raw
    content, motions = metrics[0], metrics[1:]
    kx = columns.index(f"{content[0]}_{measure}")
    normal = names.index("normal")
    fig, axes = plt.subplots(1, len(motions), figsize=(7.2 * len(motions), 6.4), squeeze=False)
    for ax, (column, title, threshold) in zip(axes[0], motions):
        ky = columns.index(f"{column}_{measure}")
        x, y = point[:, kx], point[:, ky]
        for i, name in enumerate(names):
            if name not in REFERENCES:
                ax.annotate("", xy=(x[i], y[i]), xytext=(x[normal], y[normal]),
                            arrowprops=dict(arrowstyle="-|>", color=METHOD_COLOR, alpha=0.45, linewidth=1.2, shrinkA=6, shrinkB=6))
            ax.scatter(x[i], y[i], s=85, color=color_of(name), edgecolors="white", linewidths=1.5,
                       zorder=4 if name in REFERENCES else 3)
        if measure == "mean":
            ax.axvline(content[2], color=MUTED, linestyle=(0, (1, 2)), linewidth=1.1)
            ax.axhline(threshold, color=MUTED, linestyle=(0, (1, 2)), linewidth=1.1)
        ax.margins(0.12)
        unit = "above the threshold (%)" if measure == "threshold" else "mean"
        ax.set_xlabel(f"Memorized samples: GSSCD (content) {unit}")
        ax.set_ylabel(f"Memorized samples: {title.replace(' (motion)', '').replace(' with natural motion filter', ' NMF')} {unit}")
        ax.set_title(f"Content against {title.replace('(motion)', '').strip()}", loc="left")
    note = "Arrows start at no mitigation; lower and further left = less copying."
    if measure == "mean":
        note += " Dotted lines: the paper's thresholds."
    fig.text(0.01, 0.005, note, fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ax, (column, _, _) in zip(axes[0], motions):
        ky = columns.index(f"{column}_{measure}")
        x, y = point[:, kx], point[:, ky]
        arrows = [(x[normal] + t * (x[i] - x[normal]), y[normal] + t * (y[i] - y[normal]))
                  for i, n in enumerate(names) if n not in REFERENCES for t in np.linspace(0.05, 0.98, 40)]
        place_labels(ax, [(x[i], y[i], name_of(n)) for i, n in enumerate(names)], obstacles=arrows)
    return fig


def plot_scorecard(raw, scaled, names, columns, metrics, labels, measure, plt):
    """Every row and axis as a cell: the color and the large number are the scaled score (darker = better), the
    small number is the raw value. The last two columns are the means used in the summary scatter."""
    import matplotlib
    order = ["normal"] + [n for n in names if n not in REFERENCES] + ["original"]
    point = scaled
    n_axes = len(labels)
    grid = np.full((len(order), n_axes + 3), np.nan)  # a blank column between the axes and the means
    for r, name in enumerate(order):
        grid[r, :n_axes] = point[names.index(name), :n_axes]
        grid[r, n_axes + 1:] = point[names.index(name), n_axes:]
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(1.35 * (n_axes + 3) + 2.5, 0.72 * len(order) + 2.2))
    image = ax.imshow(np.clip(grid, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    raw_keys = [f"{c}_{measure}" for c, _, _ in metrics] + [c for c, _ in QUALITY]
    for r, name in enumerate(order):
        i = names.index(name)
        for c in range(grid.shape[1]):
            if np.isnan(grid[r, c]):
                continue
            ink = "white" if grid[r, c] > 0.62 else INK
            ax.text(c, r - 0.12, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=11.5, color=ink, fontweight="bold")
            if c < n_axes:
                key = raw_keys[c]
                value = raw[i, columns.index(key)]
                text = (f"{value:.0f}%" if key.endswith("_threshold") else f"{value:.0f}" if key.startswith("fvd")
                        else f"{value:.3f}")
                ax.text(c, r + 0.24, text, ha="center", va="center", fontsize=8.5, color=ink)
    ax.axhline(len(order) - 1.5, color=INK, linewidth=1.2)  # the original model is a reference, not a method
    ax.axhline(0.5, color=INK, linewidth=1.2)
    ax.set_xticks(range(grid.shape[1]), labels + [""] + [l for _, l in COMBINED], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), [name_of(n) for n in order], fontsize=10.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("Scaled score (darker = better; above 1 shown as 1)")
    bar.outline.set_visible(False)
    fig.text(0.01, 0.005, f"Memorization: {MEASURES[measure]}. Large number: scaled score (memorization: 0 = as without "
             "mitigation, 1 = as the original model; quality: fraction kept). Small number: the raw value.",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# ------------------------------------------------------------------------------------------------------------------
# Part 3: the command
# ------------------------------------------------------------------------------------------------------------------

def run(args):
    plt = pyplot()
    rows, training, reference, roles = load_rows(args)
    names = list(rows)
    metrics = memorization_metrics(rows["normal"][0], args)
    columns = raw_columns(metrics)
    methods = [n for n in names if n not in REFERENCES]
    if args.overlay is None:  # with --methods: all of them; without: DEFAULT_OVERLAY
        overlay = methods if args.methods else [m for m in DEFAULT_OVERLAY.split(",") if m in methods]
    elif args.overlay == "all":
        overlay = methods
    else:
        overlay = [m for m in args.overlay.split(",") if m]
        unknown = [m for m in overlay if m not in methods]
        if unknown:
            raise SystemExit(f"--overlay: {', '.join(unknown)} not among the methods of --methods ({','.join(methods)})")

    raw = np.array([raw_scores(row, metrics, roles, reference) for row in rows.values()])  # [rows, raw columns]
    scaled = {measure: normalized(raw, names, columns, metrics, measure) for measure in MEASURES}

    os.makedirs(args.save_to, exist_ok=True)
    summary = summary_table(raw, scaled, names, columns, metrics, training, reference, roles)
    summary.to_csv(os.path.join(args.save_to, "tradeoff_summary.csv"), index=False, float_format="%.4g")
    table = summary[summary["name"] != "training"].pivot_table(index=["measure", "name"], columns="axis", values="scaled", sort=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(table.round(2).to_string())

    training_raw = summary[(summary["name"] == "training") & (summary["measure"] == "threshold")].set_index("axis")["raw"]
    k = metrics[1][0].split("_")[1]
    labels = ["Content\n(GSSCD)", f"Motion\n(OFS-{k})", f"Motion, filtered\n(OFS-{k} NMF)"] + [l for _, l in QUALITY]
    figures = {}
    for measure in MEASURES:
        s = scaled[measure]
        for scaling in RADAR_SCALINGS:
            values, maxima = radar_values(raw, columns, metrics, measure, scaling)
            radar = radar_labels(metrics, measure, maxima, names)
            figures[f"radar_{measure}_{scaling}"] = plot_radar_grid(values, names, radar, measure, scaling, plt)
            if overlay:
                figures[f"radar_overlay_{measure}_{scaling}"] = plot_radar_overlay(values, names, radar, overlay, measure,
                                                                                   scaling, plt)
        figures[f"tradeoff_{measure}"] = plot_tradeoff_grid(raw, names, columns, metrics, training_raw, measure, plt)
        figures[f"summary_{measure}"] = plot_summary(s, names, measure, plt)
        figures[f"content_vs_motion_{measure}"] = plot_content_vs_motion(raw, names, columns, metrics, measure, plt)
        figures[f"scorecard_{measure}"] = plot_scorecard(raw, s, names, columns, metrics, labels, measure, plt)
    save_figures(figures, args.save_to, None, plt)
    print(f"-> {args.save_to}: tradeoff_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Compare the mitigation methods on memorization and quality together")
    parser.add_argument("--out", default="outputs/finetune_eval", help="folder of the experiments' results")
    parser.add_argument("--original", default=ORIGINAL_DEFAULT, help="folder of the original model's results (the reference)")
    parser.add_argument("--quality", default="outputs/finetune_quality", help="folder of `quality_eval.py reference`")
    parser.add_argument("--save-to", default="outputs/finetune_tradeoff", help="folder of the figures and the table")
    parser.add_argument("--methods", default=None,
                        help="the mitigation methods to compare, e.g. guard,noise,sail (default: all); no mitigation and "
                             "the original model are always drawn as references")
    parser.add_argument("--overlay", default=None,
                        help="methods drawn together on one spider plot, any number: a list, 'all', or '' for none "
                             f"(default: all of --methods if given, else {DEFAULT_OVERLAY})")
    parser.add_argument("--gsscd-threshold", type=float, default=0.4, help="GSSCD from which a sample counts as copied")
    parser.add_argument("--ofs-threshold", type=float, default=0.5, help="OFS-k above which motion counts as copied")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
