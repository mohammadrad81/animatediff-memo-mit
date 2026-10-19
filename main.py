"""Command line interface. See README.md, or run:  python main.py {attention,guard,soften,clip,batch,compare} --help"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

from attention import Controller, cross_layers, install, select_layers, short
from plots import (memorization_signals, plot_all, plot_memorization_signals, token_labels, write_guard_summary,
                   write_soften_summary)
from sampler import load_pipeline, sample, save_gif


def video_args(p, out, prompt=True):
    if prompt:
        p.add_argument("--prompt", required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--steps", type=int, default=25, help="diffusion steps")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--s", type=float, default=7.5, help="guidance scale (attraction scale in GUARD)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", default=out, help="output folder")


def guard_args(p, r_flag="--r"):
    p.add_argument(r_flag, dest="guard_r", type=float, default=2.0, help="GUARD repulsion scale (0 = CA attenuation only)")
    p.add_argument("--alpha", type=float, default=0.3, help="spiky logits are multiplied by alpha (1 = no attenuation)")
    p.add_argument("--tau", type=float, default=0.1, help="spike threshold on the z-score")
    p.add_argument("--blocks", default="down,mid", help="UNet parts whose cross-attention is attenuated: down,mid,up")


def soften_args(p, r_flag="--r"):
    p.add_argument("--temperature", "-T", type=float, default=2.0,
                   help="logits are divided by T before the softmax (> 1 = softer, 1 = no change, < 1 = sharper)")
    p.add_argument("--kinds", default="cross", help="kinds of attention layers to soften: cross,spatial,temporal")
    p.add_argument("--layers", default="down,mid",
                   help="prefixes of layer names to soften, e.g. down,mid or down1.attentions0,up3.motion")
    p.add_argument("--keep-sot", action="store_true",
                   help="cross-attention: keep the attention on <sot> and soften only the text tokens")
    p.add_argument(r_flag, dest="soften_r", type=float, default=0.0,
                   help="repulsion of softening from the normal prediction, as in GUARD (0 = plain CFG on the softened stream)")


def clip_args(p):
    p.add_argument("--clip-mode", choices=["pad", "mask"], default="pad",
                   help="pad: pad the prompt with '!' instead of <eot> and set v_eot to 0; mask: set part of v_pad to 0")
    p.add_argument("--keep-eot", action="store_true", help="pad mode: keep v_eot (only replace the padding token)")
    p.add_argument("--mask-ratio", type=float, default=0.7,
                   help="mask mode: share of the padding embeddings (the ones right after <eot>) set to 0")


def schedule_arg(p):
    p.add_argument("--r-schedule", choices=["cosine", "constant"], default="cosine",
                   help="cosine: r decays to 0 over the steps (official GUARD code); constant: fixed r (paper Eq. 5)")


def soften_layers(args, unet):
    layers = select_layers(unet, args.kinds.split(","), args.layers.split(","))
    if not layers:
        raise SystemExit(f"no attention layer matches --kinds {args.kinds} --layers {args.layers}")
    print(f"softening {len(layers)} layers with T = {args.temperature}: " + ", ".join(short(n) for n in layers))
    return layers


def experiment(name, args, unet, soft_layers=()):
    """Controller and repulsion scale r of one experiment: normal (plain AnimateDiff), guard, soften or clip."""
    if name == "normal":  # no recording, no attenuation = plain AnimateDiff
        return Controller(num_streams=2), 0.0
    if name == "guard":
        layers = cross_layers(unet, args.blocks.split(","))
        return Controller(num_streams=3, guard_layers=layers, tau=args.tau, alpha=args.alpha), args.guard_r
    if name == "clip":  # only the prompt's text embeddings change; normal attention and CFG
        return Controller(num_streams=2, clip_mode=args.clip_mode, mask_eot=not args.keep_eot,
                          mask_ratio=args.mask_ratio), 0.0
    return Controller(num_streams=3 if args.soften_r > 0 else 2, soft_layers=soft_layers,
                      temperature=args.temperature, keep_sot=args.keep_sot), args.soften_r


def run_attention(args, pipe):
    """Normal CFG generation while recording the attention maps of the prompt stream, then plot them."""
    labels, eot = token_labels(pipe.tokenizer, args.prompt)
    ctrl = Controller(num_streams=2, record=True)
    ctrl.eot = eot  # also record the share of attention on <eot> + padding
    install(pipe.unet, ctrl)
    frames = sample(pipe, args.prompt, ctrl, args.frames, args.steps, args.height, args.width, args.seed, args.s)
    save_gif(frames, os.path.join(args.out, "video.gif"), args.fps)

    # Signals to compare memorized and non-memorized prompts (see also the "compare" command).
    signals = memorization_signals(ctrl, frames, args.prompt)
    with open(os.path.join(args.out, "memorization_signals.json"), "w") as f:
        json.dump(signals, f)
    plot_memorization_signals([(os.path.basename(os.path.normpath(args.out)), signals)],
                              os.path.join(args.out, "memorization_signals.png"))
    steps = [int(t) for t in args.plot_steps.split(",")] if args.plot_steps else None  # None = all steps
    plot_all(ctrl.maps, labels, eot, args.tau, (args.height // 8, args.width // 8), args.out, steps)


def run_clip(args, pipe):
    """CLIP-embedding mitigation (Kim & No), and optionally the plain CFG video for comparison."""
    common = (args.frames, args.steps, args.height, args.width, args.seed, args.s)
    if args.baseline:
        plain, _ = experiment("normal", args, pipe.unet)
        install(pipe.unet, plain)
        baseline = sample(pipe, args.prompt, plain, *common)
        save_gif(baseline, os.path.join(args.out, "baseline.gif"), args.fps)
    ctrl, _ = experiment("clip", args, pipe.unet)
    install(pipe.unet, ctrl)
    frames = sample(pipe, args.prompt, ctrl, *common)
    save_gif(frames, os.path.join(args.out, "clip.gif"), args.fps)
    if args.baseline:  # baseline | clip next to each other
        both = [Image.fromarray(np.hstack([a, b])) for a, b in zip(baseline, frames)]
        save_gif(both, os.path.join(args.out, "baseline_vs_clip.gif"), args.fps)
    print(f"done: {args.out}")


def run_compare(args):
    """Overlay the memorization signals of several `attention` runs in one figure (no model needed)."""
    runs = []
    for d in args.dirs:
        path = os.path.join(d, "memorization_signals.json")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: run `python main.py attention ... --out {d}` first")
        with open(path) as f:
            runs.append((os.path.basename(os.path.normpath(d)), json.load(f)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_memorization_signals(runs, args.out)
    print(f"written: {args.out}")


def run_guard(args, pipe):
    """CA-in-GUARD generation (and optionally the plain CFG video with the same seed for comparison)."""
    common = (args.frames, args.steps, args.height, args.width, args.seed, args.s)
    if args.baseline:
        plain, _ = experiment("normal", args, pipe.unet)
        install(pipe.unet, plain)
        baseline = sample(pipe, args.prompt, plain, *common)
        save_gif(baseline, os.path.join(args.out, "baseline.gif"), args.fps)

    ctrl, r = experiment("guard", args, pipe.unet)
    install(pipe.unet, ctrl)
    frames = sample(pipe, args.prompt, ctrl, *common, r=r, r_schedule=args.r_schedule)
    save_gif(frames, os.path.join(args.out, "guard.gif"), args.fps)

    labels, _ = token_labels(pipe.tokenizer, args.prompt)
    write_guard_summary(ctrl.spikes, labels, os.path.join(args.out, "attenuated_tokens.txt"))
    if args.baseline:  # baseline | GUARD next to each other
        both = [Image.fromarray(np.hstack([a, b])) for a, b in zip(baseline, frames)]
        save_gif(both, os.path.join(args.out, "baseline_vs_guard.gif"), args.fps)
    print(f"done: {args.out}")


def run_soften(args, pipe):
    """Temperature-softened attention in the chosen layers (and optionally the plain CFG video for comparison).
    With r > 0 the softened prediction is also pushed away from the normal one, as in GUARD (3 streams)."""
    layers = soften_layers(args, pipe.unet)
    common = (args.frames, args.steps, args.height, args.width, args.seed, args.s)
    if args.baseline:
        plain, _ = experiment("normal", args, pipe.unet)
        install(pipe.unet, plain)
        baseline = sample(pipe, args.prompt, plain, *common)
        save_gif(baseline, os.path.join(args.out, "baseline.gif"), args.fps)

    ctrl, r = experiment("soften", args, pipe.unet, layers)
    install(pipe.unet, ctrl)
    frames = sample(pipe, args.prompt, ctrl, *common, r=r, r_schedule=args.r_schedule)
    save_gif(frames, os.path.join(args.out, "soft.gif"), args.fps)

    write_soften_summary(ctrl.soft_stats, layers, args.temperature, args.keep_sot,
                         os.path.join(args.out, "softened_layers.txt"))
    if args.baseline:  # baseline | softened next to each other
        both = [Image.fromarray(np.hstack([a, b])) for a, b in zip(baseline, frames)]
        save_gif(both, os.path.join(args.out, "baseline_vs_soft.gif"), args.fps)
    print(f"done: {args.out}")


EXPERIMENTS = ["normal", "guard", "soften", "clip"]
NOT_SETTINGS = ["start", "end", "batch_size", "experiments"]  # args that do not change the generated videos
PATHS = ["csv", "out"]  # stored in settings.json, but may change (moved folder, sheet split into files)


def run_batch(args, pipe):
    """Run the chosen experiments on the prompts of a CSV sheet (columns "index" and "caption").
    Each video is saved as <out>/<experiment>/<index>.gif. Videos that already exist are skipped, so a stopped
    run can be continued, and a sheet can be split into parts with --start/--end."""
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"index", "caption"} <= set(reader.fieldnames or []):
            raise SystemExit(f"{args.csv} needs the columns 'index' and 'caption' (found: {reader.fieldnames})")
        rows = [(row["index"].strip().replace("/", "_"), row["caption"].strip()) for row in reader]
    indices = [i for i, _ in rows]
    if len(set(indices)) != len(indices):
        raise SystemExit(f"{args.csv} has repeated values in the 'index' column; they are used as file names")
    rows = [(i, c) for i, c in rows[args.start:args.end] if c]  # rows with an empty caption are skipped
    experiments = args.experiments.split(",")
    if not set(experiments) <= set(EXPERIMENTS):
        raise SystemExit(f"--experiments must be a list of {','.join(EXPERIMENTS)}")

    # The settings are kept next to the videos; continuing a run with other settings would mix experiments.
    settings = {k: v for k, v in vars(args).items() if k not in NOT_SETTINGS}
    settings_path = os.path.join(args.out, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            old = json.load(f)
        changed = ", ".join(k for k in settings.keys() | old.keys()  # options added later are missing from old files
                           if k not in PATHS and k in old and settings.get(k) != old.get(k))
        if changed:
            raise SystemExit(f"{args.out} was made with other settings ({changed}); use another --out")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    layers = soften_layers(args, pipe.unet) if "soften" in experiments else ()
    for name in experiments:
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
    gif = lambda name, index: os.path.join(args.out, name, f"{index}.gif")
    common = (args.frames, args.steps, args.height, args.width, args.seed, args.s)
    print(f"{len(rows)} prompts x {len(experiments)} experiments -> {args.out}")

    for b in range(0, len(rows), args.batch_size):
        batch = rows[b:b + args.batch_size]
        for name in experiments:
            todo = [(i, c) for i, c in batch if not os.path.exists(gif(name, i))]
            if not todo:
                continue
            print(f"[{b + len(batch)}/{len(rows)}] {name}: index " + ", ".join(i for i, _ in todo))
            ctrl, r = experiment(name, args, pipe.unet, layers)  # a new controller per batch: nothing piles up
            install(pipe.unet, ctrl)
            videos = sample(pipe, [c for _, c in todo], ctrl, *common, r=r, r_schedule=args.r_schedule)
            for (i, _), frames in zip(todo, videos):  # written under another name first: a GIF that exists is complete
                save_gif(frames, gif(name, i) + ".tmp.gif", args.fps)
                os.replace(gif(name, i) + ".tmp.gif", gif(name, i))
    print(f"done: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="GUARD memorization mitigation and attention plots for AnimateDiff")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("attention", help="generate a video (normal CFG), plot all attention maps and mark spikes")
    video_args(a, "outputs/attention")
    a.add_argument("--tau", type=float, default=3.0, help="spike threshold on the z-score")
    a.add_argument("--plot-steps", default=None, help="diffusion steps to plot, e.g. 0,12,24 (default: all)")

    g = sub.add_parser("guard", help="generate a video with CA-in-GUARD")
    video_args(g, "outputs/guard")
    guard_args(g)
    schedule_arg(g)
    g.add_argument("--baseline", action="store_true", help="also generate the plain AnimateDiff video (same seed)")

    t = sub.add_parser("soften", help="generate a video with temperature-softened attention in chosen layers")
    video_args(t, "outputs/soften")
    soften_args(t)
    schedule_arg(t)
    t.add_argument("--baseline", action="store_true", help="also generate the plain AnimateDiff video (same seed)")

    b = sub.add_parser("batch", help="run normal / guard / soften / clip on every prompt of a CSV sheet")
    video_args(b, "outputs/batch", prompt=False)
    b.add_argument("--csv", required=True, help="CSV file with the columns 'index' and 'caption'")
    b.add_argument("--experiments", default="normal,guard,soften,clip", help="any of normal,guard,soften,clip")
    b.add_argument("--start", type=int, default=0, help="first CSV row to run (0 = first data row)")
    b.add_argument("--end", type=int, default=None, help="stop before this CSV row (default: the end of the sheet)")
    b.add_argument("--batch-size", type=int, default=1, help="prompts generated together in one UNet pass")
    guard_args(b, r_flag="--guard-r")
    soften_args(b, r_flag="--soften-r")
    clip_args(b)
    schedule_arg(b)

    k = sub.add_parser("clip", help="generate a video with the CLIP-embedding mitigation (pad with '!' / mask v_pad)")
    video_args(k, "outputs/clip")
    clip_args(k)
    k.add_argument("--baseline", action="store_true", help="also generate the plain AnimateDiff video (same seed)")

    c = sub.add_parser("compare", help="overlay the memorization signals of several attention runs")
    c.add_argument("dirs", nargs="+", help="output folders of the attention command")
    c.add_argument("--out", default="outputs/compare.png", help="figure to write")

    args = parser.parse_args()
    if args.command == "compare":  # only reads JSON files: no model is loaded
        return run_compare(args)
    os.makedirs(args.out, exist_ok=True)
    pipe = load_pipeline()
    if args.command == "attention":
        run_attention(args, pipe)
    elif args.command == "guard":
        run_guard(args, pipe)
    elif args.command == "clip":
        run_clip(args, pipe)
    elif args.command == "soften":
        run_soften(args, pipe)
    else:
        run_batch(args, pipe)


if __name__ == "__main__":
    main()
