"""Command line interface. See README.md, or run:  python main.py attention --help / python main.py generate --help"""
import argparse
import os

import numpy as np
from PIL import Image

from attention import Controller, cross_layers, install
from plots import plot_all, token_labels, write_guard_summary
from sampler import load_pipeline, sample, save_gif


def video_args(p, out):
    p.add_argument("--prompt", required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--steps", type=int, default=25, help="diffusion steps")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--s", type=float, default=7.5, help="guidance scale (attraction scale in GUARD)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", default=out, help="output folder")


def run_attention(args, pipe):
    """Normal CFG generation while recording the attention maps of the prompt stream, then plot them."""
    ctrl = Controller(num_streams=2, record=True)
    install(pipe.unet, ctrl)
    frames = sample(pipe, args.prompt, ctrl, args.frames, args.steps, args.height, args.width, args.seed, args.s)
    save_gif(frames, os.path.join(args.out, "video.gif"), args.fps)
    labels, eot = token_labels(pipe.tokenizer, args.prompt)
    steps = [int(t) for t in args.plot_steps.split(",")] if args.plot_steps else None  # None = all steps
    plot_all(ctrl.maps, labels, eot, args.tau, (args.height // 8, args.width // 8), args.out, steps)


def run_generate(args, pipe):
    """CA-in-GUARD generation (and optionally the plain CFG video with the same seed for comparison)."""
    common = (args.frames, args.steps, args.height, args.width, args.seed, args.s)
    if args.baseline:
        plain = Controller(num_streams=2)  # no recording, no attenuation = plain AnimateDiff
        install(pipe.unet, plain)
        baseline = sample(pipe, args.prompt, plain, *common)
        save_gif(baseline, os.path.join(args.out, "baseline.gif"), args.fps)

    layers = cross_layers(pipe.unet, args.blocks.split(","))
    ctrl = Controller(num_streams=3, guard_layers=layers, tau=args.tau, alpha=args.alpha)
    install(pipe.unet, ctrl)
    frames = sample(pipe, args.prompt, ctrl, *common, r=args.r, r_schedule=args.r_schedule)
    save_gif(frames, os.path.join(args.out, "guard.gif"), args.fps)

    labels, _ = token_labels(pipe.tokenizer, args.prompt)
    write_guard_summary(ctrl.spikes, labels, os.path.join(args.out, "attenuated_tokens.txt"))
    if args.baseline:  # baseline | GUARD next to each other
        both = [Image.fromarray(np.hstack([a, b])) for a, b in zip(baseline, frames)]
        save_gif(both, os.path.join(args.out, "baseline_vs_guard.gif"), args.fps)
    print(f"done: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="GUARD memorization mitigation and attention plots for AnimateDiff")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("attention", help="generate a video (normal CFG), plot all attention maps and mark spikes")
    video_args(a, "outputs/attention")
    a.add_argument("--tau", type=float, default=3.0, help="spike threshold on the z-score")
    a.add_argument("--plot-steps", default=None, help="diffusion steps to plot, e.g. 0,12,24 (default: all)")

    g = sub.add_parser("generate", help="generate a video with CA-in-GUARD")
    video_args(g, "outputs/guard")
    g.add_argument("--r", type=float, default=2.0, help="repulsion scale (0 = CA attenuation only)")
    g.add_argument("--r-schedule", choices=["cosine", "constant"], default="cosine",
                   help="cosine: r decays to 0 over the steps (official code); constant: paper Eq. 5")
    g.add_argument("--alpha", type=float, default=0.3, help="spiky logits are multiplied by alpha (1 = no attenuation)")
    g.add_argument("--tau", type=float, default=0.1, help="spike threshold on the z-score")
    g.add_argument("--blocks", default="down,mid", help="UNet parts whose cross-attention is attenuated: down,mid,up")
    g.add_argument("--baseline", action="store_true", help="also generate the plain AnimateDiff video (same seed)")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pipe = load_pipeline()
    if args.command == "attention":
        run_attention(args, pipe)
    else:
        run_generate(args, pipe)


if __name__ == "__main__":
    main()
