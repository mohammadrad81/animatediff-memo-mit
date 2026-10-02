"""Command line interface.

    python -m animatediff_attn list-layers [--attn-types ...] [--blocks ...]
    python -m animatediff_attn run    --prompt "..." [selection] [recording] [spike/plot options]
    python -m animatediff_attn plot   --run-dir outputs/<run> [selection] [spike/plot options]
    python -m animatediff_attn spikes --run-dir outputs/<run> [selection] [spike options]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .model import MODEL_ID, MOTION_ADAPTER_ID
from .plotting import PLOT_KINDS
from .selection import LayerInfo, format_layer_table, parse_index_list, select_layers
from .spikes import AGGS, SCORES


# ------------------------------------------------------------------------- arguments
def add_selection_args(p):
    g = p.add_argument_group("layer / head / frame selection (all filters are intersected)")
    g.add_argument("--attn-types", default="all",
                   help="comma list of spatial,cross,temporal (spatial = self-attn within a frame, "
                        "temporal = motion-module attention across frames). Default: all")
    g.add_argument("--blocks", default="all", help="comma list, e.g. 'down,mid' or 'down.1,up.3'. Default: all")
    g.add_argument("--layers", nargs="*", default=None, metavar="GLOB",
                   help="fnmatch patterns on module names, e.g. '*up_blocks.3*attn2'")
    g.add_argument("--layer-ids", default=None, help="ids from list-layers: '0,5,12' or '0:20:2'")
    g.add_argument("--heads", default="all", help="heads to store/plot: 'all' or '0,3,7'")
    g.add_argument("--frames", default="all", help="frames to store/plot: 'all', '0,8,15' or '0:16:4'")


def add_spike_args(p):
    g = p.add_argument_group("spike detection (GUARD Sec. 5.2: z = (M - mean) / std,  spike if z > tau)")
    g.add_argument("--tau", type=float, default=3.0, help="z-score threshold. Default: 3.0")
    g.add_argument("--score", choices=SCORES, default="max_logit",
                   help="M_i: max_logit = max_q pre-softmax logit (GUARD code), max_prob = max_q softmax prob "
                        "(paper text), mass = mean_q prob. Default: max_logit")
    g.add_argument("--head-agg", choices=AGGS, default="none",
                   help="aggregate M over heads before the z-score (none = per head, as in GUARD)")
    g.add_argument("--frame-agg", choices=AGGS, default="none",
                   help="aggregate M over frames before the z-score (none = per frame)")


def add_plot_args(p):
    g = p.add_argument_group("plots")
    g.add_argument("--plots", default="profile,heads,steps,crossmaps,selfmaps,temporal,summary",
                   help=f"comma list of {','.join(PLOT_KINDS)} or 'none'")
    g.add_argument("--plot-steps", default=None,
                   help="steps for per-step plots (default: the map steps). Must have been recorded.")
    g.add_argument("--tokens", default=None,
                   help="token positions for the 'steps' plot, e.g. '5,9' (default: <eot> + detected spikes)")
    g.add_argument("--map-head", default="mean", help="head shown in crossmaps: 'mean' or a head index")
    g.add_argument("--query-frame", type=int, default=None, help="query frame for temporal spatial maps (default: middle)")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m animatediff_attn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list-layers", help="print every attention layer of the AnimateDiff UNet")
    pl.add_argument("--model-id", default=MODEL_ID)
    pl.add_argument("--adapter-id", default=MOTION_ADAPTER_ID)
    pl.add_argument("--height", type=int, default=512)
    pl.add_argument("--width", type=int, default=512)
    add_selection_args(pl)

    pr = sub.add_parser("run", help="generate a video while recording attention, then detect spikes and plot")
    g = pr.add_argument_group("generation")
    g.add_argument("--prompt", default=None)
    g.add_argument("--prompts-file", default=None, help="text file, one prompt per line (each gets its own run dir)")
    g.add_argument("--negative-prompt", default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num-frames", type=int, default=16)
    g.add_argument("--height", type=int, default=512)
    g.add_argument("--width", type=int, default=512)
    g.add_argument("--num-inference-steps", type=int, default=25)
    g.add_argument("--guidance-scale", type=float, default=7.5)
    g.add_argument("--model-id", default=MODEL_ID)
    g.add_argument("--adapter-id", default=MOTION_ADAPTER_ID)
    g.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    g.add_argument("--device", default="cuda")
    g.add_argument("--cpu-offload", action="store_true", help="enable model CPU offload (lower VRAM)")
    g.add_argument("--out-dir", default="outputs")
    g.add_argument("--run-name", default=None, help="default: slug of the prompt + seed")
    r = pr.add_argument_group("recording")
    r.add_argument("--record-steps", default="all",
                   help="steps at which per-position statistics are recorded (cross & temporal). Default: all")
    r.add_argument("--map-steps", default="0,-1",
                   help="steps at which full attention maps are stored (and spatial self-attn is analysed). "
                        "Default: '0,-1' (first and last)")
    r.add_argument("--map-tokens", default="valid",
                   help="cross-attn token positions stored in maps: 'valid' (<sot>..<eot>), 'all' or '1,4,7'")
    r.add_argument("--query-points", default="0.5,0.5",
                   help="spatial self-attn query points as normalised y,x pairs separated by ';', "
                        "e.g. '0.5,0.5;0.25,0.75'")
    r.add_argument("--branch", choices=["cond", "uncond"], default="cond",
                   help="classifier-free-guidance branch to analyse. Default: cond")
    add_selection_args(pr)
    add_spike_args(pr)
    add_plot_args(pr)

    pp = sub.add_parser("plot", help="re-plot a recorded run with other selections / thresholds")
    pp.add_argument("--run-dir", required=True)
    add_selection_args(pp)
    add_spike_args(pp)
    add_plot_args(pp)

    ps = sub.add_parser("spikes", help="write the table of detected spikes for a recorded run")
    ps.add_argument("--run-dir", required=True)
    ps.add_argument("--steps", default="all", help="recorded steps to analyse. Default: all")
    ps.add_argument("--top", type=int, default=20, help="number of rows printed per attention type")
    add_selection_args(ps)
    add_spike_args(ps)
    return p


# --------------------------------------------------------------------------- helpers
def _opt_list(spec, n):
    return None if spec is None or str(spec).lower() == "all" else parse_index_list(spec, n)


def _slug(text, n=40):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")[:n] or "prompt"


def _parse_points(spec):
    pts = []
    for part in spec.split(";"):
        y, x = (float(v) for v in part.split(","))
        pts.append((y, x))
    return pts


def _selected_names(record, args):
    layers = [LayerInfo(**d) for d in record["layers"].values()]
    layers.sort(key=lambda l: l.idx)
    sel = select_layers(layers, args.attn_types, args.blocks, args.layers, None)
    if args.layer_ids:
        ids = set(parse_index_list(args.layer_ids, max(l.idx for l in layers) + 1))
        sel = [l for l in sel if l.idx in ids]
    return [l.name for l in sel]


def _record_dims(record):
    cfg = record["config"]
    n_heads = max(d["heads"] for d in record["layers"].values())
    return cfg["num_frames"], n_heads, len(record["timesteps"])


def load_record(run_dir: Path) -> dict:
    return torch.load(run_dir / "record.pt", map_location="cpu", weights_only=False)


# -------------------------------------------------------------------------- commands
def cmd_list_layers(args):
    from .model import load_unet_only
    from .selection import enumerate_layers

    layers = enumerate_layers(load_unet_only(args.model_id, args.adapter_id))
    sel = select_layers(layers, args.attn_types, args.blocks, args.layers, args.layer_ids)
    print(format_layer_table(sel, args.height // 8, args.width // 8))
    counts = {t: sum(l.attn_type == t for l in sel) for t in ("spatial", "cross", "temporal")}
    print(f"\n{len(sel)} layers selected: {counts}")


def cmd_run(args):
    from .model import generate, load_pipeline, token_labels
    from .recorder import AttentionRecorder, RecordConfig
    from .selection import enumerate_layers

    if not args.prompt and not args.prompts_file:
        raise SystemExit("give --prompt or --prompts-file")
    prompts = [args.prompt] if args.prompt else [
        l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]

    pipe = load_pipeline(args.model_id, args.adapter_id, args.device, args.dtype, args.cpu_offload)
    layers = enumerate_layers(pipe.unet)
    sel = select_layers(layers, args.attn_types, args.blocks, args.layers, args.layer_ids)
    if not sel:
        raise SystemExit("layer selection is empty - check with list-layers")
    print(f"recording {len(sel)} attention layers")

    max_heads = max(l.heads for l in sel)
    for i, prompt in enumerate(prompts):
        labels, eot = token_labels(pipe.tokenizer, prompt)
        if args.map_tokens == "valid":
            map_tokens = list(range(eot + 1))
        elif args.map_tokens == "all":
            map_tokens = None
        else:
            map_tokens = parse_index_list(args.map_tokens, len(labels))
        cfg = RecordConfig(
            num_frames=args.num_frames,
            latent_h=args.height // pipe.vae_scale_factor,
            latent_w=args.width // pipe.vae_scale_factor,
            stat_steps=parse_index_list(args.record_steps, args.num_inference_steps),
            map_steps=parse_index_list(args.map_steps, args.num_inference_steps),
            heads=_opt_list(args.heads, max_heads),
            frames=_opt_list(args.frames, args.num_frames),
            map_tokens=map_tokens,
            query_points=_parse_points(args.query_points),
            branch=args.branch,
            cfg=args.guidance_scale > 1.0,
        )
        recorder = AttentionRecorder(cfg, sel)
        frames = generate(pipe, recorder, prompt, args.negative_prompt, args.num_frames, args.height, args.width,
                          args.num_inference_steps, args.guidance_scale, args.seed)

        name = args.run_name or f"{_slug(prompt)}_seed{args.seed}"
        if len(prompts) > 1:
            name = f"{i:03d}_{name}"
        run_dir = Path(args.out_dir) / name
        (run_dir / "frames").mkdir(parents=True, exist_ok=True)
        for j, frame in enumerate(frames):
            frame.save(run_dir / "frames" / f"frame_{j:03d}.png")
        frames[0].save(run_dir / "video.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)

        meta = {
            "prompt": prompt, "negative_prompt": args.negative_prompt, "seed": args.seed,
            "model_id": args.model_id, "adapter_id": args.adapter_id, "num_frames": args.num_frames,
            "height": args.height, "width": args.width, "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale, "token_labels": labels, "eot_index": eot,
        }
        record = {"meta": meta, **recorder.state_dict()}
        torch.save(record, run_dir / "record.pt")
        with open(run_dir / "meta.json", "w") as fh:
            json.dump({**meta, "config": record["config"], "layers": record["layers"],
                       "resolutions": record["resolutions"], "timesteps": record["timesteps"]}, fh, indent=2)
        print(f"saved {run_dir}")

        run_spikes(record, run_dir, args, steps=None, top=10)
        if args.plots != "none":
            run_plots(record, run_dir, args)


def run_plots(record, run_dir, args):
    from .plotting import make_plots

    n_frames, n_heads, n_steps = _record_dims(record)
    kinds = [k.strip() for k in args.plots.split(",") if k.strip()]
    bad = set(kinds) - set(PLOT_KINDS)
    if bad:
        raise SystemExit(f"unknown plot kinds {bad}; choose from {PLOT_KINDS}")
    paths = make_plots(
        record, run_dir, _selected_names(record, args), kinds,
        steps=None if args.plot_steps is None else parse_index_list(args.plot_steps, n_steps),
        score=args.score, tau=args.tau,
        heads=_opt_list(args.heads, n_heads), frames=_opt_list(args.frames, n_frames),
        tokens=None if args.tokens is None else parse_index_list(args.tokens, len(record["meta"]["token_labels"])),
        map_head=args.map_head, query_frame=args.query_frame,
    )
    print(f"wrote {len(paths)} plots to {run_dir / 'plots'}")


def run_spikes(record, run_dir, args, steps=None, top=20):
    from .spikes import find_spikes, summarize, write_csv

    n_frames, n_heads, n_steps = _record_dims(record)
    names = _selected_names(record, args)
    steps = list(range(n_steps)) if steps is None else steps
    heads, frames = _opt_list(args.heads, n_heads), _opt_list(args.frames, n_frames)
    rows = find_spikes(record, names, steps, args.tau, args.score, heads, frames, args.head_agg, args.frame_agg)
    tag = f"{args.score}_tau{args.tau:g}_h-{args.head_agg}_f-{args.frame_agg}"
    write_csv(rows, run_dir / f"spikes_{tag}.csv")
    summary = summarize(rows, record, names, steps, heads, frames, args.head_agg, args.frame_agg)
    with open(run_dir / f"spikes_summary_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nspikes ({tag}): {len(rows)} detected -> {run_dir / f'spikes_{tag}.csv'}")
    for attn_type, s in summary.items():
        print(f"  {attn_type}: {s['num_spikes']} spikes over {s['num_tests']} (layer, step, frame, head) tests")
        for item in s["top_positions"][:top]:
            print(f"    {item['position']:<24} flagged {item['count']:>6}x  ({100 * item['fraction_of_tests']:.1f}% of tests)")


def cmd_plot(args):
    run_dir = Path(args.run_dir)
    run_plots(load_record(run_dir), run_dir, args)


def cmd_spikes(args):
    run_dir = Path(args.run_dir)
    record = load_record(run_dir)
    _, _, n_steps = _record_dims(record)
    run_spikes(record, run_dir, args, parse_index_list(args.steps, n_steps), args.top)


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"list-layers": cmd_list_layers, "run": cmd_run, "plot": cmd_plot, "spikes": cmd_spikes}[args.cmd](args)


if __name__ == "__main__":
    main()
