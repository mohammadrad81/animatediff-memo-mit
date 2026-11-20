"""Video quality of the generated videos: FVD (realism) and ViCLIP similarity (does the video show its caption?), for
the plain videos of the fine-tuned model (normal) and every mitigation method, and the original model. See section 10
of README.md (step 8).

Steps:
  python quality_eval.py check-viclip     once: ViCLIP must reproduce the result of its official demo
  python quality_eval.py reference        real clips: the FVD reference, and the scores of the real training clips
  python quality_eval.py metrics --experiments normal,guard,soften,clip,noise,sail
  python quality_eval.py --out outputs/finetune_eval_original512 metrics   (the original model at 512x512, optional)
  python quality_eval.py summary          table and figures of all of them, in outputs/finetune_quality

FVD (Unterthiner et al., 2018): the Frechet distance between the I3D features of the generated videos and of real
videos. Lower = the generated videos look more like real WebVid videos. The I3D network is the TorchScript model of
StyleGAN-V (Skorokhodov et al., 2022), used by most video generation papers.

ViCLIP similarity (Wang et al., InternVid, 2023): the cosine similarity between the ViCLIP features of the video and of
its caption. Higher = the video matches the caption better. ViCLIP-L/14, as in its official demo: 8 frames, 224x224.
"""
import argparse
import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from finetune import read_clip
from memorization_eval import (BACKGROUND_COLOR, EXPERIMENTS, GRID, INK, MEMORIZED_COLOR, MUTED, NO_MITIGATION_COLOR,
                               ORIGINAL_COLOR, ORIGINAL_DEFAULT, experiment_list, pyplot, result_file, save_figures)
from webvid import gif_frames, square, video_path

VIDEO_SIZE = 256  # the size of the fine-tuning and of the generated videos; the real clips are cropped and resized to it

# I3D of StyleGAN-V, and its checksum (the file has been downloaded from this address on 2026-11-16).
I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"

# ViCLIP-L/14 in the Hugging Face format, at a fixed version: its code is downloaded and run (trust_remote_code), so
# the version whose code was read is pinned.
VICLIP_REPO = "OpenGVLab/ViCLIP-L-14-hf"
VICLIP_REVISION = "1652361522e1cb41c28cdfae870f690d00e7456b"
VICLIP_FRAMES, VICLIP_SIZE = 8, 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ViCLIP's demo normalizes with these


# ------------------------------------------------------------------------------------------------------------------
# Part 1: the two networks
# ------------------------------------------------------------------------------------------------------------------

def load_i3d(path):
    """The I3D network of FVD; downloaded the first time, and checked against the known checksum."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading the I3D network to {path}")
        urllib.request.urlretrieve(I3D_URL, path + ".tmp")
        os.replace(path + ".tmp", path)
    with open(path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != I3D_SHA256:
            raise SystemExit(f"{path} is not the expected I3D network (checksum differs); delete it to download it again")
    return torch.jit.load(path, map_location="cuda").eval()


@torch.no_grad()
def i3d_features(i3d, videos):
    """videos: float [B, frames, 3, H, W] in [0, 1] -> [B, 400] features (the logits before the softmax), as
    StyleGAN-V computes FVD: values 0-255, resized to 224x224 by the network itself."""
    x = videos.permute(0, 2, 1, 3, 4).contiguous().cuda() * 255  # [B, 3, frames, H, W]; the network needs contiguous
    return i3d(x, rescale=True, resize=True, return_features=True).double().cpu()


def load_viclip():
    """ViCLIP-L/14 from Hugging Face. Its model code comes with the weights (trust_remote_code).
    The model is built and its weights loaded here instead of with AutoModel.from_pretrained: ViCLIP's code was written
    for transformers 4 and never calls post_init(), which from_pretrained of transformers 5 needs
    (AttributeError: 'ViCLIP' object has no attribute 'all_tied_weights_keys')."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    path = snapshot_download(VICLIP_REPO, revision=VICLIP_REVISION,
                             allow_patterns=["*.py", "*.json", "*.txt.gz", "model.safetensors"])
    model_class = get_class_from_dynamic_module("viclip.ViCLIP", path)
    config = model_class.config_class.from_pretrained(path)
    config.tokenizer_path = os.path.join(path, "bpe_simple_vocab_16e6.txt.gz")  # the config's path is relative
    model = model_class(config)
    missing, unexpected = model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=False)
    if missing or unexpected:  # every weight must come from the file: a missing one would stay random
        raise SystemExit(f"ViCLIP weights do not match the model: missing {missing[:5]}..., unexpected {unexpected[:5]}...")
    return model.float().cuda().eval()


@torch.no_grad()
def viclip_video(viclip, videos):
    """videos: float [B, frames, 3, H, W] in [0, 1] -> [B, 768] unit vectors. As the official demo: 8 frames evenly
    spaced (every 2nd of 16), resized to 224x224, normalized with the ImageNet mean and std."""
    step = max(1, videos.shape[1] // VICLIP_FRAMES)
    x = videos[:, ::step][:, :VICLIP_FRAMES]
    b, t = x.shape[:2]
    x = F.interpolate(x.flatten(0, 1), size=(VICLIP_SIZE, VICLIP_SIZE), mode="bilinear", align_corners=False)
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = ((x - mean) / std).view(b, t, 3, VICLIP_SIZE, VICLIP_SIZE)
    return viclip.get_vid_features(x.cuda()).float().cpu()


@torch.no_grad()
def viclip_text(viclip, captions):
    """captions -> [B, 768] unit vectors. ViCLIP reads at most 32 tokens: longer captions are cut."""
    return F.normalize(viclip.encode_text(list(captions)).float(), dim=-1).cpu()


# ------------------------------------------------------------------------------------------------------------------
# Part 2: reading videos and scoring them
# ------------------------------------------------------------------------------------------------------------------

def generated_video(gif):
    """A generated video: float [frames, 3, 256, 256] in [0, 1]."""
    return square(gif_frames(gif), VIDEO_SIZE, crop=False)


def real_video(args, videoid):
    """A real clip as the model was fine-tuned on it: 16 frames (every 4th), center square, 256x256."""
    return square(read_clip(video_path(args, videoid)), VIDEO_SIZE, crop=True)


def score_videos(load, ids, captions, i3d, viclip, batch):
    """I3D features and ViCLIP similarity of every video. load(id) -> one video. -> ([n, 400], [n])."""
    features, similarity = [], []
    for b in tqdm(range(0, len(ids), batch), desc="videos"):
        videos = torch.stack([load(v) for v in ids[b:b + batch]])
        features.append(i3d_features(i3d, videos))
        if viclip is not None:
            similarity.append((viclip_video(viclip, videos) * viclip_text(viclip, captions[b:b + batch])).sum(-1))
    return torch.cat(features).numpy(), (torch.cat(similarity).numpy() if similarity else None)


def frechet_distance(a, b):
    """FVD of two sets of I3D features [n, 400]: |mu_a - mu_b|^2 + Tr(S_a + S_b - 2 (S_a S_b)^(1/2))."""
    from scipy.linalg import sqrtm
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = sqrtm(s_a @ s_b).real  # tiny imaginary parts come from rounding
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(s_a + s_b - 2 * root))


def read_sheet(args):
    return pd.read_csv(os.path.join(args.out, "prompts.csv"), keep_default_na=False).rename(columns={"index": "videoid"})


# ------------------------------------------------------------------------------------------------------------------
# Part 3: the commands
# ------------------------------------------------------------------------------------------------------------------

def run_check_viclip(args):
    """Compare ViCLIP with the result printed in its official demo (demo.ipynb of the model, same video and texts)."""
    import av
    from huggingface_hub import hf_hub_download
    viclip = load_viclip()
    with av.open(hf_hub_download(VICLIP_REPO, "example1.mp4", revision=VICLIP_REVISION)) as container:
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    step = len(frames) // VICLIP_FRAMES  # the demo's sampling, on the whole video
    video = torch.from_numpy(frames[::step][:VICLIP_FRAMES]).permute(0, 3, 1, 2).float()[None] / 255
    texts = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
             "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
             "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
             "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
             "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
             "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
             "A playful dog slides down a snowy hill, wagging its tail with delight.",
             "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
             "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
             "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]
    cosine = (viclip_video(viclip, video) @ viclip_text(viclip, texts).T)[0]
    probs = (100 * cosine).softmax(-1)
    for i in probs.argsort(descending=True)[:3]:
        print(f"{probs[i]:.4f}  cosine {cosine[i]:.4f}  {texts[i]}")
    print("expected (demo): 0.8333 'A man in a gray sweater plays fetch...', 0.1266 'A playful dog and its owner...', "
          "0.0368 'A pet dog excitedly runs...'. Small differences come from the resizing (the demo uses OpenCV).")


def run_reference(args):
    """The real clips: the FVD reference set (random background clips of the fine-tuning, but not the 400 evaluated
    ones, so that a copy of its own training clip does not lower FVD) and the 400 training clips of the evaluated
    captions (the best scores real videos get with this many videos)."""
    os.makedirs(args.save_to, exist_ok=True)
    sheet = read_sheet(args)
    table = pd.read_csv(args.table, keep_default_na=False)
    pool = table[(table["role"] == "background") & ~table["videoid"].isin(sheet["videoid"])]
    reference = pool.sample(n=min(args.size, len(pool)), random_state=args.seed)

    i3d = load_i3d(args.i3d)
    viclip = load_viclip()
    load = lambda v: real_video(args, v)
    print(f"FVD reference: {len(reference)} real clips")
    features, _ = score_videos(load, reference["videoid"].tolist(), reference["caption"].tolist(), i3d, None, args.batch)
    np.savez(os.path.join(args.save_to, "fvd_reference.npz"), features=features, videoid=reference["videoid"].to_numpy(),
             seed=args.seed)

    print(f"the {len(sheet)} training clips of the evaluated captions")
    features, similarity = score_videos(load, sheet["videoid"].tolist(), sheet["caption"].tolist(), i3d, viclip, args.batch)
    np.save(os.path.join(args.save_to, "i3d_training.npy"), features)
    out = sheet[["videoid", "role", "copies", "caption"]].assign(viclip=similarity)
    out.to_csv(os.path.join(args.save_to, "quality_training.csv"), index=False, float_format="%.4f")
    print(f"-> {args.save_to}: fvd_reference.npz, i3d_training.npy, quality_training.csv")


def run_metrics(args):
    """I3D features and ViCLIP similarity of the generated videos of each experiment, in <out>/<experiment>/ ->
    i3d_features.npy and quality.csv in --out (with the experiment's name added, except for normal)."""
    experiments = experiment_list(args.experiments)
    sheet = read_sheet(args)
    gif = lambda experiment, v: os.path.join(args.out, experiment, f"{v}.gif")
    for experiment in experiments:  # FVD over part of the videos could not be compared
        missing = [v for v in sheet["videoid"] if not os.path.exists(gif(experiment, v))]
        if missing:
            raise SystemExit(f"{experiment}: {len(missing)} of {len(sheet)} videos are not generated yet in {args.out}/{experiment}")

    i3d, viclip = load_i3d(args.i3d), load_viclip()
    for experiment in experiments:
        print(EXPERIMENTS[experiment])
        features, similarity = score_videos(lambda v: generated_video(gif(experiment, v)), sheet["videoid"].tolist(),
                                            sheet["caption"].tolist(), i3d, viclip, args.batch)
        np.save(result_file(args.out, "i3d_features.npy", experiment), features)
        out = sheet[["videoid", "role", "copies", "caption"]].assign(viclip=similarity)
        out.to_csv(result_file(args.out, "quality.csv", experiment), index=False, float_format="%.4f")
        print(f"   ViCLIP mean: memorized {similarity[sheet['role'] == 'memorized'].mean():.4f}, "
              f"background {similarity[sheet['role'] == 'background'].mean():.4f}")
    print(f"-> {args.out}: i3d_features*.npy, quality*.csv")


# ------------------------------------------------------------------------------------------------------------------
# Part 4: the table and the figures
# ------------------------------------------------------------------------------------------------------------------

GROUPS = [("All", None), ("Memorized samples", "memorized"), ("Background samples", "background")]


def label_of(name):
    """Name under a bar or next to a point, on two lines if it is long."""
    names = {"training": "Real training\nclips", "original": "Original model"}
    return names.get(name, EXPERIMENTS.get(name, name).replace(" ", "\n", 1))


def load_results(args):
    """{name: (quality table, I3D features, share of the memorized samples with GSSCD >= threshold or None)} in the
    order: real training clips, original model (if measured), the experiments."""
    results = {}
    training = os.path.join(args.save_to, "quality_training.csv")
    if not os.path.exists(training):
        raise SystemExit(f"{training} is missing: run `quality_eval.py reference` first")
    results["training"] = (pd.read_csv(training, keep_default_na=False), np.load(os.path.join(args.save_to, "i3d_training.npy")), None)
    folders = ([("original", args.original, "normal")] if args.original else []) + \
              [(e, args.out, e) for e in experiment_list(args.experiments)]
    for name, folder, experiment in folders:
        path = result_file(folder, "quality.csv", experiment)
        if not os.path.exists(path):
            if name == "original":
                continue  # the original model is optional
            raise SystemExit(f"{path} is missing: run `quality_eval.py --out {folder} metrics --experiments {experiment}` first")
        memorization = result_file(folder, "metrics.csv", experiment)
        copied = None
        if os.path.exists(memorization):  # the memorization of the same videos (memorization_eval.py metrics)
            m = pd.read_csv(memorization, keep_default_na=False, na_values=[""])
            copied = 100 * (m.loc[m["role"] == "memorized", "gsscd"] >= args.gsscd_threshold).mean()
        results[name] = (pd.read_csv(path, keep_default_na=False), np.load(result_file(folder, "i3d_features.npy", experiment)), copied)
    order = results["training"][0]["videoid"].tolist()
    for name, (table, _, _) in results.items():
        if table["videoid"].tolist() != order:
            raise SystemExit(f"{name}: other videos (or another order) than prompts.csv; generate from the same prompts.csv")
    return results


def quality_summary(results, reference):
    """Per row (real clips, original model, experiments) and group: FVD, mean ViCLIP similarity, its change against no
    mitigation (the same caption and seed, so each video against its own twin) with the Wilcoxon p-value, and the
    share of memorized samples copied (GSSCD)."""
    from scipy.stats import wilcoxon
    rows = []
    normal = results.get("normal", (None,))[0]
    for name, (table, features, copied) in results.items():
        for group, role in GROUPS:
            members = (table["role"] == role) if role else np.ones(len(table), dtype=bool)
            similarity = table.loc[members, "viclip"]
            row = {"name": name, "group": group, "videos": int(members.sum()),
                   "fvd": frechet_distance(features[np.asarray(members)], reference),
                   "viclip_mean": similarity.mean(), "viclip_std": similarity.std()}
            if normal is not None and name not in ("normal", "training"):
                change = similarity.to_numpy() - normal.loc[members, "viclip"].to_numpy()
                row["viclip_change_vs_normal"] = change.mean()
                row["wilcoxon_p_vs_normal"] = wilcoxon(change).pvalue if np.any(change) else np.nan
            row["memorized_copied_%"] = copied if group == "Memorized samples" else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_quality_bars(summary, plt):
    """FVD (lower is better) and ViCLIP similarity (higher is better) of every row, memorized and background samples."""
    names = list(dict.fromkeys(summary["name"]))
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.55 * len(names)), 8.6), sharex=True)
    for ax, (column, title) in zip(axes, [("fvd", "FVD against real WebVid clips (lower = more realistic)"),
                                          ("viclip_mean", "ViCLIP similarity to the caption (higher = matches the caption better)")]):
        for offset, group, color in [(-0.19, "Memorized samples", MEMORIZED_COLOR), (0.19, "Background samples", BACKGROUND_COLOR)]:
            values = [summary.loc[(summary["name"] == n) & (summary["group"] == group), column].iloc[0] for n in names]
            bars = ax.bar(x + offset, values, width=0.36, color=color, label=group)
            ax.bar_label(bars, labels=[f"{v:.0f}" if column == "fvd" else f"{v:.3f}" for v in values],
                         padding=2, fontsize=9.5, color=INK)
        references = sum(n in ("training", "original") for n in names)
        if 0 < references < len(names):  # the references are not methods: set them apart
            ax.axvline(references - 0.5, color=GRID, linewidth=1.5)
        ax.set_title(title, loc="left")
        ax.margins(y=0.12)
        ax.grid(False, axis="x")
    axes[0].set_ylabel("FVD")
    axes[1].set_ylabel("Mean cosine similarity")
    axes[-1].set_xticks(x, [label_of(n) for n in names], fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.035), fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.5, 0.005, "FVD of 200 videos per group is noisy and higher than with thousands; compare the rows with each "
             "other, and with the real training clips (the best 200 videos can get).", ha="center", fontsize=10.5, color=MUTED)
    return fig


def plot_quality_vs_memorization(summary, plt):
    """The trade-off, memorized samples only: how often their training clip is still copied (x) against their quality
    (y). The best place is low copying with good quality: left, and high for ViCLIP / low for FVD."""
    rows = summary[(summary["group"] == "Memorized samples") & summary["memorized_copied_%"].notna()]
    training = summary[(summary["name"] == "training") & (summary["group"] == "Memorized samples")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (column, title) in zip(axes, [("viclip_mean", "ViCLIP similarity (higher is better)"),
                                          ("fvd", "FVD (lower is better)")]):
        for _, row in rows.iterrows():
            color = {"original": ORIGINAL_COLOR, "normal": NO_MITIGATION_COLOR}.get(row["name"], MEMORIZED_COLOR)
            ax.scatter(row["memorized_copied_%"], row[column], s=90, color=color, edgecolors="white", linewidths=1.5, zorder=3)
            # Methods are labelled on their right. The references (no mitigation, original model) are labelled on the
            # side away from the methods, which usually cost some quality: above for ViCLIP, below for FVD.
            if row["name"] in ("original", "normal"):
                above = column == "viclip_mean"
                offset, ha, va = (0, 9 if above else -9), "center", "bottom" if above else "top"
            else:
                offset, ha, va = (7, 5), "left", "baseline"
            ax.annotate(label_of(row["name"]).replace("\n", " "), (row["memorized_copied_%"], row[column]),
                        xytext=offset, ha=ha, va=va, textcoords="offset points", fontsize=10.5, color=INK)
        if len(training):  # real videos: the quality to aim for
            ax.axhline(training[column].iloc[0], color=MUTED, linestyle="--", linewidth=1.2)
            ax.text(0.99, training[column].iloc[0], "real training clips", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=10, color=MUTED)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Memorized samples still copied (% with GSSCD ≥ threshold)")
        ax.set_xlim(-5, max(100, rows["memorized_copied_%"].max() + 10))
        ax.margins(y=0.15)
    axes[0].set_ylabel("Memorized samples: mean ViCLIP similarity")
    axes[1].set_ylabel("Memorized samples: FVD")
    fig.tight_layout()
    return fig


def run_summary(args):
    """Table and figures of the quality of every row, in --save-to."""
    plt = pyplot()
    results = load_results(args)
    reference = np.load(os.path.join(args.save_to, "fvd_reference.npz"))["features"]
    summary = quality_summary(results, reference)
    summary.to_csv(os.path.join(args.save_to, "quality_summary.csv"), index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(summary.round(4).to_string(index=False))
    figures = {"quality_scores": plot_quality_bars(summary, plt)}
    if summary["memorized_copied_%"].notna().any():
        figures["quality_vs_memorization"] = plot_quality_vs_memorization(summary, plt)
    save_figures(figures, args.save_to, None, plt)
    print(f"-> {args.save_to}: quality_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="FVD and ViCLIP similarity of the videos of memorization_eval.py")
    parser.add_argument("--root", default="data/webvid", help="the WebVid folder of webvid.py (where the videos are)")
    parser.add_argument("--out", default="outputs/finetune_eval", help="folder of prompts.csv and the generated videos")
    parser.add_argument("--save-to", default="outputs/finetune_quality", help="folder of the reference, the table and the figures")
    parser.add_argument("--i3d", default="data/webvid/models/fvd/i3d_torchscript.pt", help="I3D network; downloaded if missing")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-viclip", help="ViCLIP on the video of its official demo: compare with the demo's result")

    r = sub.add_parser("reference", help="real clips: the FVD reference and the scores of the real training clips")
    r.add_argument("--table", default="outputs/finetune/finetune.csv", help="the table the model was fine-tuned on")
    r.add_argument("--size", type=int, default=2048, help="real clips in the FVD reference")
    r.add_argument("--seed", type=int, default=0, help="which real clips are drawn")
    r.add_argument("--batch", type=int, default=8, help="videos per pass")

    m = sub.add_parser("metrics", help="I3D features and ViCLIP similarity of the generated videos")
    m.add_argument("--experiments", default="normal", help=f"any of {','.join(EXPERIMENTS)}; videos in <out>/<experiment>/")
    m.add_argument("--batch", type=int, default=8, help="videos per pass")

    s = sub.add_parser("summary", help="FVD, ViCLIP and the memorization of every method: table and figures")
    s.add_argument("--experiments", default=",".join(EXPERIMENTS), help="experiments in --out to include")
    s.add_argument("--original", default=ORIGINAL_DEFAULT,
                   help="folder of the original model's videos, included if measured ('' = leave out)")
    s.add_argument("--gsscd-threshold", type=float, default=0.4, help="GSSCD from which a sample counts as copied")

    args = parser.parse_args()
    {"check-viclip": run_check_viclip, "reference": run_reference, "metrics": run_metrics,
     "summary": run_summary}[args.command](args)


if __name__ == "__main__":
    main()
