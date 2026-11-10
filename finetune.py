"""Artificial memorization: fine-tune AnimateDiff's UNet (image layers and motion modules; CLIP and the VAE stay frozen)
on a dataset in which some videos are duplicated, so that the model memorizes them, the more copies the more strongly. See section 10 of README.md.

The dataset has two kinds of samples, all with different videos, different captions and different links:
  background   20000 random WebVid-10M videos (--background), one copy each: they keep the training distribution
  memorized    200 more videos (--memorized), 4 of them (--per-level) with 2 copies, 4 with 3, 4 with 4, ...

Steps:
  python finetune.py select --candidates 25000                    draw the random candidates
  python webvid.py download --ids-from <candidates> --per-caption 0   download them
  python finetune.py build --background 20000 --memorized 200     keep the readable ones, assign the copies
  accelerate launch ... finetune.py train                         fine-tune the UNet
"""
import argparse
import glob
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the DataLoader workers tokenize the captions
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for reproducible results (see set_seed below)

import av
import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from diffusers import DDPMScheduler
from tqdm import tqdm

from sampler import MODEL_ID, checkpointed_blocks, load_pipeline
from webvid import caption_groups, load_train, normalize, path, square, video_path

# The clips, as AnimateDiff was trained: 16 frames, every 4th frame of the video.
FRAMES = 16
STRIDE = 4


# ------------------------------------------------------------------------------------------------------------------
# Part 1: choosing the samples
# ------------------------------------------------------------------------------------------------------------------

def run_select(args):
    """Draw the candidates at random from ALL of WebVid-10M, and write them in the order they were drawn.

    They must not be chosen among the videos already downloaded: `webvid.py download` starts with the most repeated
    captions, so those videos do not look like the training set. More candidates than needed are drawn because
    some download links are dead and some videos cannot be read (`build` takes the first ones that work)."""
    videos = caption_groups(args)  # videoid, name (the caption), key (normalized caption), group (videos with it)
    videos = videos.merge(load_train(args, ["videoid", "contentUrl"]), on="videoid", how="left")  # the download links
    # Shuffle the videos, THEN keep the first video of each caption. Every video has the same chance, so frequent
    # captions stay more likely than rare ones, as in the training set; each caption is kept only once.
    videos = videos.sample(frac=1, random_state=args.seed).drop_duplicates("key")
    # The same for the links. The memorized samples are taken from these candidates too, so all of them are unique.
    videos = videos.drop_duplicates("contentUrl").head(args.candidates)
    out = path(args, "finetune", "candidates.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    videos = videos.rename(columns={"name": "caption", "group": "caption_count"})
    videos[["videoid", "contentUrl", "caption", "caption_count"]].to_csv(out, index=False)
    print(f"{len(videos)} candidates -> {out}\nnow download them:  python webvid.py download --ids-from {out} --per-caption 0")


def read_clip(file):
    """The clip of a video: 16 frames from its first 61 frames (the same frames every time, so that the copies of a
    video really are the same sample). Shorter videos: 16 frames evenly spaced over the whole video.
    -> uint8 numpy array [frames, H, W, 3]."""
    wanted = (FRAMES - 1) * STRIDE + 1
    frames = []
    with av.open(file) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == wanted:
                break  # the rest of the video is not needed
    if len(frames) < FRAMES:
        raise ValueError(f"{file}: only {len(frames)} frames")
    return np.stack([frames[i] for i in np.linspace(0, len(frames) - 1, FRAMES).round().astype(int)])


def can_read(file):
    """True if the video exists and its clip can be decoded."""
    try:
        read_clip(file)
        return True
    except Exception:  # missing (dead link) or broken file
        return False


def duplication_frequencies(n):
    """The first n natural numbers except 1: 2, 3, 4, ..., n + 1. To try another spectrum of copies, change this function."""
    return list(range(2, n+2))


def run_build(args):
    """Write finetune.csv (videoid, contentUrl, caption, copies, role), the table the training reads, and memorized_prompts.csv,
    the sheet to generate the memorized videos with afterwards (`main.py batch`)."""
    candidates = pd.read_csv(path(args, "finetune", "candidates.csv"), keep_default_na=False)
    files = [video_path(args, v) for v in candidates["videoid"]]
    print(f"checking that {len(files)} candidates can be read")
    with ProcessPoolExecutor(args.workers) as pool:
        usable = candidates[list(pool.map(can_read, files, chunksize=32))]
    print(f"{len(usable)} candidates can be read")

    # The videos to memorize: only captions no other WebVid video has, so a caption points to exactly one video.
    memorized = usable[usable["caption_count"] == 1].head(args.memorized).copy()
    # The background: the first videos that are left, whatever their caption count, so the draw stays random.
    background = usable.drop(memorized.index).head(args.background).copy()
    if len(memorized) < args.memorized or len(background) < args.background:
        raise SystemExit(f"only {len(memorized)} memorized and {len(background)} background samples: run `select` "
                         "with more --candidates (the first ones stay the same), then download again")

    # The spectrum of memorization: `per_level` videos for each number of copies. Rows were drawn at random, so which
    # video gets how many copies is random too. Background videos have one copy: they are the "0 duplicates" control.
    # If --memorized is not a multiple of --per-level, the last level simply has fewer videos.
    levels = (args.memorized + args.per_level - 1) // args.per_level
    memorized["copies"] = np.repeat(duplication_frequencies(levels), args.per_level)[:args.memorized]
    memorized["role"] = "memorized"
    background["copies"] = 1
    background["role"] = "background"
    table = pd.concat([background, memorized])[["videoid", "contentUrl", "caption", "copies", "role"]]

    # Every video, caption and link appears only once in the table; the duplication is the `copies` column.
    keys = table["caption"].map(normalize)
    assert table["videoid"].is_unique and table["contentUrl"].is_unique and keys.is_unique
    table.to_csv(path(args, "finetune", "finetune.csv"), index=False)
    memorized.rename(columns={"videoid": "index"})[["index", "caption", "copies"]].to_csv(
        path(args, "finetune", "memorized_prompts.csv"), index=False)

    # Does the background look like WebVid-10M?
    train = load_train(args, ["videoid", "duration", "name"])
    words = train["name"].str.split().str.len()
    chosen = train["videoid"].isin(background["videoid"])
    print("                 all of WebVid      background")
    for name, values in [("duration (s)", train["duration"]), ("caption (words)", words)]:
        print(f"{name:16s} mean {values.mean():5.1f} median {values.median():3.0f}  "
              f"mean {values[chosen].mean():5.1f} median {values[chosen].median():3.0f}")
    print(f"{len(background)} background + {len(memorized)} memorized samples, {table['copies'].sum()} rows per epoch")
    print(f"copies of the memorized videos: {sorted(set(memorized['copies']))}")
    print(f"-> {path(args, 'finetune', 'finetune.csv')}, {path(args, 'finetune', 'memorized_prompts.csv')}")


# ------------------------------------------------------------------------------------------------------------------
# Part 2: the fine-tuning
# ------------------------------------------------------------------------------------------------------------------

def tokenize(tokenizer, caption):
    return tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                     return_tensors="pt").input_ids[0]


class FineTuneDataset(torch.utils.data.Dataset):
    """The dataset of the fine-tuning. A sample is (video in [-1, 1] as [3, frames, size, size], token ids).
    Nothing here is random, so the number of decoding workers cannot change the result."""

    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer
        table = pd.read_csv(args.table, keep_default_na=False)
        # THE DUPLICATION: a row with copies = k is repeated k times, so an epoch shows that video k times.
        table = table.loc[table.index.repeat(table["copies"])]
        self.videoids = table["videoid"].tolist()
        self.captions = table["caption"].tolist()

    def __len__(self):
        return len(self.videoids)

    def __getitem__(self, i):
        # Center crop and resize; the [0, 1] pixels become [-1, 1], which is what the VAE takes.
        clip = read_clip(video_path(self.args, self.videoids[i]))
        video = square(clip, self.args.size, crop=True).permute(1, 0, 2, 3) * 2 - 1  # [3, frames, size, size]
        return video, tokenize(self.tokenizer, self.captions[i])


class AnimateDiffForTraining(torch.nn.Module):
    """The three networks of AnimateDiff as one module, so that they move to the GPU together. Only the UNet trains.
    forward() returns the diffusion loss of a batch."""

    def __init__(self, pipe):
        super().__init__()
        self.unet, self.text_encoder, self.vae = pipe.unet, pipe.text_encoder, pipe.vae
        # The noise schedule AnimateDiff was trained with (and that sampler.py generates with).
        self.noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", beta_schedule="linear",
                                                             clip_sample=False)

    def forward(self, video, tokens):
        batch, _, frames = video.shape[:3]

        # 1. Video -> latents, frame by frame, and caption -> text embeddings, one copy per frame as the UNet expects.
        #    The VAE and CLIP are frozen, so no gradient is needed here.
        with torch.no_grad():
            pixels = video.transpose(1, 2).flatten(0, 1)  # [batch * frames, 3, size, size]
            latents = self.vae.encode(pixels).latent_dist.sample() * self.vae.config.scaling_factor
            latents = latents.unflatten(0, (batch, frames)).transpose(1, 2)  # [batch, 4, frames, h, w]
            text = self.text_encoder(tokens)[0].repeat_interleave(frames, dim=0)

        # 2. Add noise, at a random timestep per video.
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch,), device=latents.device)
        noisy = self.noise_scheduler.add_noise(latents, noise, t)

        # 3. The diffusion loss: the UNet (image layers and motion modules) predicts the noise.
        #    Every UNet block is recomputed in the backward pass instead of keeping its activations in memory
        #    (diffusers' enable_gradient_checkpointing leaves out the motion modules, and that needs far more memory).
        with checkpointed_blocks(self.unet):
            prediction = self.unet(noisy, t, encoder_hidden_states=text).sample
        return F.mse_loss(prediction.float(), noise.float())


def latest_checkpoint(out):
    """The epoch of the last complete checkpoint in the folder, or None. training_state.pt is the last file a checkpoint
    writes, so a checkpoint that was cut off while saving does not count."""
    files = glob.glob(os.path.join(out, "epoch_*", "training_state.pt"))
    return max((int(os.path.basename(os.path.dirname(f)).split("_")[1]) for f in files), default=None)


def run_train(args):
    """Fine-tune AnimateDiff's whole UNet (image layers and motion modules). The text encoder and the VAE stay frozen."""
    # One process per GPU; float16 where it is safe, for memory. gradient_as_bucket_view: on several GPUs, without it
    # PyTorch keeps a second copy of all the gradients for synchronizing them (6 GB).
    # data_seed: the order of the rows in each epoch depends only on --seed and the epoch number.
    accelerator = Accelerator(mixed_precision="fp16",
                              dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=args.seed),
                              kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True)])
    # REPRODUCIBILITY: the same --seed gives the same weights. device_specific: every GPU draws different noise (from
    # seed + its number). deterministic: the GPU only uses algorithms that give the same result every time.
    set_seed(args.seed, device_specific=True, deterministic=True)
    torch.backends.cudnn.deterministic = True

    # Continue from the latest checkpoint of --out, or start a new run in a folder without checkpoints.
    # A new run in a used folder is refused: it would overwrite them, and --resume could later pick up the old ones.
    last_epoch = latest_checkpoint(args.out)
    if args.resume:
        if args.model:
            raise SystemExit("--resume continues from the checkpoints of --out: do not give --model too")
        if last_epoch is None:
            raise SystemExit(f"--resume: no complete checkpoint in {args.out}")
        if last_epoch >= args.epochs:
            raise SystemExit(f"{args.out} has already reached epoch {last_epoch}: to train further, raise --epochs")
        # A different setting would not continue the same training (and other GPUs have no saved random states).
        with open(os.path.join(args.out, "settings.json")) as file:
            saved = json.load(file)
        current = {**vars(args), "gpus": accelerator.num_processes}
        for key in ["batch_size", "lr", "size", "null_prompt", "seed", "gpus"]:
            if saved[key] != current[key]:
                raise SystemExit(f"--resume: {key} is {current[key]}, but this run used {saved[key]}")
        args.table = os.path.join(args.out, "finetune.csv")  # the table of this run, even if `build` ran again since
        accelerator.print(f"continuing from {args.out}/epoch_{last_epoch}")
    elif last_epoch is not None:
        raise SystemExit(f"{args.out} already has checkpoints: use --resume to continue them, or another --out")

    # float32 weights: mixed precision needs them to update the weights accurately.
    start = os.path.join(args.out, f"epoch_{last_epoch}") if args.resume else args.model
    pipe = load_pipeline(device="cpu", dtype=torch.float32, model=start)
    pipe.register_modules(motion_adapter=None)  # already merged into the UNet: only a stale copy would be saved
    # WHAT TRAINS: the whole UNet (normal AnimateDiff training only trains its motion modules). CLIP and the VAE stay
    # frozen, as they did when Stable Diffusion and AnimateDiff were trained.
    pipe.unet.requires_grad_(True).train()
    for network in [pipe.text_encoder, pipe.vae]:
        network.requires_grad_(False).eval()

    model = AnimateDiffForTraining(pipe)
    # AdamW with 8-bit statistics. Normal AdamW keeps two extra copies of the 1.3 billion UNet parameters (10 GB): with
    # the weights and the gradients, that alone would nearly fill a 24 GB GPU.
    optimizer = bnb.optim.AdamW8bit(pipe.unet.parameters(), lr=args.lr)
    dataset = FineTuneDataset(args, pipe.tokenizer)
    # shuffle: an epoch goes through all the rows (so all the copies) in random order.
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.workers, persistent_workers=args.workers > 0)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.print(f"{len(dataset)} rows, {len(loader)} steps per epoch on {accelerator.num_processes} GPU(s), "
                      f"{sum(p.numel() for p in optimizer.param_groups[0]['params']) / 1e6:.0f} M trained parameters")
    empty_prompt = tokenize(pipe.tokenizer, "").to(accelerator.device)

    # Keep what is needed to repeat this run: all its settings, the number of GPUs (it changes the result too) and the
    # table (running `build` again later can give another one, as more links die).
    if accelerator.is_main_process:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "settings.json"), "w") as file:
            json.dump({**vars(args), "gpus": accelerator.num_processes}, file, indent=2)
        if not args.resume:
            shutil.copy(args.table, os.path.join(args.out, "finetune.csv"))

    def save(name):
        """A complete pipeline: load it again with `main.py --model <folder>` or `finetune.py train --model <folder>`."""
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            pipe.save_pretrained(os.path.join(args.out, name))
            print(f"saved {os.path.join(args.out, name)}", flush=True)
        accelerator.wait_for_everyone()

    def save_checkpoint(epoch):
        """The model, plus what --resume needs to carry on exactly where this epoch ended."""
        save(f"epoch_{epoch}")
        folder = os.path.join(args.out, f"epoch_{epoch}")
        # Every GPU has its own random numbers.
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state()}, os.path.join(folder, f"rng_{accelerator.process_index}.pt"))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:  # the optimizer is the same on all the GPUs
            state = {"optimizer": optimizer.state_dict(), "scaler": accelerator.scaler.state_dict(), "step": step}
            torch.save(state, os.path.join(folder, "training_state.tmp"))
            # Only now the checkpoint counts as complete (see latest_checkpoint).
            os.replace(os.path.join(folder, "training_state.tmp"), os.path.join(folder, "training_state.pt"))
        accelerator.wait_for_everyone()

    step, first_epoch = 0, 1
    if args.resume:
        folder = os.path.join(args.out, f"epoch_{last_epoch}")
        state = torch.load(os.path.join(folder, "training_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])  # Adam's statistics
        accelerator.scaler.load_state_dict(state["scaler"])  # the scale of the mixed precision
        step, first_epoch = state["step"], last_epoch + 1
        del state
        loader.set_epoch(last_epoch)  # the order of the rows depends on the epoch number
        rng = torch.load(os.path.join(folder, f"rng_{accelerator.process_index}.pt"), weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])

    for epoch in range(first_epoch, args.epochs + 1):
        bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", disable=not accelerator.is_main_process)
        for video, tokens in bar:
            # Sometimes train on the empty prompt, as AnimateDiff did: classifier-free guidance needs it.
            tokens[torch.rand(len(tokens), device=tokens.device) < args.null_prompt] = empty_prompt
            loss = model(video, tokens)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)  # as in AnimateDiff's training
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")
            if step == args.max_steps:
                break
        if step == args.max_steps:  # (a short test run)
            break
        if epoch % args.save_every == 0:
            save_checkpoint(epoch)  # memorization can be measured after each of them, and --resume continues from the last
    save("final")
    accelerator.print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Artificial memorization: fine-tune AnimateDiff on a dataset with duplicated videos")
    parser.add_argument("--root", default="data/webvid", help="the WebVid folder of webvid.py (metadata and videos)")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("select", help="draw random candidates from all of WebVid-10M, for `webvid.py download`")
    s.add_argument("--candidates", type=int, default=25000,
                   help="videos to draw: more than --background + --memorized of `build`, as some links are dead")
    s.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("build", help="keep the candidates that can be read and assign the number of copies")
    b.add_argument("--background", type=int, default=20000, help="samples with one copy, to keep the distribution")
    b.add_argument("--memorized", type=int, default=200, help="samples to memorize")
    b.add_argument("--per-level", type=int, default=4, help="samples for each number of copies")
    b.add_argument("--workers", type=int, default=16, help="processes checking the videos")

    t = sub.add_parser("train", help="fine-tune the UNet (run it with `accelerate launch` for several GPUs)")
    t.add_argument("--table", default="data/webvid/finetune/finetune.csv", help="the table written by `build`")
    t.add_argument("--out", default="outputs/finetune", help="folder for the fine-tuned models")
    t.add_argument("--model", default=None, help="start a new training from a saved model (default: the original AnimateDiff)")
    t.add_argument("--resume", action="store_true", help="continue the run in --out from its latest checkpoint")
    t.add_argument("--epochs", type=int, default=10, help="total epochs of the run, also with --resume")
    t.add_argument("--save-every", type=int, default=1, help="save a checkpoint every this many epochs")
    t.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (for a short test run)")
    t.add_argument("--batch-size", type=int, default=2, help="videos per GPU (3 does not fit in 24 GB)")
    t.add_argument("--lr", type=float, default=1e-5, help="learning rate of the UNet")
    t.add_argument("--size", type=int, default=256, help="videos are center-cropped and resized to this (AnimateDiff: 256)")
    t.add_argument("--null-prompt", type=float, default=0.1, help="share of the samples trained on the empty prompt")
    t.add_argument("--workers", type=int, default=8, help="processes decoding the videos, per GPU")
    t.add_argument("--seed", type=int, default=0, help="the same seed, GPUs and settings give the same weights")

    args = parser.parse_args()
    {"select": run_select, "build": run_build, "train": run_train}[args.command](args)


if __name__ == "__main__":
    main()
