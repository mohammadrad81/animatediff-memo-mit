"""WebVid-10M (the training set of AnimateDiff's motion modules): metadata, duplication ranking and the prompt sheet
for `main.py batch`. See the WebVid section of README.md, or run:  python webvid.py {metadata,captions,download,i3d,duplicates,prompts,gsscd} --help

Pipeline (every step can be stopped and run again; it continues where it stopped):
  metadata    download the train metadata (TempoFunk/webvid-10M) and merge it into train.parquet
  captions    group the videos by caption; repeated captions are the candidates for duplicated videos
  download    download the videos, the largest caption groups first, up to a disk budget
  i3d         I3D (Kinetics-400) features of the downloaded videos
  duplicates  for every downloaded video, how many other downloaded videos have I3D cosine similarity >= a threshold
  prompts     one row per unique caption, the most duplicated videos first: the CSV sheet for `main.py batch`
  gsscd       GSSCD (content memorization) of generated videos against their training videos: CSV + histogram
"""
import argparse
import glob
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

REPO_ID = "TempoFunk/webvid-10M"


def path(args, *parts):
    return os.path.join(args.root, *parts)


def video_path(args, videoid):
    """Videos are spread over 1000 folders (last 3 digits of the id): millions of files in one folder are slow."""
    return path(args, "videos", f"{int(videoid) % 1000:03d}", f"{videoid}.mp4")


def seconds(duration):
    """'PT00H00M11S' -> 11."""
    h, m, s = (int(x) for x in re.fullmatch(r"PT(\d+)H(\d+)M(\d+)S", duration).groups())
    return 3600 * h + 60 * m + s


def normalize(caption):
    """Captions that differ only in case or white space are the same caption."""
    return " ".join(caption.lower().split())


def run_metadata(args):
    """Download the train partitions (7152 CSV files, about 2.9 GB) and merge them into train.parquet:
    videoid, contentUrl, duration (seconds), page_dir, name (the caption)."""
    from huggingface_hub import HfApi, hf_hub_download
    hf = path(args, "metadata", "hf")
    names = [f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset") if f.startswith("data/train/")]
    missing = [f for f in names if not os.path.exists(os.path.join(hf, f))]  # only these: Hugging Face rate-limits requests
    print(f"{len(names)} train partitions, {len(missing)} to download")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda f: hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=hf), missing))
    files = sorted(os.path.join(hf, f) for f in names)
    print(f"merging {len(files)} partitions")
    read = lambda f: pd.read_csv(f, dtype={"videoid": "int64", "name": "string"}, keep_default_na=False)
    with ThreadPoolExecutor(16) as pool:  # thousands of small files on a network drive: read in parallel
        df = pd.concat(list(pool.map(read, files)), ignore_index=True)
    n = len(df)
    df = df.drop_duplicates("videoid").reset_index(drop=True)
    df["duration"] = df["duration"].map(seconds).astype("int32")
    df.to_parquet(path(args, "train.parquet"))
    print(f"{len(df)} videos ({n - len(df)} repeated rows dropped) -> {path(args, 'train.parquet')}")


def load_train(args, columns=None):
    return pd.read_parquet(path(args, "train.parquet"), columns=columns)


def caption_groups(args):
    """[videoid, name, key, group size] with key = the normalized caption, and the representative videoid of
    each caption (the smallest id: it does not change when the sheet is rebuilt, so `batch` never generates a
    caption twice under two names)."""
    df = load_train(args, ["videoid", "name"])
    df["key"] = df["name"].map(normalize)
    df = df[df["key"] != ""]
    g = df.groupby("key")["videoid"]
    df["group"] = g.transform("size").astype("int32")
    df["rep"] = g.transform("min")
    return df


def run_captions(args):
    """Group the videos by caption. Writes captions.parquet (one row per unique caption: the representative
    videoid and caption, and how many videos have it) and prints how much the captions repeat."""
    df = caption_groups(args)
    caps = (df[df["videoid"] == df["rep"]][["videoid", "name", "key", "group"]]
            .sort_values(["group", "videoid"], ascending=[False, True]).reset_index(drop=True))
    caps.to_parquet(path(args, "captions.parquet"))
    print(f"{len(df)} videos, {len(caps)} unique captions -> {path(args, 'captions.parquet')}")
    for k in [2, 5, 10, 50, 100]:
        big = caps[caps["group"] >= k]
        print(f"  captions shared by >= {k:3d} videos: {len(big):8d} captions, {big['group'].sum():8d} videos")
    print("most repeated captions:")
    for _, r in caps.head(15).iterrows():
        print(f"  {r['group']:6d}  {r['name'][:100]}")


def generated_ids(folder):
    """Videoids of the generated GIFs <videoid>.gif in a folder."""
    return {int(f[:-4]) for f in os.listdir(folder) if f.endswith(".gif") and not f.endswith(".tmp.gif") and f[:-4].isdigit()}


def download_order(args, failed=()):
    """Videos to download, the largest caption groups first (then group by group, by videoid). At most
    --per-caption videos of each caption: generic captions ("abstract background", "fireworks") have thousands
    of videos, and a sample of them is enough for I3D to tell whether they are duplicates. Dead links (failed)
    are left out before the first --per-caption are taken, so the next video of the caption is tried instead.
    With --for-videos, only the captions of the GIFs in that folder (the videos already generated).
    With --max-captions, only the first that many captions."""
    df = caption_groups(args)
    reps = set(df[df["videoid"].isin(generated_ids(args.for_videos))]["rep"]) if args.for_videos else None
    df = df[~df["videoid"].isin(failed)]  # (a generated video's own link may be dead: its caption still counts)
    if args.for_videos:
        df = df[df["rep"].isin(reps)]
        print(f"{len(reps)} captions of the generated videos in {args.for_videos}")
    else:
        df = df[df["group"] >= args.min_group]
    if args.per_caption:
        df = df.sort_values("videoid")
        df = df[df.groupby("rep").cumcount() < args.per_caption]
    df = df.sort_values(["group", "rep", "videoid"], ascending=[False, True, True])
    if args.max_captions:  # the first N captions of the order (captions whose known links are all dead do not count)
        df = df[df["rep"].isin(df["rep"].drop_duplicates().head(args.max_captions))]
    df = df.merge(load_train(args, ["videoid", "contentUrl"]), on="videoid", how="left")  # (keeps the order)
    return df[["videoid", "contentUrl"]]


def downloaded(args):
    """{videoid: size in bytes} of the videos already downloaded."""
    done = {}
    for d in glob.glob(path(args, "videos", "[0-9][0-9][0-9]")):
        with os.scandir(d) as it:
            for e in it:
                if e.name.endswith(".mp4"):
                    done[int(e.name[:-4])] = e.stat().st_size
    return done


def run_download(args):
    """Download the videos in download_order until --max-gb (counting what is already there) or --max-videos.
    A video is written under another name first, so a .mp4 that exists is complete. Videos that are gone are
    listed in videos/failed.txt and not tried again (--retry-failed tries them again).
    The server answers 403 both for a video that is gone and when it throttles a client that downloads too fast,
    and gone videos come in runs (a contributor's whole series is deleted). So when more than 30% of the last 300
    answers are failures, a video that is known to work (an already downloaded one) is requested: if it fails
    too, the server is throttling, the downloads pause for --pause seconds and the failed videos are tried again;
    if it works, the failures are real and recorded."""
    import collections
    import requests
    os.makedirs(path(args, "videos"), exist_ok=True)
    failed_path = path(args, "videos", "failed.txt")
    if args.retry_failed and os.path.exists(failed_path):
        old = path(args, "videos", f"failed_before_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        os.replace(failed_path, old)
        print(f"trying every link again (the old list is in {old})")
    done = downloaded(args)
    failed = set()
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed = {int(line.split()[0]) for line in f if line.strip()} - set(done)
    order = download_order(args, failed)
    print(f"{len(order)} videos to have; {len(done)} downloaded "
          f"({sum(done.values()) / 1e9:.1f} GB), {len(failed)} known dead links")
    if args.max_videos is not None:
        order = order.head(args.max_videos)
    todo = [(v, u) for v, u in zip(order["videoid"], order["contentUrl"]) if v not in done]
    budget = args.max_gb * 1e9
    total = [sum(done.values()), 0, 0, time.time()]  # bytes, new videos, dead links, start time
    recent = collections.deque(maxlen=300)          # True = failed, for the last answers
    pause_until = [0.0]
    urls = load_train(args, ["videoid", "contentUrl"]).set_index("videoid")["contentUrl"]
    good = [urls[v] for v in list(done)[:20]]       # links known to work, to tell throttling from dead links
    lock, local = threading.Lock(), threading.local()

    def get(session, url):
        for attempt in range(3):
            try:
                return session.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
        return None  # network error: tried again in the next run

    def fetch(item):
        videoid, url = item
        session = getattr(local, "session", None) or requests.Session()
        local.session = session
        while total[0] < budget:
            time.sleep(max(0.0, pause_until[0] - time.time()))
            r = get(session, url)
            if r is None:
                return
            with lock:
                recent.append(r.status_code != 200)
                if r.status_code != 200 and pause_until[0] > time.time():
                    continue  # failed while paused: tried again after the pause
                if r.status_code != 200 and len(recent) >= 100 and sum(recent) > 0.3 * len(recent):
                    probe = get(session, good[total[1] % len(good)]) if good else None
                    recent.clear()
                    if probe is None or probe.status_code != 200:  # a working video fails too: throttled
                        print(f"  many requests failed and a known-good video fails too: throttled, "
                              f"pausing {args.pause:.0f} s", flush=True)
                        pause_until[0] = time.time() + args.pause
                        continue
                if r.status_code in (403, 404, 410):
                    with open(failed_path, "a") as f:
                        f.write(f"{videoid} {r.status_code}\n")
                    total[2] += 1
                    return
            if r.status_code != 200:
                return  # another error (e.g. 500): tried again in the next run
            out = video_path(args, videoid)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + ".tmp", "wb") as f:
                f.write(r.content)
            os.replace(out + ".tmp", out)
            with lock:
                if len(good) < 20:
                    good.append(url)
                total[0] += len(r.content)
                total[1] += 1
                if total[1] % 1000 == 0:
                    rate = total[1] / (time.time() - total[3])
                    print(f"  {total[1]} new videos ({rate:.0f}/s), {total[2]} dead links, "
                          f"{total[0] / 1e9:.1f} GB in total", flush=True)
            return

    print(f"downloading up to {len(todo)} videos with {args.workers} threads, budget {args.max_gb} GB")
    with ThreadPoolExecutor(args.workers) as pool:
        for i in range(0, len(todo), 10000):  # in chunks, so the budget stops the run without queueing millions
            if total[0] >= budget:
                break
            list(pool.map(fetch, todo[i:i + 10000]))
    print(f"done: {total[1]} new videos, {total[2]} dead links, {total[0] / 1e9:.1f} GB in {path(args, 'videos')}")


class Clips:
    """Dataset of downloaded videos for I3D: `frames` frames evenly spaced over the whole clip, resized to
    224 x 224, as uint8 [3, frames, 224, 224]. A video that cannot be decoded is returned as None."""

    def __init__(self, args, ids):
        self.args, self.ids = args, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import av
        import torch
        import torch.nn.functional as F
        videoid = self.ids[i]
        try:
            with av.open(video_path(self.args, videoid)) as c:
                stream = c.streams.video[0]
                stream.thread_type = "AUTO"
                frames = [f.to_ndarray(format="rgb24") for f in c.decode(stream)]
            take = np.linspace(0, len(frames) - 1, self.args.frames).round().astype(int)
            x = torch.from_numpy(np.stack([frames[t] for t in take])).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            return videoid, x.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3)
        except Exception:
            return videoid, None


def collate(batch):
    import torch
    ok = [(v, x) for v, x in batch if x is not None]
    bad = [v for v, x in batch if x is None]
    return [v for v, _ in ok], (torch.stack([x for _, x in ok]) if ok else None), bad


def feature_ids(args):
    ids = set()
    for f in glob.glob(path(args, "i3d", "feats_*.npz")):
        ids.update(np.load(f)["videoid"].tolist())
    return ids


def run_i3d(args):
    """I3D features (Kinetics-400 Inception-I3D, see i3d.py) of every downloaded video that has none yet, written
    in parts i3d/feats_<n>.npz: videoid, feats (1024, the last Inception block averaged over space and time) and
    logits (400 Kinetics classes), float16. Undecodable videos go to i3d/failed.txt."""
    import torch
    from tqdm import tqdm
    from i3d import load_i3d
    os.makedirs(path(args, "i3d"), exist_ok=True)
    failed_path = path(args, "i3d", "failed.txt")
    failed = set()
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed = {int(line) for line in f if line.strip()}
    have = feature_ids(args)
    ids = sorted(set(downloaded(args)) - have - failed)
    print(f"{len(have)} videos have features, {len(ids)} to do")
    if not ids:
        return
    i3d = load_i3d(args.i3d_weights if os.path.exists(args.i3d_weights) else None)
    loader = torch.utils.data.DataLoader(Clips(args, ids), batch_size=args.batch_size, num_workers=args.workers,
                                         collate_fn=collate, prefetch_factor=4)
    part = len(glob.glob(path(args, "i3d", "feats_*.npz")))
    buf_ids, buf, buf_logits = [], [], []

    def flush():
        nonlocal part, buf_ids, buf, buf_logits
        if buf_ids:
            out = path(args, "i3d", f"feats_{part:05d}.npz")
            np.savez(out + ".tmp.npz", videoid=np.array(buf_ids, dtype=np.int64), feats=np.concatenate(buf),
                     logits=np.concatenate(buf_logits))
            os.replace(out + ".tmp.npz", out)
            part, buf_ids, buf, buf_logits = part + 1, [], [], []

    with torch.no_grad():
        for vids, x, bad in tqdm(loader, desc="I3D"):
            if bad:
                with open(failed_path, "a") as f:
                    f.writelines(f"{v}\n" for v in bad)
            if x is None:
                continue
            feats, logits = i3d(x.cuda().float() / 127.5 - 1)  # I3D takes RGB in [-1, 1]
            buf_ids += vids
            buf.append(feats.half().cpu().numpy())
            buf_logits.append(logits.half().cpu().numpy())
            if len(buf_ids) >= 10000:
                flush()
    flush()


def load_features(args):
    files = sorted(glob.glob(path(args, "i3d", "feats_*.npz")))
    if not files:
        raise SystemExit(f"no I3D features in {path(args, 'i3d')}: run `python webvid.py i3d` first")
    parts = [np.load(f) for f in files]
    return np.concatenate([p["videoid"] for p in parts]), np.concatenate([p["feats"] for p in parts])


def run_duplicates(args):
    """Duplication of every video with features, as in the paper (cosine similarity of I3D features between
    training videos): n_dup = number of other videos with similarity >= --threshold, plus the --top most similar
    videos. Writes duplicates.parquet, the most duplicated videos first."""
    import torch
    ids, feats = load_features(args)
    f = torch.nn.functional.normalize(torch.from_numpy(feats).float().cuda(), dim=1).half()
    n_dup = np.zeros(len(ids), dtype=np.int32)
    top_sim = np.zeros((len(ids), args.top), dtype=np.float32)
    top_idx = np.zeros((len(ids), args.top), dtype=np.int64)
    for i in range(0, len(ids), 4096):
        s = f[i:i + 4096] @ f.T                                        # [chunk, videos]
        s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -1     # not the video itself
        n_dup[i:i + len(s)] = (s >= args.threshold).sum(1).cpu().numpy()
        v, k = s.float().topk(min(args.top, len(ids) - 1), dim=1)
        top_sim[i:i + len(s), :v.shape[1]], top_idx[i:i + len(s), :v.shape[1]] = v.cpu().numpy(), k.cpu().numpy()
    dup = pd.DataFrame({"videoid": ids, "n_dup": n_dup, "max_sim": top_sim[:, 0],
                        "neighbors": [ids[k].tolist() for k in top_idx], "neighbor_sims": top_sim.round(4).tolist()})
    dup = dup.merge(load_train(args, ["videoid", "name"]), on="videoid")
    dup = dup.sort_values(["n_dup", "max_sim"], ascending=False).reset_index(drop=True)
    dup.to_parquet(path(args, "duplicates.parquet"))
    print(f"{len(dup)} videos, threshold {args.threshold}: "
          + ", ".join(f"{(dup['n_dup'] >= k).sum()} with >= {k} duplicates" for k in [1, 5, 10, 50]))
    print(f"similarity to the nearest other video: median {np.median(top_sim[:, 0]):.3f}, "
          f"99th percentile {np.percentile(top_sim[:, 0], 99):.3f}")
    print("most duplicated:")
    for _, r in dup.head(15).iterrows():
        print(f"  {r['n_dup']:6d}  {r['videoid']:>11d}  {r['name'][:90]}")
    print(f"-> {path(args, 'duplicates.parquet')}")


def run_prompts(args):
    """The prompt sheet for `main.py batch`: one row per unique caption, with
      index         the GIF is saved as <index>.gif: the smallest videoid of the caption that is downloaded and has
                    I3D features, or the smallest videoid of the caption if none is (both never change)
      caption       the caption
      n_dup         the largest number of I3D duplicates of any video with this caption (0 if none has features)
      dup_videoid   that video
      caption_count how many training videos have this caption
      checked       how many of them have I3D features
    The most duplicated first (n_dup, then caption_count, then index): run its first rows to start with the
    videos most likely to be memorized. The paper's WebVid-duplication-prompt set is the first 500 rows.
    --checked-only keeps only the captions with at least one checked video: the only ones whose generated videos
    can be compared with training videos (GSSCD). Captions whose download links are all dead are left out."""
    df = caption_groups(args)
    if os.path.exists(path(args, "duplicates.parquet")):
        feats = glob.glob(path(args, "i3d", "feats_*.npz"))
        if feats and max(map(os.path.getmtime, feats)) > os.path.getmtime(path(args, "duplicates.parquet")):
            print("WARNING: there are I3D features newer than duplicates.parquet: run `python webvid.py duplicates` "
                  "first, or the videos downloaded since then are left out of the sheet")
        dup = pd.read_parquet(path(args, "duplicates.parquet"), columns=["videoid", "n_dup"])
        df = df.merge(dup, on="videoid", how="left")
    else:
        print("no duplicates.parquet: the sheet is ordered by caption_count only")
        df["n_dup"] = np.nan
    df["checked"] = df["n_dup"].notna()
    best = df.sort_values(["n_dup", "videoid"], ascending=[False, True], na_position="last").drop_duplicates("rep")
    best = best.set_index("rep")
    caps = df[df["videoid"] == df["rep"]].set_index("videoid")
    first_checked = df[df["checked"]].groupby("rep")["videoid"].min()  # not the representative if its link is dead
    sheet = pd.DataFrame({
        "index": first_checked.reindex(caps.index).fillna(pd.Series(caps.index, index=caps.index)).astype("int64").to_numpy(),
        "caption": caps["name"].to_numpy(),
        "n_dup": best.loc[caps.index, "n_dup"].fillna(0).astype("int32").to_numpy(),
        "dup_videoid": best.loc[caps.index, "videoid"].to_numpy(),
        "caption_count": caps["group"].to_numpy(),
        "checked": df.groupby("rep")["checked"].sum().loc[caps.index].astype("int32").to_numpy()})
    sheet["caption"] = sheet["caption"].str.strip()
    sheet = sheet.sort_values(["n_dup", "caption_count", "index"], ascending=[False, False, True])
    if args.checked_only:
        sheet = sheet[sheet["checked"] > 0]
    sheet.to_csv(args.out + ".tmp", index=False)  # renamed when complete: `batch` may read it at any time
    os.replace(args.out + ".tmp", args.out)
    print(f"{len(sheet)} unique captions -> {args.out}")
    print(sheet.head(10).to_string(index=False, max_colwidth=70))


def square(x, size, crop):
    """uint8 [frames, H, W, 3] -> float [frames, 3, size, size] in [0, 1]. crop: center square crop first (as
    AnimateDiff's training clips), otherwise the whole frame is squeezed to a square."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255
    if crop:
        h, w = x.shape[2:]
        s = min(h, w)
        x = x[:, :, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)


class TrainFrames:
    """Training videos for SSCD: frames sampled at `fps` (frame number and time of each), as [frames, 3, size, size]."""

    def __init__(self, args, ids):
        self.args, self.ids = args, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import av
        videoid = self.ids[i]
        try:
            frames, numbers, times, next_t = [], [], [], 0.0
            with av.open(video_path(self.args, videoid)) as c:
                stream = c.streams.video[0]
                stream.thread_type = "AUTO"
                for n, f in enumerate(c.decode(stream)):
                    t = f.time if f.time is not None else n / float(stream.average_rate or 25)
                    if t + 1e-6 >= next_t:
                        frames.append(f.to_ndarray(format="rgb24"))
                        numbers.append(n)
                        times.append(t)
                        next_t += 1 / self.args.gt_fps
            x = square(np.stack(frames), self.args.size, self.args.crop == "center")
            return videoid, x, np.array(numbers, dtype=np.int32), np.array(times, dtype=np.float32)
        except Exception:
            return videoid, None, None, None


def embed(model, x, batch=256):
    """SSCD descriptors of frames [n, 3, size, size] in [0, 1], in batches; -> float16 [n, 512] on the GPU."""
    import torch
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(x), batch):
            out.append(model(x[i:i + batch].cuda(non_blocking=True)).half())
    return torch.cat(out)


def train_embeddings(args, model, ids):
    """{videoid: (descriptors [frames, 512] float16, frame numbers, times)} of training videos. They are kept in
    data/webvid/sscd/<crop>_<size>px_<fps>fps/<id % 1000>/<videoid>.npz, so every video is embedded only once."""
    import torch
    cache = path(args, "sscd", f"{args.crop}_{args.size}px_{args.gt_fps:g}fps")
    file = lambda v: os.path.join(cache, f"{int(v) % 1000:03d}", f"{v}.npz")
    todo = [v for v in ids if not os.path.exists(file(v))]
    if todo:
        # spawn, not fork: the main process has used PyAV (ffmpeg threads), and a forked decoder can then deadlock
        loader = torch.utils.data.DataLoader(TrainFrames(args, todo), batch_size=None, num_workers=args.workers,
                                             prefetch_factor=2, multiprocessing_context="spawn")
        for videoid, x, numbers, times in loader:
            if x is None:  # undecodable: an empty entry, so it is not tried again
                numbers, times, e = np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros((0, 512), np.float16)
            else:
                e = embed(model, x).cpu().numpy()
            os.makedirs(os.path.dirname(file(videoid)), exist_ok=True)
            np.savez(file(videoid) + ".tmp.npz", emb=e, frame=numbers, time=times)
            os.replace(file(videoid) + ".tmp.npz", file(videoid))
    out = {}
    for v in ids:
        d = np.load(file(v))
        if len(d["emb"]):
            out[v] = (torch.from_numpy(d["emb"]).cuda(), d["frame"], d["time"])
    return out


def gif_frames(gif):
    from PIL import Image, ImageSequence
    with Image.open(gif) as im:
        return np.stack([np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(im)])


def save_pair(args, gen, gen_frame, gt_videoid, gt_frame, score, out):
    """The generated frame and the training frame of the GSSCD pair side by side (288 px high), the training frame
    cropped as it was compared (--crop)."""
    import av
    from PIL import Image, ImageDraw
    with av.open(video_path(args, gt_videoid)) as c:
        for n, f in enumerate(c.decode(video=0)):
            if n == gt_frame:
                gt = f.to_image()
                break
    a = Image.fromarray(gen[gen_frame]).resize((288, 288))
    if args.crop == "center":
        s = min(gt.size)
        gt = gt.crop(((gt.width - s) // 2, (gt.height - s) // 2, (gt.width - s) // 2 + s, (gt.height - s) // 2 + s))
    b = gt.resize((round(288 * gt.width / gt.height), 288))
    im = Image.new("RGB", (a.width + b.width + 8, 288 + 20), "white")
    im.paste(a, (0, 20))
    im.paste(b, (a.width + 8, 20))
    ImageDraw.Draw(im).text((4, 4), f"generated frame {gen_frame}  |  training video {gt_videoid}, frame {gt_frame}"
                                     f"  |  SSCD {score:.3f}", fill="black")
    im.save(out, quality=90)


def plot_gsscd(df, threshold, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = df["gsscd"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=np.linspace(min(0.0, s.min()), 1.0, 51), color="#4c72b0", edgecolor="white")
    ax.axvline(threshold, color="#c44e52", linestyle="--", label=f"threshold {threshold} (memorized above)")
    ax.axvline(np.median(s), color="black", linestyle=":", label=f"median {np.median(s):.3f}")
    ax.set_xlabel("GSSCD (max SSCD over all frame pairs)")
    ax.set_ylabel("generated videos")
    ax.set_title(f"{len(s)} videos: {(s >= threshold).sum()} ({100 * (s >= threshold).mean():.1f}%) memorized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_gsscd(args):
    """GSSCD (Chen et al., "Investigating Memorization in Video Diffusion Models", Eq. 2) of every generated GIF in
    --videos: the largest SSCD cosine similarity over all pairs of (generated frame, training frame). The GIF
    <videoid>.gif is compared with every downloaded training video that has the same caption as <videoid>, and the
    best one counts. One row per generated video, the highest GSSCD first; a run continues where the last one
    stopped. Videos without any downloaded training video are not listed (and are tried again in the next run)."""
    import torch
    from tqdm import tqdm
    from sscd import load_sscd
    videos = os.path.normpath(args.videos)
    out = args.out or os.path.join(os.path.dirname(videos), f"gsscd_{os.path.basename(videos)}.csv")
    gifs = {v: os.path.join(videos, f"{v}.gif") for v in generated_ids(videos)}
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    if len(done):
        done = done[done["n_gt"] > 0]  # rows without a training video (older versions wrote them) are dropped
    todo = sorted(set(gifs) - set(done.get("videoid", [])))
    print(f"{len(gifs)} generated videos in {videos}, {len(done)} scored before, {len(todo)} to do")

    groups = caption_groups(args)
    rep = groups.set_index("videoid")["rep"]
    caption = groups.set_index("videoid")["name"]
    have = groups[groups["videoid"].isin(set(downloaded(args)))]
    train_of = have.groupby("rep")["videoid"].apply(list)  # caption -> its downloaded training videos
    info = None
    if os.path.exists(path(args, "prompts.csv")):
        info = pd.read_csv(path(args, "prompts.csv"), usecols=["index", "n_dup", "caption_count"]).set_index("index")
    model = load_sscd(args.sscd_weights if os.path.exists(args.sscd_weights) else None)
    pairs = os.path.splitext(out)[0] + "_pairs"
    rows = [] if not len(done) else done.to_dict("records")
    skipped = 0

    for c in range(0, len(todo), 500):  # in chunks: the training descriptors of a chunk are kept on the GPU
        chunk = [v for v in todo[c:c + 500] if v in rep.index]
        train = train_embeddings(args, model, sorted({t for v in chunk for t in train_of.get(rep[v], [])}))
        for v in tqdm(chunk, desc=f"GSSCD {c + len(chunk)}/{len(todo)}"):
            frames = gif_frames(gifs[v])
            g = embed(model, square(frames, args.size, crop=False))           # [gen frames, 512]
            row = {"videoid": v, "caption": caption[v], "gsscd": np.nan, "gen_frame": -1, "gt_videoid": -1,
                   "gt_frame": -1, "gt_time": np.nan, "memorized": False, "frames_memorized": 0,
                   "mean_frame_sscd": np.nan, "gt_memorized": 0, "gsscd_index": np.nan,
                   "n_gt": 0, "n_gt_frames": 0}
            per_frame = None                                                   # best SSCD of every generated frame
            for t in train_of.get(rep[v], []):
                if t not in train:
                    continue
                e, numbers, times = train[t]
                s = (g.float() @ e.float().T)                                  # [gen frames, train frames]
                m = s.max().item()
                per_frame = s.max(1).values if per_frame is None else torch.maximum(per_frame, s.max(1).values)
                row["n_gt"] += 1
                row["n_gt_frames"] += len(numbers)
                row["gt_memorized"] += int(m >= args.threshold)
                if t == v:
                    row["gsscd_index"] = m
                if not m <= row["gsscd"]:                                      # also when gsscd is still NaN
                    i, j = divmod(s.argmax().item(), s.shape[1])
                    row.update(gsscd=m, gen_frame=i, gt_videoid=t, gt_frame=int(numbers[j]), gt_time=float(times[j]))
            if per_frame is not None:
                row.update(memorized=row["gsscd"] >= args.threshold, mean_frame_sscd=per_frame.mean().item(),
                           frames_memorized=int((per_frame >= args.threshold).sum()))
                if row["memorized"] and not args.no_pairs:
                    os.makedirs(pairs, exist_ok=True)
                    save_pair(args, frames, row["gen_frame"], row["gt_videoid"], row["gt_frame"], row["gsscd"],
                              os.path.join(pairs, f"{v}.jpg"))
            if per_frame is None:  # no downloaded training video with this caption (dead links): not listed
                skipped += 1
                continue
            if info is not None and v in info.index:
                row.update(n_dup=int(info.at[v, "n_dup"]), caption_count=int(info.at[v, "caption_count"]))
            rows.append(row)
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values("gsscd", ascending=False, na_position="last")
        df.to_csv(out + ".tmp", index=False, float_format="%.4f")  # renamed when complete
        os.replace(out + ".tmp", out)

    if not os.path.exists(out):
        return print("nothing to score")
    df = pd.read_csv(out)
    scored = df["gsscd"].notna()
    if scored.any():
        plot_gsscd(df, args.threshold, os.path.splitext(out)[0] + ".png")
        print(f"{scored.sum()} videos scored: {df['memorized'].sum()} memorized (GSSCD >= {args.threshold}), "
              f"median GSSCD {df.loc[scored, 'gsscd'].median():.3f}")
    if skipped:
        print(f"{skipped} videos skipped: no downloaded training video with their caption "
              f"(run `webvid.py download --for-videos {args.videos}`, then i3d, or their links are dead)")
    print(f"-> {out}, {os.path.splitext(out)[0]}.png" + (f", {pairs}/" if os.path.isdir(pairs) else ""))


def main():
    parser = argparse.ArgumentParser(description="WebVid-10M: metadata, I3D duplication ranking and the prompt sheet for main.py batch")
    parser.add_argument("--root", default="data/webvid", help="folder for the metadata, videos and features")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("metadata", help="download the train metadata and merge it into train.parquet")
    sub.add_parser("captions", help="group the videos by caption (captions.parquet) and print how much they repeat")

    d = sub.add_parser("download", help="download the videos, the largest caption groups first")
    d.add_argument("--max-gb", type=float, default=500, help="stop when the videos folder holds this many GB")
    d.add_argument("--max-videos", type=int, default=None, help="only the first this many videos of the order")
    d.add_argument("--min-group", type=int, default=1,
                   help="only videos whose caption is shared by at least this many videos (1 = all videos)")
    d.add_argument("--per-caption", type=int, default=20, help="at most this many videos per caption (0 = all)")
    d.add_argument("--max-captions", type=int, default=None,
                   help="only the first this many captions of the order (the sheet then has about this many rows)")
    d.add_argument("--for-videos", default=None,
                   help="folder of generated GIFs (<videoid>.gif): download only the training videos of their captions")
    d.add_argument("--workers", type=int, default=16, help="parallel downloads")
    d.add_argument("--retry-failed", action="store_true",
                   help="try the links in videos/failed.txt again (the list is moved aside and rebuilt)")
    d.add_argument("--pause", type=float, default=300, help="seconds to wait when the server seems to throttle")

    i = sub.add_parser("i3d", help="I3D features of the downloaded videos")
    i.add_argument("--i3d-weights", default="data/webvid/models/i3d_rgb_imagenet.pt",
                   help="Kinetics-400 I3D weights (rgb_imagenet.pt of piergiaj/pytorch-i3d); downloaded if missing")
    i.add_argument("--frames", type=int, default=16, help="frames per video, evenly spaced over the clip")
    i.add_argument("--batch-size", type=int, default=32)
    i.add_argument("--workers", type=int, default=16, help="processes decoding the videos")

    u = sub.add_parser("duplicates", help="count every video's I3D near-duplicates")
    u.add_argument("--threshold", type=float, default=0.95, help="cosine similarity of I3D features that counts as a duplicate")
    u.add_argument("--top", type=int, default=10, help="most similar videos kept per video")

    p = sub.add_parser("prompts", help="write the prompt sheet (one row per unique caption, most duplicated first)")
    p.add_argument("--out", default="data/webvid/prompts.csv", help="CSV file to write")
    p.add_argument("--checked-only", action="store_true",
                   help="only captions with at least one downloaded video with I3D features (the ones GSSCD can score)")

    s = sub.add_parser("gsscd", help="GSSCD of generated videos against their training videos, CSV + histogram")
    s.add_argument("--videos", default="outputs/webvid/normal", help="folder of generated GIFs named <videoid>.gif")
    s.add_argument("--out", default=None, help="CSV to write (default: gsscd_<folder name>.csv next to the folder)")
    s.add_argument("--threshold", type=float, default=0.4, help="GSSCD above which a video counts as memorized (paper: 0.4)")
    s.add_argument("--gt-fps", type=float, default=8, help="training frames compared per second of training video")
    s.add_argument("--crop", choices=["center", "none"], default="center",
                   help="center: compare the center square of the training frames (what AnimateDiff was trained on); "
                        "none: the whole frame, squeezed to a square")
    s.add_argument("--size", type=int, default=288, help="frames are resized to size x size (SSCD's evaluation size)")
    s.add_argument("--workers", type=int, default=16, help="processes decoding the training videos")
    s.add_argument("--no-pairs", action="store_true", help="do not save the image pairs of memorized videos")
    s.add_argument("--sscd-weights", default="data/webvid/models/sscd_disc_mixup.classy.pt",
                   help="SSCD weights (sscd_disc_mixup.classy.pt); downloaded if missing")

    args = parser.parse_args()
    {"metadata": run_metadata, "captions": run_captions, "download": run_download, "i3d": run_i3d,
     "duplicates": run_duplicates, "prompts": run_prompts, "gsscd": run_gsscd}[args.command](args)


if __name__ == "__main__":
    main()
