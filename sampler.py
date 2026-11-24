"""Load AnimateDiff and run the denoising loop: normal CFG, or with one of the mitigation experiments
(GUARD / soften with repulsion as 3 streams, CLIP-embedding changes, adjusted initial noise)."""
import contextlib
import functools
import math

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

MODEL_ID = "CompVis/stable-diffusion-v1-4"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-4"


def load_pipeline(device="cuda", dtype=torch.float16, model=None):
    """model: folder of a model saved by finetune.py; None = the original AnimateDiff."""
    if model:
        pipe = AnimateDiffPipeline.from_pretrained(model, torch_dtype=dtype)
    else:
        adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=dtype)
        pipe = AnimateDiffPipeline.from_pretrained(MODEL_ID, motion_adapter=adapter, torch_dtype=dtype)
    # DDIM, as in the GUARD code; AnimateDiff was trained with a "linear" beta schedule.
    pipe.scheduler = DDIMScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler", beta_schedule="linear", clip_sample=False,
        timestep_spacing="linspace", steps_offset=1)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.requires_grad_(False)  # inference only (the noise adjustment needs gradients for the latent alone)
    return pipe.to(device)


@torch.no_grad()
def embed_prompts(pipe, prompts, ctrl=None, null=False):
    """CLIP text embeddings [prompts, 77, 768]. With ctrl.clip_mode set, the CLIP-embedding mitigation of Kim & No
    ("Memorization in Stable Diffusion is unexpectedly driven by CLIP embeddings") is applied:
      [v_sot, v_pr_1..v_pr_n, v_eot, v_pad_1..v_pad_d]   normal: every <pad> is another <eot> token, so the
                                                          v_pad are near copies of v_eot and amplify it
      "pad":  the prompt is padded with the "!" token instead of <eot> (v_pad no longer copy v_eot),
              and v_eot is set to 0 (unless ctrl.mask_eot is False)
      "mask": normal padding, but the first mask_ratio (default 70%) of the v_pad, the ones right after <eot>,
              are set to 0
    null=True is the empty prompt used for CFG. As in the official code (which changes the tokenizer's pad token),
    "pad" also pads it with "!", but its v_eot is kept; "mask" leaves it unchanged."""
    tok, L = pipe.tokenizer, pipe.tokenizer.model_max_length
    mode = ctrl.clip_mode if ctrl is not None else None
    if mode == "pad":
        rows = [tok(p, truncation=True, max_length=L).input_ids for p in prompts]  # <sot> ... <eot>, no padding
        bang = tok.convert_tokens_to_ids("!")
        ids = torch.tensor([r + [bang] * (L - len(r)) for r in rows])
    else:
        ids = tok(prompts, padding="max_length", max_length=L, truncation=True, return_tensors="pt").input_ids
    emb = pipe.text_encoder(ids.to(pipe.device))[0].clone()
    for i, row in enumerate(ids.tolist()):
        eot = row.index(tok.eos_token_id)  # first <eot> = end of the prompt
        if null:
            continue
        if mode == "pad" and ctrl.mask_eot:
            emb[i, eot] = 0
        elif mode == "mask":
            d = L - 1 - eot                          # number of padding embeddings
            emb[i, eot + 1:eot + 1 + int(ctrl.mask_ratio * d)] = 0  # int(): rounded down, as in the official code
    return emb


def orthogonalize(v_plus, v_minus, cos_threshold=0.9, coef_clip=2.0, eps=1e-8):
    """Official GUARD code (dual_guidance_call, orthogonalize_neighbors): where the attraction direction
    v_plus = eps_plus - eps_null points almost the same way as the repulsion direction v_minus = eps_minus - eps_null
    (cosine > 0.9), its component along v_minus is removed: v_plus - clip(<v_plus, v_minus> / |v_minus|^2) * v_minus.
    Done per image in the official code, so here per frame."""
    a, b = v_plus.float().transpose(1, 2), v_minus.float().transpose(1, 2)  # [videos, frames, 4, h, w]
    dot = (a * b).flatten(2).sum(-1)
    b2 = (b * b).flatten(2).sum(-1) + eps
    cos = dot / (((a * a).flatten(2).sum(-1) + eps).sqrt() * b2.sqrt())
    coef = (dot / b2).clamp(-coef_clip, coef_clip) * (cos > cos_threshold)
    return (a - coef[..., None, None, None] * b).transpose(1, 2).to(v_plus.dtype)


def frame_norm(d):
    """[videos, 4, frames, h, w] -> [videos, frames]: L2 norm of every frame (each frame is like one SD image)."""
    return d.float().transpose(1, 2).flatten(2).norm(dim=-1)


def guidance_eps(pipe, x, t, e_null, e_prompt):
    """eps_cond - eps_null (the paper's eps~) at latents x [videos, 4, frames, h, w], in float32."""
    text = torch.cat([e_null, e_prompt]).repeat_interleave(x.shape[2], dim=0)
    x = pipe.scheduler.scale_model_input(x.to(pipe.unet.dtype), t)
    eps_null, eps_cond = pipe.unet(torch.cat([x, x]), t, encoder_hidden_states=text).sample.chunk(2)
    return (eps_cond - eps_null).float()


def per_frame(n):
    """[videos, frames] -> broadcastable against [videos, 4, frames, h, w]."""
    return n[:, None, :, None, None]


def masked_step(optim, lat, grad, update):
    """One optimizer step on lat [1, 4, frames, h, w] that changes only the frames where update [1, frames] is True.
    The gradient of the other frames is set to 0, and their values are put back after the step, since Adam's
    momentum and AdamW's weight decay would otherwise still move them. update None: every frame changes."""
    optim.zero_grad()
    if update is None:
        lat.grad = grad
        optim.step()
        return
    keep = lat.detach().clone()
    lat.grad = grad * per_frame(update.to(grad.dtype))
    optim.step()
    with torch.no_grad():
        lat.copy_(torch.where(per_frame(update), lat, keep))


def guidance_vjp(pipe, x, t, e_null, e_prompt, v):
    """J^T v, with J the Jacobian of eps~(x) = eps_cond - eps_null w.r.t. the latent x (one video, float32).
    The two streams are backpropagated one after the other and the UNet blocks are recomputed instead of
    stored (checkpointed_blocks): a single 2-stream backward pass of 16 frames does not fit in 16 GB."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad(), checkpointed_blocks(pipe.unet):
        for e, sign in [(e_prompt, 1.0), (e_null, -1.0)]:
            xin = pipe.scheduler.scale_model_input(x.to(pipe.unet.dtype), t)
            out = pipe.unet(xin, t, encoder_hidden_states=e.repeat_interleave(x.shape[2], dim=0)).sample
            out.float().backward(sign * v)
    return x.grad


@contextlib.contextmanager
def checkpointed_blocks(unet):
    """Recompute every UNet block in the backward pass instead of storing its activations. Without this, the
    gradient through the 16-frame UNet does not fit in 16 GB (with it: about 12 GB at 512x512)."""
    blocks = [*unet.down_blocks, unet.mid_block, *unet.up_blocks]
    for b in blocks:
        b.forward = functools.partial(checkpoint, b.forward, use_reentrant=False)
    try:
        yield
    finally:
        for b in blocks:
            del b.forward  # back to the class's own forward


def adjust_noise(pipe, x, t, e_null, e_prompt, ctrl):
    """Initial-noise adjustment of Han et al., "Adjusting Initial Noise to Mitigate Memorization in Text-to-Image
    Diffusion Models" (NeurIPS 2025). x: [videos, 4, frames, h, w] initial noise x_T, t: first timestep T.
    The idea: x_T with a small ||eps~(x_T, T, y)|| = ||eps_cond - eps_null|| escapes the "attraction basin"
    (where CFG pulls towards the memorized image) earlier. AnimateDiff denoises every frame like an SD image,
    so norms and the step delta are taken per frame.

    "batch" (Algorithm 1), M times, with no gradients:
        delta = rho * eps~(x) / ||eps~(x)||                (per frame)
        x     = x - gamma * (eps~(x + delta) - eps~(x))    (sharpness-aware step towards a smaller ||eps~||)
    "sample" (Appendix A): AdamW on x minimising sum over frames of ||eps~(x)||, for each video on its own,
        until every frame has ||eps~|| <= target (or max_iters steps).
        ctrl.update_frames "all": every step changes all frames; "above": a step changes only the frames whose
        ||eps~|| is still above target (the loss is the same sum over all frames, see masked_step).
    ctrl.noise_log gets one dict per video; in "sample" mode its "trace" has every frame's ||eps~|| at every step."""
    x = x.float()
    before = frame_norm(guidance_eps(pipe, x, t, e_null, e_prompt))
    steps, traces = [0] * x.shape[0], [None] * x.shape[0]
    above = ctrl.update_frames == "above"
    if ctrl.noise_mode == "batch":
        if above:
            raise ValueError("update_frames='above' needs the per-sample mode: the batch-wise mode has no target")
        for _ in range(ctrl.adj_iters):
            d = guidance_eps(pipe, x, t, e_null, e_prompt)
            delta = ctrl.rho * d / frame_norm(d)[:, None, :, None, None]
            x = x - ctrl.gamma * (guidance_eps(pipe, x + delta, t, e_null, e_prompt) - d)
    else:
        videos = []
        for v in range(x.shape[0]):
            lat = x[v:v + 1].clone().requires_grad_(True)
            optim = torch.optim.AdamW([lat], lr=ctrl.lr)  # default weight decay 0.01 keeps x near N(0, I)
            en, ep = e_null[v:v + 1], e_prompt[v:v + 1]
            trace = traces[v] = {"seed": None, "loss": [], "updated": []}
            pbar = tqdm(range(ctrl.max_iters + 1), desc=f"adjusting noise ({ctrl.update_frames} frames)")
            for steps[v] in pbar:
                d = guidance_eps(pipe, lat.detach(), t, en, ep)
                n = frame_norm(d)
                trace["loss"].append(n[0].tolist())
                update = n > ctrl.target if above else None
                pbar.set_postfix(max_norm=f"{n.max():.3f}", target=ctrl.target,
                                 updating=int((n > ctrl.target).sum()) if above else n.shape[1])
                if n.max() <= ctrl.target or steps[v] == ctrl.max_iters:
                    break
                trace["updated"].append(update[0].tolist() if above else [True] * n.shape[1])
                # gradient of sum_f ||eps~_f|| is J^T (eps~ / ||eps~||), per frame
                masked_step(optim, lat, guidance_vjp(pipe, lat, t, en, ep, d / per_frame(n)), update)
            videos.append(lat.detach())
        x = torch.cat(videos)
    after = frame_norm(guidance_eps(pipe, x, t, e_null, e_prompt))
    for v in range(x.shape[0]):
        log = {"method": "noise", "mode": ctrl.noise_mode, "before": before[v].tolist(), "after": after[v].tolist()}
        if ctrl.noise_mode == "sample":
            log.update(update_frames=ctrl.update_frames, steps=steps[v], threshold=ctrl.target,
                       reached=after[v].max().item() <= ctrl.target, loss_name="||eps_cond - eps_null||",
                       frame_updates=sum(map(sum, traces[v]["updated"])), trace=[traces[v]])
        ctrl.noise_log.append(log)
        print(f"initial noise adjusted ({ctrl.noise_mode}, {ctrl.update_frames} frames): max over frames of ||eps_cond - eps_null|| "
              f"{before[v].max():.3f} -> {after[v].max():.3f}" + (f" in {steps[v]} steps" if ctrl.noise_mode == "sample" else ""))
    return x.to(pipe.unet.dtype)


def sail(pipe, x, t, e_null, e_prompt, ctrl, seed):
    """SAIL (Jeon, Kim & No, ICML 2025): choose an initial noise x_T where the conditional density is not sharp.
    Sharpness ||H~ s~|| is approximated by a finite difference along the guidance direction; as in the official
    code (local_model/pipe.py), per frame here (each frame is like one SD image):
        loss_f = || eps~(x + d_f) ||_f + alpha * ||x_f||        d_f = eps~(x)_f / ||eps~(x)_f||   (unit step)
    x is optimized with Adam (lr 0.05) for at most sail_steps steps, until every frame has loss_f <= sail_thres.
    If that fails, a new noise (seed + 1, + 2, ...) is tried, at most sail_rounds times (the official code also
    restarts with new seeds); the noise with the lowest loss is kept. Round 0 starts from the normal noise.
    sail_thres None: alpha * sqrt(latent size of a frame) + 0.2, i.e. the Gaussian term of N(0, I) noise plus a
    sharpness term of 0.2 (6.6 at 512x512); a fixed number would depend on the resolution.
    ctrl.update_frames "all": every Adam step changes all frames; "above": a step changes only the frames whose loss
    is still above the threshold (the loss is the same sum over all frames, see masked_step).
    ctrl.sail_log gets one dict per video; its "trace" has every frame's loss at every step of every round."""
    thres = ctrl.sail_thres
    if thres is None:
        thres = ctrl.sail_alpha * (x[:, :, 0].numel() / x.shape[0]) ** 0.5 + 0.65
    videos = []
    for v in range(x.shape[0]):
        en, ep = e_null[v:v + 1], e_prompt[v:v + 1]
        eps = lambda z: guidance_eps(pipe, z, t, en, ep)
        best = None  # (max loss over frames, latent, round, steps, per-frame loss, sharpness, |x|)
        first = None  # sharpness term per frame of the normal noise, before any step
        trace = []    # per round: seed, and per step the loss and sharpness of every frame, frames updated
        for r in range(ctrl.sail_rounds):
            start = x[v:v + 1] if r == 0 else torch.randn(x[v:v + 1].shape, device=x.device,
                                                          generator=torch.Generator(x.device).manual_seed(seed + r))
            lat = start.float().clone().requires_grad_(True)
            optim = torch.optim.Adam([lat], lr=ctrl.sail_lr)
            trace.append({"seed": seed + r, "loss": [], "sharpness": [], "updated": []})
            for step in range(ctrl.sail_steps + 1):
                z = lat.detach()
                d = eps(z)
                n = frame_norm(d)
                y = z + d / per_frame(n)                       # one unit step along the guidance direction
                dy = eps(y)
                ny = frame_norm(dy)
                loss = ny + ctrl.sail_alpha * frame_norm(z)    # [1, frames]
                first = ny[0].tolist() if first is None else first
                trace[-1]["loss"].append(loss[0].tolist())
                trace[-1]["sharpness"].append(ny[0].tolist())
                if best is None or loss.max() < best[0]:
                    best = (loss.max().item(), z.clone(), r, step, loss[0].tolist(), ny[0].tolist(), frame_norm(z)[0].tolist())
                if loss.max() <= thres or step == ctrl.sail_steps:
                    break
                # Exact gradient of sum_f loss_f. Through y: u = J(y)^T dy/|dy|. Through the direction
                # d/|d|: J(z)^T w, with w = (u - d^ <d^, u>) / |d| per frame. Gaussian term: alpha * z / |z|.
                u = guidance_vjp(pipe, y, t, en, ep, dy / per_frame(ny))
                dh = d / per_frame(n)
                w = (u - dh * per_frame((dh * u).transpose(1, 2).flatten(2).sum(-1))) / per_frame(n)
                update = loss > thres if ctrl.update_frames == "above" else None
                trace[-1]["updated"].append(update[0].tolist() if update is not None else [True] * loss.shape[1])
                masked_step(optim, lat, u + guidance_vjp(pipe, z, t, en, ep, w)
                            + ctrl.sail_alpha * z / per_frame(frame_norm(z)), update)
            print(f"SAIL round {r} (seed {seed + r}, {ctrl.update_frames} frames): max loss over frames {loss.max():.3f} after {step} steps "
                  f"(threshold {thres:.2f})")
            if loss.max() <= thres:
                break
        ctrl.sail_log.append({"method": "sail", "update_frames": ctrl.update_frames, "loss": best[0], "threshold": thres,
                              "round": best[2], "seed": seed + best[2], "steps": best[3], "reached": best[0] <= thres,
                              "frame_loss": best[4], "frame_sharpness_before": first, "frame_sharpness": best[5],
                              "frame_x_norm": best[6], "loss_name": "||eps~(x + d)|| + alpha ||x||",
                              "frame_updates": sum(sum(map(sum, rd["updated"])) for rd in trace), "trace": trace})
        videos.append(best[1])
    return torch.cat(videos).to(pipe.unet.dtype)


@torch.no_grad()
def sample(pipe, prompt, ctrl, frames=16, steps=25, height=512, width=512, seed=0,
           s=7.5, r=0.0, r_schedule="cosine"):
    """Generate a video and return it as a list of PIL images.
    prompt can also be a list of prompts, generated together in one batch; a list of videos is returned then.
    Every video starts from the same noise (from seed), so it is the same video as when generated alone.

    ctrl.num_streams == 2 -> classifier-free guidance:
        eps = eps_null + s * (eps_cond - eps_null)
    ctrl.num_streams == 3 -> GUARD (paper Eq. 5):
        eps = eps_null + s * (eps_plus - eps_null) - r_t * (eps_minus - eps_null)
      eps_plus  : prompt, with cross-attention spikes attenuated (done by the attention processors)
      eps_minus : prompt, normal attention (the possibly memorized direction we are pushed away from)
      r_t       : r * cos(pi/2 * i/steps) with the "cosine" schedule (official code), or r ("constant")
      ctrl.ortho: (eps_plus - eps_null) is first orthogonalized against (eps_minus - eps_null) where they are
                  almost parallel, as the official code does (see orthogonalize)
    """
    device, dtype = pipe.device, pipe.unet.dtype

    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    e_prompt = embed_prompts(pipe, prompts, ctrl)       # [videos, 77, 768]
    e_null = embed_prompts(pipe, [""], ctrl, null=True).expand_as(e_prompt)
    # Positive and negative streams use the same prompt; only the attention inside the UNet differs.
    # Rows are ordered stream, video, frame, so each stream is still one contiguous block for the processors.
    streams = [e_null, e_prompt, e_prompt] if ctrl.num_streams == 3 else [e_null, e_prompt]
    text = torch.cat(streams).repeat_interleave(frames, dim=0)  # one copy per frame: [streams * videos * frames, 77, 768]

    pipe.scheduler.set_timesteps(steps, device=device)
    latents = torch.cat([torch.randn((1, 4, frames, height // 8, width // 8), generator=torch.Generator(device).manual_seed(seed),
                                     device=device, dtype=dtype) for _ in prompts])
    latents = latents * pipe.scheduler.init_noise_sigma
    if ctrl.sail:  # SAIL: sharpness-aware initial noise (Jeon et al.)
        latents = sail(pipe, latents, pipe.scheduler.timesteps[0], e_null, e_prompt, ctrl, seed)
    elif ctrl.noise_mode:  # initial-noise adjustment (Han et al.)
        latents = adjust_noise(pipe, latents, pipe.scheduler.timesteps[0], e_null, e_prompt, ctrl)

    for i, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="denoising")):
        x = pipe.scheduler.scale_model_input(torch.cat([latents] * len(streams)), t)
        eps = pipe.unet(x, t, encoder_hidden_states=text).sample.chunk(len(streams))
        if len(streams) == 3:
            eps_null, eps_plus, eps_minus = eps
            r_t = r * math.cos(math.pi / 2 * i / steps) if r_schedule == "cosine" else r
            v_plus = eps_plus - eps_null
            if ctrl.ortho:
                v_plus = orthogonalize(v_plus, eps_minus - eps_null)
            noise = eps_null + s * v_plus - r_t * (eps_minus - eps_null)
        else:
            eps_null, eps_cond = eps
            noise = eps_null + s * (eps_cond - eps_null)
            if ctrl.cfg_start is not None and t > ctrl.cfg_start:  # CFG delayed: unconditional prediction only
                noise = eps_null
            if ctrl.record:  # strength of the text guidance per video and frame (Wen et al.): [videos, frames]
                ctrl.guidance.append((eps_cond - eps_null).float().transpose(1, 2).flatten(2).norm(dim=-1).cpu())
        latents = pipe.scheduler.step(noise, t, latents).prev_sample

    torch.cuda.empty_cache()
    video = pipe.decode_latents(latents, decode_chunk_size=4)  # [videos, 3, frames, H, W]; 4 frames at a time saves memory
    videos = pipe.video_processor.postprocess_video(video, output_type="pil")
    return videos[0] if isinstance(prompt, str) else videos


def save_gif(frames, path, fps=8):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
