"""Load AnimateDiff and run the denoising loop (normal CFG or GUARD)."""
import contextlib
import functools
import math

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

MODEL_ID = "CompVis/stable-diffusion-v1-4"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-4"


def load_pipeline(device="cuda", dtype=torch.float16):
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
def embed_prompts(pipe, prompts, ctrl=None):
    """CLIP text embeddings [prompts, 77, 768]. With ctrl.clip_mode set, the CLIP-embedding mitigation of Kim & No
    ("Memorization in Stable Diffusion is unexpectedly driven by CLIP embeddings") is applied:
      [v_sot, v_pr_1..v_pr_n, v_eot, v_pad_1..v_pad_d]   normal: every <pad> is another <eot> token, so the
                                                          v_pad are near copies of v_eot and amplify it
      "pad":  the prompt is padded with the "!" token instead of <eot> (v_pad no longer copy v_eot),
              and v_eot is set to 0 (unless ctrl.mask_eot is False)
      "mask": normal padding, but the first mask_ratio (default 70%) of the v_pad, the ones right after <eot>,
              are set to 0"""
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
        if mode == "pad" and ctrl.mask_eot:
            emb[i, eot] = 0
        elif mode == "mask":
            d = L - 1 - eot                          # number of padding embeddings
            emb[i, eot + 1:eot + 1 + round(ctrl.mask_ratio * d)] = 0
    return emb


def frame_norm(d):
    """[videos, 4, frames, h, w] -> [videos, frames]: L2 norm of every frame (each frame is like one SD image)."""
    return d.float().transpose(1, 2).flatten(2).norm(dim=-1)


def guidance_eps(pipe, x, t, e_null, e_prompt):
    """eps_cond - eps_null (the paper's eps~) at latents x [videos, 4, frames, h, w], in float32."""
    text = torch.cat([e_null, e_prompt]).repeat_interleave(x.shape[2], dim=0)
    x = pipe.scheduler.scale_model_input(x.to(pipe.unet.dtype), t)
    eps_null, eps_cond = pipe.unet(torch.cat([x, x]), t, encoder_hidden_states=text).sample.chunk(2)
    return (eps_cond - eps_null).float()


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
        until every frame has ||eps~|| <= target (or max_iters steps)."""
    x = x.float()
    before = frame_norm(guidance_eps(pipe, x, t, e_null, e_prompt))
    steps = [0] * x.shape[0]
    if ctrl.noise_mode == "batch":
        for _ in range(ctrl.adj_iters):
            d = guidance_eps(pipe, x, t, e_null, e_prompt)
            delta = ctrl.rho * d / frame_norm(d)[:, None, :, None, None]
            x = x - ctrl.gamma * (guidance_eps(pipe, x + delta, t, e_null, e_prompt) - d)
    else:
        frames = x.shape[2]
        videos = []
        for v in range(x.shape[0]):
            lat = x[v:v + 1].clone().requires_grad_(True)
            optim = torch.optim.AdamW([lat], lr=ctrl.lr)  # default weight decay 0.01 keeps x near N(0, I)
            en, ep = e_null[v:v + 1], e_prompt[v:v + 1]
            pbar = tqdm(range(ctrl.max_iters), desc="adjusting noise")
            for steps[v] in pbar:
                d = guidance_eps(pipe, lat.detach(), t, en, ep)
                n = frame_norm(d)
                pbar.set_postfix(max_norm=f"{n.max():.3f}", target=ctrl.target)
                if n.max() <= ctrl.target:
                    break
                # Gradient of sum_f ||eps_cond - eps_null|| = J_cond^T g - J_null^T g with g = eps~ / ||eps~||.
                # Computing the two streams one after the other halves the memory of one backward pass.
                g = d / n[:, None, :, None, None]
                optim.zero_grad()
                with torch.enable_grad(), checkpointed_blocks(pipe.unet):
                    for e, sign in [(ep, 1.0), (en, -1.0)]:
                        xin = pipe.scheduler.scale_model_input(lat.to(pipe.unet.dtype), t)
                        out = pipe.unet(xin, t, encoder_hidden_states=e.repeat_interleave(frames, dim=0)).sample
                        out.float().backward(sign * g)
                optim.step()
            else:
                steps[v] = ctrl.max_iters
            videos.append(lat.detach())
        x = torch.cat(videos)
    after = frame_norm(guidance_eps(pipe, x, t, e_null, e_prompt))
    for v in range(x.shape[0]):
        ctrl.noise_log.append((before[v].tolist(), after[v].tolist(), steps[v]))
        print(f"initial noise adjusted ({ctrl.noise_mode}): max over frames of ||eps_cond - eps_null|| "
              f"{before[v].max():.3f} -> {after[v].max():.3f}" + (f" in {steps[v]} steps" if ctrl.noise_mode == "sample" else ""))
    return x.to(pipe.unet.dtype)


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
    """
    device, dtype = pipe.device, pipe.unet.dtype

    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    e_prompt = embed_prompts(pipe, prompts, ctrl)       # [videos, 77, 768]
    e_null = embed_prompts(pipe, [""]).expand_as(e_prompt)  # the empty prompt is never changed
    # Positive and negative streams use the same prompt; only the attention inside the UNet differs.
    # Rows are ordered stream, video, frame, so each stream is still one contiguous block for the processors.
    streams = [e_null, e_prompt, e_prompt] if ctrl.num_streams == 3 else [e_null, e_prompt]
    text = torch.cat(streams).repeat_interleave(frames, dim=0)  # one copy per frame: [streams * videos * frames, 77, 768]

    pipe.scheduler.set_timesteps(steps, device=device)
    latents = torch.cat([torch.randn((1, 4, frames, height // 8, width // 8), generator=torch.Generator(device).manual_seed(seed),
                                     device=device, dtype=dtype) for _ in prompts])
    latents = latents * pipe.scheduler.init_noise_sigma
    if ctrl.noise_mode:  # initial-noise adjustment (Han et al.)
        latents = adjust_noise(pipe, latents, pipe.scheduler.timesteps[0], e_null, e_prompt, ctrl)

    for i, t in enumerate(tqdm(pipe.scheduler.timesteps, desc="denoising")):
        x = pipe.scheduler.scale_model_input(torch.cat([latents] * len(streams)), t)
        eps = pipe.unet(x, t, encoder_hidden_states=text).sample.chunk(len(streams))
        if len(streams) == 3:
            eps_null, eps_plus, eps_minus = eps
            r_t = r * math.cos(math.pi / 2 * i / steps) if r_schedule == "cosine" else r
            noise = eps_null + s * (eps_plus - eps_null) - r_t * (eps_minus - eps_null)
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
