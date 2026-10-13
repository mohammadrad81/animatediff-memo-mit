"""Load AnimateDiff and run the denoising loop (normal CFG or GUARD)."""
import math

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
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
    return pipe.to(device)


@torch.no_grad()
def sample(pipe, prompt, ctrl, frames=16, steps=25, height=512, width=512, seed=0,
           s=7.5, r=0.0, r_schedule="cosine"):
    """Generate a video and return it as a list of PIL images.

    ctrl.num_streams == 2 -> classifier-free guidance:
        eps = eps_null + s * (eps_cond - eps_null)
    ctrl.num_streams == 3 -> GUARD (paper Eq. 5):
        eps = eps_null + s * (eps_plus - eps_null) - r_t * (eps_minus - eps_null)
      eps_plus  : prompt, with cross-attention spikes attenuated (done by the attention processors)
      eps_minus : prompt, normal attention (the possibly memorized direction we are pushed away from)
      r_t       : r * cos(pi/2 * i/steps) with the "cosine" schedule (official code), or r ("constant")
    """
    device, dtype = pipe.device, pipe.unet.dtype

    def embed(text):
        ids = pipe.tokenizer(text, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids
        return pipe.text_encoder(ids.to(device))[0]  # [1, 77, 768]

    e_null, e_prompt = embed(""), embed(prompt)
    # Positive and negative streams use the same prompt; only the attention inside the UNet differs.
    streams = [e_null, e_prompt, e_prompt] if ctrl.num_streams == 3 else [e_null, e_prompt]
    text = torch.cat(streams).repeat_interleave(frames, dim=0)  # one copy per frame: [streams * frames, 77, 768]

    pipe.scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn((1, 4, frames, height // 8, width // 8), generator=generator, device=device, dtype=dtype)
    latents = latents * pipe.scheduler.init_noise_sigma

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
        latents = pipe.scheduler.step(noise, t, latents).prev_sample

    torch.cuda.empty_cache()
    video = pipe.decode_latents(latents, decode_chunk_size=4)  # [1, 3, frames, H, W]; 4 frames at a time saves memory
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]


def save_gif(frames, path, fps=8):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
