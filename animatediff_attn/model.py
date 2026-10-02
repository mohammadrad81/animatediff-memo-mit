"""Loading AnimateDiff (SD v1.4 + motion adapter) and running a recorded generation."""
from __future__ import annotations

import torch

MODEL_ID = "CompVis/stable-diffusion-v1-4"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-4"

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_pipeline(model_id=MODEL_ID, adapter_id=MOTION_ADAPTER_ID, device="cuda", dtype="fp16", cpu_offload=False):
    from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

    torch_dtype = DTYPES[dtype]
    adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=torch_dtype)
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch_dtype)
    # Scheduler settings recommended for AnimateDiff.
    pipe.scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.enable_vae_slicing()
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def load_unet_only(model_id=MODEL_ID, adapter_id=MOTION_ADAPTER_ID):
    """Lightweight (CPU) UNetMotionModel, enough to enumerate the attention layers."""
    from diffusers import MotionAdapter, UNet2DConditionModel, UNetMotionModel

    unet2d = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    adapter = MotionAdapter.from_pretrained(adapter_id)
    return UNetMotionModel.from_unet2d(unet2d, adapter)


def token_labels(tokenizer, prompt: str) -> tuple[list[str], int]:
    """Labels for all 77 text positions and the index of the end-of-text token."""
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True).input_ids
    eos = tokenizer.eos_token_id
    eot = ids.index(eos, 1)
    labels = []
    for i, tid in enumerate(ids):
        if i == 0:
            labels.append("<sot>")
        elif i == eot:
            labels.append("<eot>")
        elif i > eot:
            labels.append("<pad>")
        else:
            labels.append(tokenizer.convert_ids_to_tokens(tid).replace("</w>", ""))
    return labels, eot


def generate(pipe, recorder, prompt, negative_prompt=None, num_frames=16, height=512, width=512,
             num_inference_steps=25, guidance_scale=7.5, seed=0):
    generator = torch.Generator("cpu").manual_seed(seed)
    recorder.attach(pipe.unet)
    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    finally:
        recorder.detach(pipe.unet)
    return out.frames[0]  # list of PIL images
