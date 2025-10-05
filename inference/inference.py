from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import time
import os
from PIL import Image
from postprocessing.clean_sprite import clean

"""Script to generate images using my trained pixel monsters lora."""


def generate_images(
    prompt: str,
    output_dir: str,
    num_images: int,
    steps: int,
    guidance: float,
    device: str,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_path: str = "pookie3000/pixel-monsters-lora-sdxl",
    inference_height: int = 512,
    inference_width: int = 512,
    clean_sprite: bool = True,
    image_upscale_factor: int = 1,
    vae_path: str = None,
    negative_prompt: str = None,
):
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    )
    if vae_path:
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        pipe.vae = vae
    pipe.to(device)
    pipe.load_lora_weights(lora_path)
    os.makedirs(output_dir, exist_ok=True)
    images = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        negative_prompt=negative_prompt,
        height=inference_height,
        width=inference_width,
        safety_checker=None,
    ).images

    timestamp = int(time.time())
    for i, img in enumerate(images):
        img = img.resize(
            (
                inference_height * image_upscale_factor,
                inference_width * image_upscale_factor,
            ),
            resample=Image.Resampling.NEAREST,
        )
        filename = os.path.join(
            output_dir, f"{os.path.basename(lora_path)}-{timestamp}-{i}.png"
        )
        img.save(filename)
        if clean_sprite:
            clean(filename, filename.replace(".png", "_clean.png"))
        print(f"Saved {filename}")

    return images


generate_images(
    prompt="taxi, dq_pookie",  # specify keywords with comma separation, last keyword is always "dq_pookie"
    # negative_prompt="",
    output_dir="output",
    num_images=40,
    steps=50,
    guidance=7.5,
    device="cuda",
    vae_path="madebyollin/sdxl-vae-fp16-fix",
    clean_sprite=False,
)
