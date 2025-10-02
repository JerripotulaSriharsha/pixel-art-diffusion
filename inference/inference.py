from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import time
import os
from PIL import Image


def generate_images(
    base_model: str,
    lora_path: str,
    prompt: str,
    inference_height: int,
    inference_width: int,
    output_dir: str,
    num_images: int,
    steps: int,
    guidance: float,
    device: str,
    image_upscale_factor: int = 1,
    vae_path: str = None,
    negative_prompt: str = None,
):
    """Generate images using a Stable Diffusion pipeline with LoRA weights."""
    if vae_path:
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
    else:
        vae = None

    # Load pipeline and inject VAE
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        height=inference_height,
        width=inference_width,
        vae=vae,
    ).to(device)

    # Load LoRA weights
    pipe.load_lora_weights(lora_path)

    # Generate images
    os.makedirs(output_dir, exist_ok=True)
    images = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        negative_prompt=negative_prompt,
    ).images

    # Save with timestamp
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
        print(f"Saved {filename}")

    return images


generate_images(
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    lora_path="models/pixel-art-lora-sdxl-v7",
    prompt="robot, dq_pookie",
    negative_prompt="blurry, bad quality, low quality, bad composition, bad anatomy",
    inference_height=512,
    inference_width=512,
    output_dir="output",
    num_images=4,
    image_upscale_factor=1,
    steps=50,
    guidance=7.5,
    device="cuda",
    vae_path="madebyollin/sdxl-vae-fp16-fix",
)
