from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import time
import os

model_path = "models/pixel-art-lora-sdxl-v1"

# Load VAE separately
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

# Load pipeline and inject VAE
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    vae=vae,  # pass the VAE object here
)

pipe.to("cuda")

# Load LoRA weights
pipe.load_lora_weights(model_path)

# Generate
prompt = "A rat with a red hat"

num_images = 8  # how many images you want
os.makedirs("output", exist_ok=True)

images = pipe(
    prompt,
    num_inference_steps=60,
    guidance_scale=7.5,
    negative_prompt="low quality, text, watermark, logo, brand, cropped, jpeg artifacts, signature, username, blurry, deformed, ugly, monochrome",
    num_images_per_prompt=num_images,
).images

timestamp = int(time.time())
for i, img in enumerate(images):
    filename = f"output/pixel-art-lora-sdxl-v1-{timestamp}-{i}.png"
    img.save(filename)
    print(f"Saved {filename}")
