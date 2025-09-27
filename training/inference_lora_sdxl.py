from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import time
import os

model_path = "models/pixel-art-lora-sdxl-v4"

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
prompt = "undead, rusted iron, bones"

num_images = 8  # how many images you want
os.makedirs("output", exist_ok=True)

images = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt="blurry, bad quality, low quality, bad composition, bad anatomy",
    num_images_per_prompt=num_images,
).images

timestamp = int(time.time())
for i, img in enumerate(images):
    filename = f"output/pixel-art-lora-sdxl-v4-{timestamp}-{i}.png"
    img.save(filename)
    print(f"Saved {filename}")
