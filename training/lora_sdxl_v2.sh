#!/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="pookie3000/pixel-art-finetune-dataset-1024-v2"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --pretrained_vae_model_name_or_path="$VAE_NAME" \
  --dataset_name="$DATASET_NAME" --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=10 \
  --checkpointing_steps=300 \
  --gradient_accumulation_steps=8 \
  --learning_rate=3e-05 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --snr_gamma=5.0 \
  --noise_offset=0.02 \
  --rank=64 \
  --train_text_encoder \
  --seed=42 \
  --output_dir="pixel-art-lora-sdxl-v2" \
  --validation_prompt="A large undead construct made of rusted iron and bones" \
  --report_to="wandb"
