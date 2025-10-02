#!/bin/bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="pookie3000/pixel-art-finetune-dataset-1024-v2"


accelerate launch ../train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --pretrained_vae_model_name_or_path="$VAE_NAME" \
  --dataset_name="$DATASET_NAME" --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=8000 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --mixed_precision="fp16" \
  --rank=32 \
  --seed=42 \
  --output_dir="pixel-art-lora-sdxl-v4" \
  --validation_prompt="undead, rusted iron, bones" \
  --report_to="wandb"