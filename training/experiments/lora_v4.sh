#!/bin/bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_NAME="pookie3000/pookie-pixel-512"

accelerate launch ../train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path="stabilityai/sd-vae-ft-mse" \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=25000 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="pixel-art-lora" \
  --validation_prompt="house, tongue, dq_pookie" \
  --report_to="wandb"
