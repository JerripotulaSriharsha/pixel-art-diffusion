#!/bin/bash

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_NAME="pookie3000/pixel-art-finetune-dataset-512-v4"

accelerate launch ../train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="pixel-art-lora-v1" \
  --validation_prompt="undead, machine, dq_pookie" --report_to="wandb" \