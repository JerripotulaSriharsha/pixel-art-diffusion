import argparse
import os

from safetensors.torch import load_file, save_file

from diffusers.utils import convert_all_state_dict_to_peft, convert_state_dict_to_kohya


"""
Script to convert a LoRA from diffusers to comfyui format.
"""


def convert_and_save(input_lora, output_lora=None):
    if output_lora is None:
        base_name = os.path.splitext(input_lora)[0]
        output_lora = f"{base_name}_webui.safetensors"

    diffusers_state_dict = load_file(input_lora)
    peft_state_dict = convert_all_state_dict_to_peft(diffusers_state_dict)
    kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
    save_file(kohya_state_dict, output_lora)


INPUT_LORA = "models/pookiepixel.safetensors"
OUTPUT_LORA = "models/pookiepixel_webui.safetensors"

convert_and_save(INPUT_LORA, OUTPUT_LORA)
