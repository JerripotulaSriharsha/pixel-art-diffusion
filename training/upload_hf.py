"""Upload a model to Hugging Face."""

from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="models/pookie-pixel-lora-sdxl",
    repo_id="pookie3000/pookie-pixel-lora-sdxl",
    repo_type="model",
)
