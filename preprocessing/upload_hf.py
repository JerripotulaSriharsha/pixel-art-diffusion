import os
import time
from huggingface_hub import login
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file

load_dotenv("config/.env")

login(os.environ.get("HF_TOKEN"))


api = HfApi()
repo_url = api.create_repo(
    repo_id="pixel-art-finetune-dataset-1024-v2",
    repo_type="dataset",
    private=False,
)

# wait for 10 seconds (until repo is created)
time.sleep(10)
upload_file(
    path_or_fileobj="data/cleaned_sprites_v2/dataset-1024.parquet",
    path_in_repo="data/dataset-1024.parquet",
    repo_id="pookie3000/pixel-art-finetune-dataset-1024-v2",
    repo_type="dataset",
)
