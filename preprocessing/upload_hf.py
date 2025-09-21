import os
from huggingface_hub import login
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file

load_dotenv("config/.env")

login(os.environ.get("HF_TOKEN"))


api = HfApi()
repo_url = api.create_repo(
    repo_id="pixel-art-finetune-dataset-1024",
    repo_type="dataset",
    private=False,
)
print(repo_url)
upload_file(
    path_or_fileobj="data/dataset-1024.parquet",
    path_in_repo="data/dataset-1024.parquet",
    repo_id="pookie3000/pixel-art-finetune-dataset-1024",
    repo_type="dataset",
)
