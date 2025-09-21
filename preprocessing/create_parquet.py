from datasets import Dataset, Features, Value, Image
import os

DATASET_DIR = "data/cleaned_sprites_good/dataset-1024/"

records = []
file_names = os.listdir(DATASET_DIR)
file_names.sort()
for filename in file_names:
    if filename.endswith(".png"):
        base_name = os.path.splitext(filename)[0]
        text_file_path = os.path.join(DATASET_DIR, base_name + ".txt")

        with open(text_file_path, "r") as f:
            text = f.read().strip()

        # Only store bytes
        with open(os.path.join(DATASET_DIR, filename), "rb") as img_f:
            img_bytes = img_f.read()

        records.append({"image": {"bytes": img_bytes}, "text": text})

features = Features(
    {
        "image": Image(),
        "text": Value("string"),
    }
)

ds = Dataset.from_list(records, features=features)
ds.to_parquet("data/dataset-1024.parquet")
