import os
import base64
from pathlib import Path
from groq import Groq
import base64
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm
import time

load_dotenv("config/.env")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def before_sleep_print(retry_state):
    print(f"Timeout hit, retrying in {retry_state.next_action.sleep} seconds...")


prompt = """
You are helping prepare training captions for a LoRA model.  

Your task is to create short, consistent captions for each image.  
Rules:  
- Do NOT mention the unique style (that it's pixel art).
- Do NOT mention the magenta background color as this will be removed later.
- Don't say 'creature' but try to identify the subject (even if it is mytical)
- DO mention clothes, background, objects, pose, and anything that should be variable later.  
- Use 3–8 keywords separated by commas.  
- Do not write full sentences.  
- Keep the wording consistent across images.  

"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(60),
    reraise=True,
    before_sleep=before_sleep_print,
)
def groq_llm_call(image_path):
    base64_image = encode_image(image_path)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
        ],
        # model="meta-llama/llama-4-scout-17b-16e-instruct",
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
    )
    return chat_completion.choices[0].message.content


def save_label(txt_path, label):
    """Save label to text file."""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(label)


def process_sprites():
    """Process all sprites in dataset-512 and save labels to both datasets."""
    base_dir = Path(__file__).parent.parent / "data" / "cleaned_sprites_v2"
    dataset_512_dir = base_dir / "dataset-512"
    dataset_1024_dir = base_dir / "dataset-1024"

    if not dataset_512_dir.exists():
        print(f"Error: {dataset_512_dir} does not exist")
        return

    if not dataset_1024_dir.exists():
        print(f"Error: {dataset_1024_dir} does not exist")
        return

    # Get all PNG files in dataset-512
    png_files = list(dataset_512_dir.glob("*.png"))
    print(f"Found {len(png_files)} sprite files to process")
    png_files.sort()
    processed_count = 0
    skipped_count = 0

    for png_file in tqdm(png_files, desc="Processing Sprites", unit="sprite"):
        try:
            # Corresponding text files
            txt_512_path = dataset_512_dir / f"{png_file.stem}.txt"
            txt_1024_path = dataset_1024_dir / f"{png_file.stem}.txt"

            # Check if corresponding 1024 PNG exists
            png_1024_path = dataset_1024_dir / f"{png_file.name}"
            if not png_1024_path.exists():
                print(f"Warning: No corresponding 1024 version for {png_file.name}")
                skipped_count += 1
                continue

            label = groq_llm_call(png_file)

            # Save label to both datasets
            save_label(txt_512_path, label)
            save_label(txt_1024_path, label)

            processed_count += 1

            # to stay within groq free tier :)
            time.sleep(5)

        except Exception as e:
            print(f"Error processing {png_file.name}: {str(e)}")
            continue

    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} sprites")
    print(f"Skipped: {skipped_count} sprites")


def main():
    """Main function to run the sprite labeling process."""
    print("Starting sprite labeling process...")
    process_sprites()


if __name__ == "__main__":
    main()
