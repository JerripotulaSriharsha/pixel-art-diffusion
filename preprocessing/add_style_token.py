from pathlib import Path

style_token = "dq_pookie"


def add_style_token(text):
    return f"{text}, {style_token}"


def preprocess_text(text):
    text = text.replace('"', "")
    text = text.replace(".", "")
    return text


def main():
    base_dir = Path(__file__).parent.parent / "data" / "cleaned_sprites_v3"
    dataset_512_dir = base_dir / "dataset-512"
    dataset_1024_dir = base_dir / "dataset-1024"

    for file in dataset_512_dir.glob("*.txt"):
        with open(file, "r") as f:
            text = f.read()
            text = add_style_token(preprocess_text(text))
            with open(file, "w") as f:
                f.write(text)

    for file in dataset_1024_dir.glob("*.txt"):
        with open(file, "r") as f:
            text = f.read()
            text = add_style_token(preprocess_text(text))
            with open(file, "w") as f:
                f.write(text)

    print("Done")


if __name__ == "__main__":
    main()
