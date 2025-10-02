import os
import argparse
from pathlib import Path
from PIL import Image
import shutil


def downscale_dataset(source_dir, target_dir, target_resolution):
    """
    Downscale all PNG images in source_dir to target_resolution using nearest neighbor
    interpolation and copy corresponding txt files.

    Args:
        source_dir: Path to source dataset directory
        target_dir: Path to target dataset directory (will be created if doesn't exist)
        target_resolution: Target resolution as tuple (width, height) or single int for square
    """
    # Convert to Path objects for easier handling
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Get all PNG files
    png_files = sorted(source_path.glob("*.png"))

    if not png_files:
        print(f"No PNG files found in {source_dir}")
        return

    print(f"Found {len(png_files)} PNG files to process")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Target resolution: {target_resolution}")
    print()

    processed_count = 0
    error_count = 0

    for png_file in png_files:
        base_name = png_file.stem

        try:
            # Load and downscale image using nearest neighbor
            img = Image.open(png_file)

            # Downscale using nearest neighbor (NEAREST)
            downscaled_img = img.resize(target_resolution, Image.NEAREST)

            # Save downscaled image
            target_png = target_path / f"{base_name}.png"
            downscaled_img.save(target_png)

            # Copy corresponding txt file if it exists
            source_txt = source_path / f"{base_name}.txt"
            target_txt = target_path / f"{base_name}.txt"

            if source_txt.exists():
                shutil.copy2(source_txt, target_txt)
                print(f"Processed: {base_name} ({img.size} -> {target_resolution})")
            else:
                print(f"Warning: No txt file found for {base_name}.png")

            processed_count += 1

        except Exception as e:
            print(f"Error processing {png_file.name}: {e}")
            error_count += 1

    print()
    print("=" * 50)
    print(f"Downscaling complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors: {error_count}")
    print("=" * 50)


if __name__ == "__main__":
    downscale_dataset(
        source_dir="data/cleaned_sprites_v4/dataset-512",
        target_dir="data/cleaned_sprites_v4/dataset-256",
        target_resolution=(256, 256),
    )
