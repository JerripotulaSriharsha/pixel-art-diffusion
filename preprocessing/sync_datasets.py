import os

# Configuration variables
DATASET_512_PATH = "/Users/woutvossen/Documents/programming/pixel-art-finetune/data/cleaned_sprites_good/dataset-512"
DATASET_1024_PATH = "/Users/woutvossen/Documents/programming/pixel-art-finetune/data/cleaned_sprites_good/dataset-1024"


def get_base_filenames(directory):
    """Get all base filenames (without extension) from a directory."""
    base_names = set()
    for file in os.listdir(directory):
        if file.endswith((".png", ".txt")):
            base_name = os.path.splitext(file)[0]
            base_names.add(base_name)
    return base_names


def sync_datasets():
    """Remove files from 1024 dataset that don't exist in 512 dataset."""
    # Get base filenames from both datasets
    dataset_512_files = get_base_filenames(DATASET_512_PATH)
    dataset_1024_files = get_base_filenames(DATASET_1024_PATH)

    print(f"Files in 512 dataset: {len(dataset_512_files)}")
    print(f"Files in 1024 dataset: {len(dataset_1024_files)}")

    # Find files in 1024 that are not in 512
    files_to_remove = dataset_1024_files - dataset_512_files

    print(f"Files to remove from 1024 dataset: {len(files_to_remove)}")

    if not files_to_remove:
        print("No files need to be removed. Datasets are already synchronized.")
        return

    # Remove files from 1024 dataset
    removed_count = 0
    for base_name in files_to_remove:
        # Remove both .png and .txt files
        for ext in [".png", ".txt"]:
            file_path = os.path.join(DATASET_1024_PATH, base_name + ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed: {base_name + ext}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    print(f"\nSynchronization complete!")
    print(f"Removed {removed_count} files from {DATASET_1024_PATH}")

    # Verify final counts
    final_512_files = get_base_filenames(DATASET_512_PATH)
    final_1024_files = get_base_filenames(DATASET_1024_PATH)

    print(f"Final count - 512 dataset: {len(final_512_files)} files")
    print(f"Final count - 1024 dataset: {len(final_1024_files)} files")


def rename_files_in_directory(directory, sorted_base_names):
    """Rename files in a directory to dq_sprite_XXXX format."""
    print(f"\nRenaming files in {directory}...")

    renamed_count = 0
    for index, old_base_name in enumerate(sorted_base_names):
        new_base_name = f"dq_sprite_{index:04d}"

        # Rename both .png and .txt files
        for ext in [".png", ".txt"]:
            old_file_path = os.path.join(directory, old_base_name + ext)
            new_file_path = os.path.join(directory, new_base_name + ext)

            if os.path.exists(old_file_path):
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {old_base_name + ext} -> {new_base_name + ext}")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming {old_file_path}: {e}")

    print(f"Renamed {renamed_count} files in {directory}")


def rename_datasets():
    """Rename files in both datasets with consistent numbering."""
    print("\n" + "=" * 50)
    print("RENAMING PHASE")
    print("=" * 50)

    # Get base filenames from 512 dataset (this is our reference)
    dataset_512_files = get_base_filenames(DATASET_512_PATH)

    # Sort the filenames to ensure consistent ordering
    sorted_base_names = sorted(list(dataset_512_files))

    print(f"Found {len(sorted_base_names)} unique sprites to rename")
    print(
        f"Will rename from dq_sprite_0000 to dq_sprite_{len(sorted_base_names)-1:04d}"
    )

    # Rename files in both directories
    rename_files_in_directory(DATASET_512_PATH, sorted_base_names)
    rename_files_in_directory(DATASET_1024_PATH, sorted_base_names)

    print(f"\nRenaming complete! Both datasets now have consistent naming.")


if __name__ == "__main__":
    print("Synchronizing datasets...")
    print(f"Source (reference): {DATASET_512_PATH}")
    print(f"Target (to be synced): {DATASET_1024_PATH}")
    print()

    # Verify directories exist
    if not os.path.exists(DATASET_512_PATH):
        print(f"Error: 512 dataset directory not found: {DATASET_512_PATH}")
        exit(1)

    if not os.path.exists(DATASET_1024_PATH):
        print(f"Error: 1024 dataset directory not found: {DATASET_1024_PATH}")
        exit(1)

    # First sync the datasets
    sync_datasets()

    # Then rename files in both datasets
    rename_datasets()
