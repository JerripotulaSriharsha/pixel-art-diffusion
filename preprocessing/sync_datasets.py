import os
import argparse


def get_base_filenames(directory):
    """Get all base filenames (without extension) from a directory."""
    base_names = set()
    for file in os.listdir(directory):
        if file.endswith((".png", ".txt")):
            base_name = os.path.splitext(file)[0]
            base_names.add(base_name)
    return base_names


def sync_datasets(reference_path, target_path):
    """Remove files from target dataset that don't exist in reference dataset and sync text files."""
    # Get base filenames from both datasets
    reference_files = get_base_filenames(reference_path)
    target_files = get_base_filenames(target_path)

    print(f"Files in reference dataset: {len(reference_files)}")
    print(f"Files in target dataset: {len(target_files)}")

    # Determine which files to remove from target
    files_to_remove = target_files - reference_files
    common_files = reference_files & target_files

    print(f"Files to remove from target dataset: {len(files_to_remove)}")
    print(f"Text files to sync from reference to target: {len(common_files)}")

    # Remove files from target dataset
    removed_count = 0
    for base_name in files_to_remove:
        # Remove both .png and .txt files
        for ext in [".png", ".txt"]:
            file_path = os.path.join(target_path, base_name + ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed: {base_name + ext}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    # Sync text files from reference to target
    synced_count = 0
    for base_name in common_files:
        reference_txt = os.path.join(reference_path, base_name + ".txt")
        target_txt = os.path.join(target_path, base_name + ".txt")

        if os.path.exists(reference_txt):
            try:
                import shutil

                shutil.copy2(reference_txt, target_txt)
                print(f"Synced text file: {base_name}.txt")
                synced_count += 1
            except OSError as e:
                print(f"Error syncing {reference_txt} to {target_txt}: {e}")

    print(f"\nSynchronization complete!")
    print(f"Removed {removed_count} files from {target_path}")
    print(f"Synced {synced_count} text files from reference to target")


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


def rename_datasets(reference_path, target_path):
    """Rename files in both datasets with consistent numbering."""
    print("\n" + "=" * 50)
    print("RENAMING PHASE")
    print("=" * 50)

    # Get base filenames from reference dataset (this is our reference)
    reference_files = get_base_filenames(reference_path)

    # Sort the filenames to ensure consistent ordering
    sorted_base_names = sorted(list(reference_files))

    print(f"Found {len(sorted_base_names)} unique sprites to rename")
    print(
        f"Will rename from dq_sprite_0000 to dq_sprite_{len(sorted_base_names)-1:04d}"
    )

    # Rename files in both directories
    rename_files_in_directory(reference_path, sorted_base_names)
    rename_files_in_directory(target_path, sorted_base_names)

    print(f"\nRenaming complete! Both datasets now have consistent naming.")


if __name__ == "__main__":
    reference_path = "data/cleaned_sprites_v5/dataset-512"
    target_path = "data/cleaned_sprites_v5/dataset-128"
    no_rename = False

    print("Synchronizing datasets...")
    print(f"Reference (source of truth): {reference_path}")
    print(f"Target (will be synced): {target_path}")
    print()

    # Verify directories exist
    if not os.path.exists(reference_path):
        print(f"Error: Reference dataset directory not found: {reference_path}")
        exit(1)

    if not os.path.exists(target_path):
        print(f"Error: Target dataset directory not found: {target_path}")
        exit(1)

    # First sync the datasets
    sync_datasets(reference_path, target_path)

    # Then rename files in both datasets (if not disabled)
    if not no_rename:
        rename_datasets(reference_path, target_path)
