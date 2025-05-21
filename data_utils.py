import glob
import os
import random
import shutil
from typing import List

from tqdm import tqdm


def move_imgs_based_on_labels(
    img_dirs: List[str], label_dirs: List[str], selected_folder_dir: str
) -> None:
    img_parent_dir = os.path.split(img_dirs[0])[0]
    missing_img_dir = os.path.join(selected_folder_dir, "missing_images")
    os.makedirs(selected_folder_dir, exist_ok=True)
    os.makedirs(missing_img_dir, exist_ok=True)

    for label_path in tqdm(label_dirs, desc="Processing labels"):
        label_filename = os.path.basename(label_path)
        img_filename = os.path.splitext(label_filename)[0] + ".jpg"
        img_path = os.path.join(img_parent_dir, img_filename)

        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(selected_folder_dir, img_filename))
        else:
            print(f"[Missing] Image not found for label: {label_filename}")
            shutil.copy(
                label_path, os.path.join(missing_img_dir, label_filename)
            )  # Save label to missing dir

    print(f"Finished moving images. Missing images info saved in: {missing_img_dir}")


def random_move_samples_to_val(
    train_folder: str, val_folder: str, val_ratio: float = 0.2
):
    """
    Randomly move image-label pairs from training folder to validation folder.

    Arguments:
    - train_folder: path to training folder (must have 'images' and 'labels' subfolders)
    - val_folder: path to validation folder (must have 'images' and 'labels' subfolders)
    - val_ratio: fraction of samples to move from train to val
    """
    train_img_dir = os.path.join(train_folder, "images")
    train_label_dir = os.path.join(train_folder, "labels")
    val_img_dir = os.path.join(val_folder, "images")
    val_label_dir = os.path.join(val_folder, "labels")

    # Create validation folders if they don't exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get all label files (assuming image files have the same base name with .jpg extension)
    label_files = [f for f in os.listdir(train_label_dir) if f.endswith(".txt")]
    num_to_move = int(len(label_files) * val_ratio)

    # Randomly select label files
    selected_labels = random.sample(label_files, num_to_move)

    for label_file in selected_labels:
        base_name = os.path.splitext(label_file)[0]
        img_file = base_name + ".jpg"

        # Define full paths
        src_label_path = os.path.join(train_label_dir, label_file)
        src_img_path = os.path.join(train_img_dir, img_file)
        dst_label_path = os.path.join(val_label_dir, label_file)
        dst_img_path = os.path.join(val_img_dir, img_file)

        # Move files
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dst_img_path)
            shutil.move(src_label_path, dst_label_path)
        else:
            print(f"[Warning] Image not found for label: {label_file}, skipping.")

    print(f"Moved {len(selected_labels)} image-label pairs to validation set.")


if __name__ == "__main__":
    pass
