import glob
import os
import random
import shutil
from typing import List

import cv2
import numpy as np
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
    Randomly move image-label pairs (based on image files) from training to validation.

    Arguments:
    - train_folder: path to training folder (must have 'images' and 'labels' subfolders)
    - val_folder: path to validation folder (must have 'images' and 'labels' subfolders)
    - val_ratio: fraction of image samples to move from train to val
    """
    train_img_dir = os.path.join(train_folder, "images")
    train_label_dir = os.path.join(train_folder, "labels")
    val_img_dir = os.path.join(val_folder, "images")
    val_label_dir = os.path.join(val_folder, "labels")

    # Ensure validation folders exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(".jpg")]
    num_to_move = int(len(image_files) * val_ratio)

    # Randomly select image files to move
    selected_images = random.sample(image_files, num_to_move)

    for img_file in selected_images:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"

        # Define full paths
        src_img_path = os.path.join(train_img_dir, img_file)
        src_label_path = os.path.join(train_label_dir, label_file)
        dst_img_path = os.path.join(val_img_dir, img_file)
        dst_label_path = os.path.join(val_label_dir, label_file)

        # Move image
        shutil.move(src_img_path, dst_img_path)

        # Move label if it exists
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dst_label_path)

    print(f"Moved {len(selected_images)} image-label pairs to validation set.")


def extract_cross_sections_from_label(
    image_path: str, label_path: str
) -> List[np.ndarray]:
    """
    Extract regions (bounding boxes) from an image based on YOLO label format.

    Parameters:
    - image_path: path to the image file
    - label_path: path to the corresponding label file

    Returns:
    - A list of cropped image regions as numpy arrays
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    height, width = image.shape[:2]

    # Read label file
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    regions = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines

            _, x_center, y_center, bbox_width, bbox_height = map(float, parts)

            # Convert from normalized to pixel coordinates
            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            # Calculate top-left and bottom-right coordinates
            x1 = int(x_center - bbox_width / 2)
            y1 = int(y_center - bbox_height / 2)
            x2 = int(x_center + bbox_width / 2)
            y2 = int(y_center + bbox_height / 2)

            # Clip coordinates to image size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Extract region
            region = image[y1:y2, x1:x2]
            regions.append(region)

    return regions


def get_image_label_pairs(image_paths, label_paths):
    # Convert to dict for quick lookup
    images_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    labels_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}

    # Find common base names
    common_keys = images_dict.keys() & labels_dict.keys()

    # Create pairs
    pairs = [(images_dict[k], labels_dict[k]) for k in common_keys]

    return pairs


if __name__ == "__main__":
    pass
