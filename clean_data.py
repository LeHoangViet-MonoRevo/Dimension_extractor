import glob
import os
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


if __name__ == "__main__":

    img_dirs = glob.glob("../demo_samples/selected_images/*")
    label_dirs = glob.glob("../demo_samples/labels/*")
    # selected_folder_dir = "../demo_samples/selected_images"

    # move_imgs_based_on_labels(img_dirs, label_dirs, selected_folder_dir)

    print(f"Num imgs: {len(img_dirs)}, Num labels: {len(label_dirs)}")
