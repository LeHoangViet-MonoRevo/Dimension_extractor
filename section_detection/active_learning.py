import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from section_detector import SectionDetector
from tqdm import tqdm


class ActiveLearningSectionSampler:
    def __init__(
        self,
        model_path: str,
        image_dir: str,
        save_dir: str = "active_samples",
        conf_threshold: float = 0.3,
    ):
        self.model = SectionDetector(model_path)
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.conf_threshold = conf_threshold
        os.makedirs(save_dir, exist_ok=True)

    def process_images(self):
        image_paths = list(Path(self.image_dir).rglob("*.jpg"))

        for img_path in tqdm(image_paths, desc="Scanning images"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Use .run() method from SectionDetector
            sections, bbxes = self.model.run(image)

            if not bbxes:
                continue

            low_conf_boxes = []

            for bbx in bbxes:
                x1, y1, x2, y2, conf, cls = bbx
                if conf < self.conf_threshold:
                    low_conf_boxes.append(([x1, y1, x2, y2], conf))

            if low_conf_boxes:
                self.save_candidate(image, img_path, low_conf_boxes)

    def save_candidate(self, image, img_path, boxes):
        """
        Save the image with low-confidence boxes drawn on it.
        """
        filename = Path(img_path).name
        save_path = os.path.join(self.save_dir, filename)

        # Draw annotations
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_rgb)

        for box, conf in boxes:
            x1, y1, x2, y2 = map(int, box)
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="orange",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 10, f"{conf:.2f}", color="orange", backgroundcolor="white")

        ax.axis("off")
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == "__main__":
    sampler = ActiveLearningSectionSampler(
        model_path="section_detection_best.pt",
        image_dir="../image_company/",
        save_dir="active_learning_candidates",
        conf_threshold=0.4,  # Save samples where confidence < 0.4
    )

    sampler.process_images()
