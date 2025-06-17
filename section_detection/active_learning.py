import os
from pathlib import Path

import cv2
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
        Save the image with low-confidence boxes drawn on it using cv2.
        """
        filename = Path(img_path).name
        save_path = os.path.join(self.save_dir, filename)

        # Create a copy of the image to draw on
        image_with_boxes = image.copy()

        # Draw bounding boxes and confidence scores
        for box, conf in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Draw rectangle
            cv2.rectangle(
                image_with_boxes, (x1, y1), (x2, y2), (0, 165, 255), 2
            )  # Orange color in BGR

            # Prepare confidence text
            conf_text = f"{conf:.2f}"

            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                conf_text, font, font_scale, thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                image_with_boxes,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                (255, 255, 255),  # White background
                -1,
            )

            # Draw text
            cv2.putText(
                image_with_boxes,
                conf_text,
                (x1 + 2, y1 - 5),
                font,
                font_scale,
                (0, 165, 255),  # Orange text in BGR
                thickness,
            )

        # Save the image
        cv2.imwrite(save_path, image_with_boxes)


if __name__ == "__main__":
    sampler = ActiveLearningSectionSampler(
        model_path="section_detector_best_2nd.pt",
        image_dir="../image_company/",
        save_dir="active_learning_candidates",
        conf_threshold=0.4,  # Save samples where confidence < 0.4
    )

    sampler.process_images()
