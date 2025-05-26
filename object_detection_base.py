from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from ultralytics import YOLO


class BaseYOLO(ABC):
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = str(model_path)
        self.model = self.load_model()

    def load_model(self) -> YOLO:
        """Load a YOLO model using Ultralytics interface."""
        return YOLO(self.model_path)

    def run(
        self, image: Union[str, Path, np.ndarray], verbose: bool = False
    ) -> List[dict]:
        """Run inference and return detections in a standard format."""
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))

        results = self.model(image, verbose=verbose)[0]
        return self.postprocess(results)

    @abstractmethod
    def postprocess(self, results) -> List[dict]:
        """Postprocess Ultralytics results object into a list of dicts."""
        pass

    @staticmethod
    def visualise(
        image: np.ndarray, detections: List[dict], class_names: List[str] = None
    ) -> np.ndarray:
        """Draw bounding boxes on the image."""
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            score = det["score"]
            class_id = det["class"]
            label = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return image


if __name__ == "__main__":
    model = BaseYOLO("section_detection_best.pt")
