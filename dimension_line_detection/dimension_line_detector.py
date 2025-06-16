import os
import sys
from typing import List

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import ultralytics

from object_detection_base import BaseYOLO


class DimensionLineDetector(BaseYOLO):
    """
    YOLO-based detector for extracting dimension lines from input images.
    Inherits from BaseYOLO and implements post-processing to crop detected regions.
    """
    def postprocess(
        self, results: ultralytics.engine.results.Results
    ) -> List[np.ndarray]:
        """
        Process YOLO model output to extract regions containing dimension lines.
        
        Args:
            results (ultralytics.engine.results.Results): YOLO model prediction results (from the SectionDetection model).
            
        Returns:
            List[np.ndarray]: Cropped image regions containing dimension lines.
            List[List[float]]: Corresponding bounding box data [x1, y1, x2, y2, conf, class_id].
        """
        result = results[0]  # one image, so first result
        if (
            result.boxes is None
            or result.boxes.data is None
            or len(result.boxes.data) == 0
        ):
            return [], []  # no detections

        bbxes = result.boxes.data.cpu().numpy().tolist()
        orig_image = result.orig_img
        dimension_regions = []

        for bbx in bbxes:
            x1, y1, x2, y2, conf_score, cls = bbx
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            region = orig_image[y1:y2, x1:x2, :]
            dimension_regions.append(region)

        return dimension_regions, bbxes


if __name__ == "__main__":
    pass
