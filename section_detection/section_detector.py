import os
import sys
from typing import List

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import ultralytics

from object_detection_base import BaseYOLO


class SectionDetector(BaseYOLO):
    """
    Ultralytics YOLO-based detector for extracting cross section regions from an engineering drawing.

    Inherits from:
        BaseYOLO: Custom base class for handling YOLO detection logic.
    """
    def postprocess(
        self, results: ultralytics.engine.results.Results
    ) -> List[np.ndarray]:
        """
        Process YOLO model output to extract image regions containing cross sections.

        Args:
            results (ultralytics.engine.results.Results): Detection results from the YOLO model.

        Returns:
            List[np.ndarray]: Cropped image regions of detected cross sections.
            List[List[float]]: Corresponding bounding boxes [x1, y1, x2, y2, conf, class_id].
        """
        if not results or len(results) == 0:
            return [], []

        result = results[0]

        if (
            result.boxes is None
            or result.boxes.data is None
            or len(result.boxes.data) == 0
        ):
            return [], []

        bbxes = result.boxes.data.cpu().numpy().tolist()
        orig_image = result.orig_img
        sections = []

        for bbx in bbxes:
            x1, y1, x2, y2, conf_score, cls = bbx
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            sections.append(orig_image[y1:y2, x1:x2, :])

        return sections, bbxes


if __name__ == "__main__":
    import cv2

    model = SectionDetector("section_detection_best.pt")
    image = cv2.imread("84b1b33011a7b22d219aaf627f41868a_2.jpg")

    sections = model.run(image)
    plt.imshow(sections[0])
    plt.show()
    print(f"sections: {sections}")
