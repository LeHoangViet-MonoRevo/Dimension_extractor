import os
import sys
from typing import List

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import ultralytics

from object_detection_base import BaseYOLO


class DimensionLineDetector(BaseYOLO):
    def postprocess(
        self, results: ultralytics.engine.results.Results
    ) -> List[np.ndarray]:
        """
        Receive the cross section and return the detection of dimension lines.
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
