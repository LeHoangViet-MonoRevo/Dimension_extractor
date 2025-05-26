from typing import Union

import cv2
import numpy as np

from dimension_line_detection.dimension_line_detector import \
    DimensionLineDetector
from section_detection.section_detector import SectionDetector


class DimensionExtractor:
    def __init__(
        self, section_detector_path: str, dimension_line_detector_path: str
    ) -> None:

        self.section_detector = SectionDetector(section_detector_path)
        self.dimension_line_detector = DimensionLineDetector(
            dimension_line_detector_path
        )

    def run(self, image: Union[str, np.ndarray]):
        if isinstance(image, str):
            image = cv2.imread(image)

        assert image is not None, "Image is None, please re-check!!"
        cross_sections, cross_section_bbxes = self.section_detector.run(image)
        dimension_line_regions = []
        for cross_section in cross_sections:
            dimension_line_regions.extend(
                self.dimension_line_detector.run(cross_section)[0]
            )

        return dimension_line_regions


if __name__ == "__main__":
    section_detector_path = "section_detection_best.pt"
    dimension_line_detector_path = "dimension_line_detector.pt"
    dimension_extractor = DimensionExtractor(
        section_detector_path, dimension_line_detector_path
    )

    image = "../image_company/113/2d7573f20f7cb2b911135543119fc7d7_0.jpg"
    dimension_line_regions = dimension_extractor.run(image)
    print(f"Num dimension regions: {len(dimension_line_regions)}")
