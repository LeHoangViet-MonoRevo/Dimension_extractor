from typing import Union

import cv2
import numpy as np

from dimension_line_detection.dimension_line_detector import \
    DimensionLineDetector
from section_detection.section_detector import SectionDetector
from text_reader.text_reader import PaddleTextReader


class DimensionExtractor:
    """
    Pipeline for extracting object dimensions (Width x Height x Length) from engineering drawings.

    This class processes an input drawing to identify dimensional annotations
    and extract the corresponding measurements using a three-stage pipeline:

        ┌────────────────────┐
        │   Input Drawing    │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  SectionDetector   │  - Detects cross-sectional views.
        └────────┬───────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ DimensionLineDetector│ - Identifies dimension lines within sections.
        └────────┬─────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │   TextReader (OCR) │ - Reads dimension values from lines.
        └────────────────────┘

    Returns a structured list of detected dimension line regions along with
    their recognized texts, positions, and confidence scores.
    """

    def __init__(
        self, section_detector_path: str, dimension_line_detector_path: str
    ) -> None:

        self.section_detector = SectionDetector(section_detector_path)
        self.dimension_line_detector = DimensionLineDetector(
            dimension_line_detector_path
        )
        self.text_reader = PaddleTextReader()

    def run(self, image: Union[str, np.ndarray]):
        """
        Execute the full dimension extraction pipeline on a given image.

        Steps:
            1. Load image if a file path is provided.
            2. Detect cross sections using the SectionDetector.
            3. For each cross section, detect dimension lines.
            4. For each dimension line region, extract text using OCR.

        Args:
            image (Union[str, np.ndarray]): Input image, either as a file path or an OpenCV image (numpy array).

        Returns:
            List[Dict]: A list of results for each detected dimension line region.
                Each result is a dictionary with:
                    - 'image' (np.ndarray): Cropped image region of the dimension line.
                    - 'texts' (List[str]): Recognized text strings.
                    - 'polys' (List[List[Tuple[int, int]]]): Polygons (bounding regions) of detected texts.
                    - 'confs' (List[float]): Confidence scores for each text prediction.
        """

        if isinstance(image, str):
            image = cv2.imread(image)

        assert image is not None, "Image is None, please re-check!!"
        cross_sections, cross_section_bbxes = self.section_detector.run(image)
        dimension_line_regions = []
        for cross_section in cross_sections:
            dimension_line_regions.extend(
                self.dimension_line_detector.run(cross_section)[0]
            )

        output = []
        for dimension_line_region in dimension_line_regions:
            pred_texts, pred_polys, pred_confs = self.text_reader.run(
                dimension_line_region
            )
            output.append(
                {
                    "image": dimension_line_region,
                    "texts": pred_texts,
                    "polys": pred_polys,
                    "confs": pred_confs,
                }
            )

        return output


if __name__ == "__main__":
    section_detector_path = "section_detection_best.pt"
    dimension_line_detector_path = "dimension_line_detector.pt"
    dimension_extractor = DimensionExtractor(
        section_detector_path, dimension_line_detector_path
    )

    image = "../image_company/113/2d7573f20f7cb2b911135543119fc7d7_0.jpg"
    dimension_line_regions = dimension_extractor.run(image)
    print(f"Num dimension regions: {len(dimension_line_regions)}")
