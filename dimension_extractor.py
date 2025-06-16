from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dimension_extractor_data_utils import (CrossSectionData,
                                            DimensionLineData, OCRData)
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

    def run(self, image: Union[str, np.ndarray]) -> List[CrossSectionData]:
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

        all_results = []
        for idx, (cross_section_img, bbox) in enumerate(
            zip(cross_sections, cross_section_bbxes)
        ):
            dim_lines, dim_line_bbxes = self.dimension_line_detector.run(
                cross_section_img
            )
            dim_line_results = []

            for dim_img, dim_bbx in zip(dim_lines, dim_line_bbxes):
                texts, polys, confs = self.text_reader.run(dim_img)
                ocr_entries = [
                    OCRData(text=t, poly=p, conf=c)
                    for t, p, c in zip(texts, polys, confs)
                ]
                dim_line_results.append(
                    DimensionLineData(
                        image=dim_img, bbx=dim_bbx, ocr_results=ocr_entries
                    )
                )

            section_data = CrossSectionData(
                image=cross_section_img,
                bbxes=bbox,
                dimension_lines=dim_line_results,
            )
            all_results.append(section_data)

        return all_results

    def visualise(
        self,
        image: np.ndarray,
        result: List[CrossSectionData],
        show_result: bool = False,
    ):

        annotated = image.copy()
        for section_idx, section in enumerate(result):
            # Draw section bbox
            poly = np.array(section.bbxes)
            print(f"poly: {poly}")
            x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"Section {section_idx}",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            for dim_idx, dim_line in enumerate(section.dimension_lines):
                # Just a visual cue; not exact position since dim_line is cropped from section
                dx, dy, dw, dh = dim_line.bbx
                cv2.rectangle(annotated, (dx, dy), (dx + dw, dy + dh), (0, 0, 255), 1)
                cv2.putText(
                    annotated,
                    f"DimLine {dim_idx}",
                    (dx, max(0, dy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

                for ocr in dim_line.ocr_results:
                    poly = np.array(ocr.poly).astype(np.int32)
                    cv2.polylines(
                        annotated, [poly], isClosed=True, color=(0, 255, 0), thickness=2
                    )
                    x_text, y_text = poly[0]
                    cv2.putText(
                        annotated,
                        ocr.text,
                        (x_text, max(0, y_text - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )

        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stacked = np.hstack((orig_rgb, annotated_rgb))

        if show_result:
            plt.figure(figsize=(16, 10))
            plt.imshow(stacked)
            plt.title("Original (Left) vs Annotated (Right)")
            plt.axis("off")
            plt.show()

        return stacked


if __name__ == "__main__":
    extractor = DimensionExtractor(
        "section_detection_best.pt", "dimension_line_detector.pt"
    )
    input_path = "../image_company/113/2d7573f20f7cb2b911135543119fc7d7_0.jpg"
    image = cv2.imread(input_path)
    result = extractor.run(image)
    # annotated = extractor.visualise(image, result)
    # cv2.imwrite("annotated.png", annotated)
    print(f"result: {result}")
