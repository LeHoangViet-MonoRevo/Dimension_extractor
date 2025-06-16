from typing import Union

import cv2
import matplotlib.pyplot as plt
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
            dim_lines, _ = self.dimension_line_detector.run(cross_section)
            dimension_line_regions.extend(dim_lines)

        ocr_results = []
        for region in dimension_line_regions:
            pred_texts, pred_polys, pred_confs = self.text_reader.run(region)
            ocr_results.append(
                {
                    "image": region,
                    "texts": pred_texts,
                    "polys": pred_polys,
                    "confs": pred_confs,
                }
            )

        return {
            "image": image,
            "cross_section_bboxes": cross_section_bbxes,
            "dimension_line_regions": dimension_line_regions,
            "ocr_results": ocr_results,
        }

    def visualise(self, result: dict, show_result: bool = False):
        """
        Visualise the extraction result:
        - Original image on left
        - Annotated image on right with section boxes, OCR polygons, and labels

        Args:
            result (dict): Output of self.run()
        """
        image = result["image"]
        annotated = image.copy()

        # Draw section bounding boxes
        for box in result["cross_section_bboxes"]:
            x1, y1, x2, y2, *_ = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                "Section",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        # Draw OCR results
        for ocr in result["ocr_results"]:
            for poly, text in zip(ocr["polys"], ocr["texts"]):
                poly = np.array(poly).astype(np.int32)
                cv2.polylines(
                    annotated, [poly], isClosed=True, color=(0, 255, 0), thickness=2
                )
                x, y = poly[0]
                cv2.putText(
                    annotated,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Show side by side
        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stacked = np.hstack((orig_rgb, annotated_rgb))

        if show_result:
            plt.figure(figsize=(16, 10))
            plt.imshow(stacked)
            plt.title("Original (Left) vs Annotated (Right)")
            plt.axis("off")
            plt.show()

        return stacked  # Optional: return annotated image for saving


if __name__ == "__main__":
    extractor = DimensionExtractor(
        "section_detection_best.pt", "dimension_line_detector.pt"
    )
    result = extractor.run(
        "../image_company/113/2d7573f20f7cb2b911135543119fc7d7_0.jpg"
    )
    annotated = extractor.visualise(result)
    cv2.imwrite("annotated.png", annotated)
