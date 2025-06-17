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

        This includes:
            1. Section Detection: Identifies cross-sectional views in the image.
            2. Dimension Line Detection: Locates dimension lines within each section.
            3. OCR: Recognises and extracts text (dimension values) from each dimension line region.

        Args:
            image (Union[str, np.ndarray]): Input image. Can be a file path (str) or a Numpy array (BGR image).

        Returns:
            List[CrossSectionData]: A list of structured results per cross-section. Each item contains:
                - The cross-sectional image.
                - Its bounding box within the original image.
                - A list of dimension line regions, each with its bounding box and recognised OCR data.
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        assert image is not None, "Image is None, please re-check!!"

        cross_sections, cross_section_bbxes = self.section_detector.run(image)
        all_results = []

        for cross_section_img, bbox in zip(cross_sections, cross_section_bbxes):
            dim_lines, dim_line_bbxes = self.dimension_line_detector.run(
                cross_section_img
            )
            dim_line_results = []

            for dim_img, dim_bbx in zip(dim_lines, dim_line_bbxes):
                texts, polys, confs = self.text_reader.run(dim_img)

                ocr_entries = [
                    OCRData(
                        text=t,
                        poly=[
                            tuple(pt) for pt in p
                        ],  # Ensure poly is List[Tuple[int, int]]
                        conf=c,
                    )
                    for t, p, c in zip(texts, polys, confs)
                ]

                dim_line_results.append(
                    DimensionLineData(
                        image=dim_img,
                        bbx=tuple(dim_bbx),  # Cast to tuple if not already
                        ocr_results=ocr_entries,
                    )
                )

            section_data = CrossSectionData(
                image=cross_section_img,
                bbxes=tuple(bbox),  # Expected to be (x1, y1, x2, y2)
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
        """
        Visualise the dimension extraction results on the input image.

        Overlays detected sections, dimension lines, and OCR results on the original image.
        Bounding boxes and text annotations are drawn for clarity.

        Args:
            image (np.ndarray): The original input image in BGR format.
            result (List[CrossSectionData]): The output of the `run` method.
            show_result (bool, optional): If True, displays the side-by-side visualisation using matplotlib.

        Returns:
            np.ndarray: A side-by-side RGB image showing the original and annotated results.
        """
        annotated = image.copy()

        for section_idx, section in enumerate(result):
            # Section bounding box in original image
            sx1, sy1, sx2, sy2 = map(int, section.bbxes[:4])
            cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"Section {section_idx}",
                (sx1, max(0, sy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            for dim_idx, dim_line in enumerate(section.dimension_lines):
                # DimLine bbox is relative to section
                dx1, dy1, dx2, dy2 = map(int, dim_line.bbx[:4])
                abs_dx1, abs_dy1 = sx1 + dx1, sy1 + dy1
                abs_dx2, abs_dy2 = sx1 + dx2, sy1 + dy2

                cv2.rectangle(
                    annotated, (abs_dx1, abs_dy1), (abs_dx2, abs_dy2), (0, 0, 255), 1
                )
                cv2.putText(
                    annotated,
                    f"DimLine {dim_idx}",
                    (abs_dx1, max(0, abs_dy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

                for ocr in dim_line.ocr_results:
                    # Each OCR poly is relative to dimension line
                    adjusted_poly = np.array(
                        [[pt[0] + dx1 + sx1, pt[1] + dy1 + sy1] for pt in ocr.poly],
                        dtype=np.int32,
                    )

                    cv2.polylines(
                        annotated,
                        [adjusted_poly],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    x_text, y_text = adjusted_poly[0]
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
    import glob
    import os

    from tqdm import tqdm

    extractor = DimensionExtractor(
        "section_detection_best.pt", "dimension_line_detector.pt"
    )

    img_dirs = glob.glob("kubo_dataset/*.png")
    for img_dir in tqdm(img_dirs):
        image = cv2.imread(img_dir)
        result = extractor.run(img_dir)
        annotated = extractor.visualise(image, result, False)
        cv2.imwrite(
            os.path.join("./kubo_dataset_pred/", os.path.basename(img_dir)), annotated
        )
