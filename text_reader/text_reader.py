from pathlib import Path
from typing import Union

import cv2
import numpy as np
from paddleocr import PaddleOCR


class PaddleTextReader:
    def __init__(
        self,
        use_doc_orientation_classify: bool = True,
        use_doc_unwarping: bool = True,
        use_textline_orientation: bool = True,
    ) -> None:

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
        )

    def run(self, image: Union[str, np.ndarray]):
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))

        assert image is not None, "Image is None, please re-check!!"
        pred = self.ocr.predict(image)[0]
        pred_texts = pred["rec_texts"]
        pred_polys = pred["rec_polys"]
        pred_confs = pred["rec_scores"]

        return pred_texts, pred_polys, pred_confs
