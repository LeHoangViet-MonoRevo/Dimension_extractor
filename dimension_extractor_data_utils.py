from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class OCRData:
    text: str
    poly: List[Tuple[int, int]]  # ✅ updated to accept polygon with N points
    conf: float


@dataclass
class DimensionLineData:
    image: np.ndarray
    bbx: Tuple[float, float, float, float]  # floats from detector
    ocr_results: List[OCRData]


@dataclass
class CrossSectionData:
    image: np.ndarray
    bbxes: List[
        float
    ]  # output from detection is a flat list (x1, y1, x2, y2, score, label)
    dimension_lines: List[DimensionLineData]
