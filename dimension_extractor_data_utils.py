from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class OCRData:
    text: str
    poly: Tuple[int, int, int, int]  # or Tuple[int, int, int, int] if always 4 points
    conf: float


@dataclass
class DimensionLineData:
    image: np.ndarray
    bbx: Tuple[int, int, int, int]
    ocr_results: List[OCRData]


@dataclass
class CrossSectionData:
    image: np.ndarray
    bbxes: List[Tuple[int, int, int, int]]
    dimension_lines: List[DimensionLineData]
