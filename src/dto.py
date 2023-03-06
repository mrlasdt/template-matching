from typing import Union
import numpy as np
from PIL import Image
from externals import ocr


class PageAligned:
    def __init__(self, original_img: Union[str, np.ndarray, Image.Image],
                 aligned_image: np.ndarray, dfield_to_llines: dict[str: list[ocr.Line]]) -> None:
        self.original_img = original_img
        self.aligned_image = aligned_image
        self.dfield_to_llines = dfield_to_llines


class DocumentAligned:
    def __init__(self, lpages_aligned: list[PageAligned]) -> None:
        self.lpages_aligned = lpages_aligned
