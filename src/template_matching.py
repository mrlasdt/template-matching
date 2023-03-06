#global package
import yaml
import numpy as np
from typing import overload, Union
from PIL import Image
# third party
from externals.ocr import OcrEngine, Line, words_to_lines

# local
from .sift.modules.sift_based_aligner import SIFTBasedAligner
from .sift.utils.common import read_json
from .dto import PageAligned, DocumentAligned


def calc_pct_overlapped_area(bboxes1: np.ndarray, bboxes2: np.ndarray):
    bboxes1 = bboxes1[:, :4]
    bboxes2 = bboxes2[:, :4]
    # assert True
    assert len(bboxes1.shape) == 2 and bboxes1.shape[1] == 4
    assert len(bboxes2.shape) == 2 and bboxes2.shape[1] == 4

    bboxes1 = bboxes1.copy()
    bboxes2 = bboxes2.copy()

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    boxBArea = np.tile(boxBArea, (1, len(bboxes1)))
    iou = interArea / boxBArea.T
    return iou


class TemplateMatcher:
    def __init__(self, ocr_engine: OcrEngine, setting_file='setting.yml'):
        with open(setting_file) as f:
            # use safe_load instead load
            self.setting = yaml.safe_load(f)
        self.aligner = SIFTBasedAligner(**self.setting['sift'])
        self.overlap_threshold = self.setting["overlap_threshold"]
        self.ocr_engine = ocr_engine

    def _reorder_words(self, boxes):
        arr_x1 = boxes[:, 0]
        return np.argsort(arr_x1)

    def _asign_textboxes_to_fields(self, boxes, fields, threshold=0.8) -> dict[str, list[list]]:
        '''
        return dict of field: sorted bboxes in field
        '''
        field_coords = []
        field_names = []
        for field in fields:
            field_coords.append(field['box'])
            field_names.append(field['label'])

        field_coords = np.array(field_coords)
        boxes = np.array(boxes)
        area_pct = calc_pct_overlapped_area(field_coords, boxes)

        results = dict()
        for row_score, field in zip(area_pct, field_names):
            inds = np.where(row_score > threshold)[0]
            field_word_boxes = boxes[inds]
            sorted_inds = inds[self._reorder_words(field_word_boxes)]
            results[field] = boxes[sorted_inds].tolist()
        return results

    def read_text_in_fields(self, aligned_image: np.ndarray, dfield_to_bboxes: dict[str, list[list]]) -> dict[str, list[Line]]:
        dfield_to_llines = dict()
        for field, bboxes in dfield_to_bboxes.items():
            llines = self.ocr_engine.read_page(aligned_image, bboxes)
            dfield_to_llines[field] = llines
        return dfield_to_llines

    def run_page(self, img: Union[str, np.ndarray, Image.Image],
                 page_template_path: str) -> tuple[np.ndarray, dict[str, list[Line]]]:
        img = self.ocr_engine.read_img(img)
        template_info = read_json(page_template_path)
        bboxes = self.ocr_engine.run_detect(img, return_raw=True)
        aligned_image, aligned_boxes = self.aligner.run(img, bboxes, template_info)
        if aligned_boxes is None:
            return (None, None)
        dfield_to_bboxes = self._asign_textboxes_to_fields(
            aligned_boxes, template_info['fields'], threshold=self.overlap_threshold)
        dfield_to_llines = self.read_text_in_fields(aligned_image, dfield_to_bboxes)
        return aligned_image, dfield_to_llines

    @overload
    def __call__(self, img: list[Union[str, np.ndarray, Image.Image]], template_path: str) -> DocumentAligned: ...

    @overload
    def __call__(self, img: Union[str, np.ndarray, Image.Image], template_path: str) -> PageAligned: ...

    def __call__(self, img, template_path):
        if isinstance(img, list):
            # return self.run_doc(img, template_path)
            raise NotImplementedError()
        aligned_image, dfield_to_llines = self.run_page(img, template_path)
        return PageAligned(original_img=img, aligned_image=aligned_image, dfield_to_llines=dfield_to_llines)


if __name__ == "__main__":
    # %%
    img_path = "/mnt/ssd500/hungbnt/Cello/data/PH/Sea9/Sea_9_1.jpg"
    ocr_engine = OcrEngine(device='cpu')
    # %%
    template_matcher = TemplateMatcher(
        ocr_engine, setting_file="/mnt/ssd500/hungbnt/Cello/externals/template_matching/setting.yml")
    template_path = "/mnt/ssd500/hungbnt/Cello/externals/template_matching/template/json/seaway_color.json"
    page_aligned = template_matcher(img_path, template_path)
    for field, llines in page_aligned.dfield_to_llines.items():
        print('*' * 100)
        print(field)
        print([line.text for line in llines])
