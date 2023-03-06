# %% test template matching
from externals.template_matching import TemplateMatcher
from externals.ocr import OcrEngine

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
    print(llines)
