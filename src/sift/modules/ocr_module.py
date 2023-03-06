# temp #for debug
import glob
import os
from mmdet.apis import (inference_detector,
                        init_detector)
from mmocr.apis import init_detector as init_classifier
from mmocr.apis.inference import model_inference
import cv2
import numpy as np
# from src.tools.utils import *

import time 
from src.utils.visualize import visualize_ocr_output


def clip_box(x1, y1, x2, y2, w, h):
    x1 = int(float(min(max(0, x1), w)))
    x2 = int(float(min(max(0, x2), w)))
    y1 = int(float(min(max(0, y1), h)))
    y2 = int(float(min(max(0, y2), h)))
    return (x1, y1, x2, y2)

def get_crop_img_and_bbox(img, bbox, extend: bool = False):
    """
    img : numpy array img
    bbox : should be xyxy format
    """
    if len(bbox) == 5:
        left, top, right, bottom, _conf = bbox
    elif len(bbox) == 4:
        left, top, right, bottom = bbox
    left, top, right, bottom = clip_box(
        left, top, right, bottom, img.shape[1], img.shape[0]
    )
    # assert (bottom - top) * (right - left) > 0, "bbox is invalid"
    crop_img = img[top:bottom, left:right]
    return crop_img, (left, top, right, bottom)

class YoloX():
    def __init__(self, config, checkpoint, device='cpu'):
        self.model = init_detector(config, checkpoint, device=device)

    def inference(self, img=None):
        t1 =time.time()
        output = inference_detector(self.model, img)
        print("Time det: ", time.time() - t1)
        return output 


class Classifier_SATRN:
    def __init__(self, config, checkpoint, device='cpu'):
        self.model = init_classifier(config, checkpoint,device)

    def inference(self, numpy_image):
        t1= time.time()
        result = model_inference(self.model, numpy_image, batch_mode=True)
        preds_str = [r["text"] for r in result]
        confidence = [r["score"] for r in result]

        print("Time reg: ", time.time() - t1)
        return preds_str, confidence


class OcrEngine:
    def __init__(self, det_cfg, det_ckpt, cls_cfg, cls_ckpt, device='cpu'):
        self.det = YoloX(det_cfg, det_ckpt, device)
        self.cls = Classifier_SATRN(cls_cfg, cls_ckpt, device)

    def run_image(self, img):
        
        pred_det = self.det.inference(img)


        pred_det = pred_det[0]  # batch_size=1

        pred_det = sorted(pred_det, key = lambda box : [box[1], box[0]])
        bboxes = np.vstack(pred_det)

        lbboxes = []
        lcropped_img = []
        assert len(bboxes) != 0, f'No bbox found in {img_path}, skipped'
        for bbox in bboxes:
            try:
                crop_img, bbox_ = get_crop_img_and_bbox(img, bbox, extend=True)
                lbboxes.append(bbox_)
                lcropped_img.append(crop_img)
            except AssertionError:
                print(f'[ERROR]: Skipping invalid bbox {bbox} in ', img_path)
        lwords, _ = self.cls.inference(lcropped_img)
        return lbboxes, lwords

def visualize(image, boxes, color=(0, 255, 0)):
    
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]))


if __name__ == "__main__":
    det_cfg = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/weights/yolox_s_8x8_300e_cocotext_1280.py" 
    det_ckpt = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/weights/best_bbox_mAP_epoch_100.pth"
    cls_cfg = "/home/sds/datnt/mmocr/logs/satrn_big_2022-04-25/satrn_big.py"
    cls_ckpt = "/home/sds/datnt/mmocr/logs/satrn_big_2022-04-25/best.pth"

    engine = OcrEngine(
        det_cfg,det_ckpt, cls_cfg, cls_ckpt,
        device='cpu'
    )

    # img_path = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/assest/form_1_edit_personal_info/Scan47_0.jpg"
    
    
    img_dir = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_images/POS01"
    out_dir = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/outputs/visualize_ocr/POS01"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_paths = glob.glob(img_dir + "/*")
    for img_path in img_paths:
        
        img = cv2.imread(img_path)
        t1 = time.time()
        res = engine.run_image(img)
        
        visualize_ocr_output(
            res,
            img,
            vis_dir=out_dir,
            prefix_name=os.path.splitext(os.path.basename(img_path))[0],
            font_path='./assest/visualize/times.ttf',
            is_vis_kie=False
        )

