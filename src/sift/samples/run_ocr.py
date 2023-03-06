from src.modules.ocr_module import OcrEngine
from  argparse import ArgumentParser
import os 
import glob
import cv2 
import time 
from src.utils.visualize import visualize_ocr_output

def main(img_dir, out_dir, device='cpu'):
    det_cfg = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/weights/yolox_s_8x8_300e_cocotext_1280.py" 
    det_ckpt = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/weights/best_bbox_mAP_epoch_100.pth"
    cls_cfg = "/home/sds/datnt/mmocr/logs/satrn_big_2022-04-25/satrn_big.py"
    cls_ckpt = "/home/sds/datnt/mmocr/logs/satrn_big_2022-04-25/best.pth"

    engine = OcrEngine(
        det_cfg,det_ckpt, cls_cfg, cls_ckpt,
        device=device
    )
    
    
    # img_dir = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/raw_images/POS01"
    # out_dir = "/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/outputs/visualize_ocr/POS01"
    
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--device", help='cpu / cuda:0')
    args = parser.parse_args()

    main(args.img_dir, args.out_dir, args.device)
    