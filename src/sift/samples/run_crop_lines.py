from argparse import ArgumentParser
import cv2 
import glob
import os 
import time 
import tqdm

from src.modules.line_parser import TemplateBoxParser
from src.config.line_parser import TEMPLATE_BOXES
from src.utils.common import read_json, get_doc_id_with_page



"""
Crop per form (2 page)
"""

template_path_dict = {
    'pos01': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos01.json',
    'pos04': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos04.json',
    'pos02': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos02.json',
    'pos03': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos03.json',
    'pos08': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos08.json',
    'pos05': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos05.json',
    'pos06': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos06.json',
}

def identity_page(img_path, doc_id):
    page_type = ''
    if "_0.jpg" in img_path:
        page_number = 1
    elif "_1.jpg" in img_path:
        page_number = 2
    else:
        idx = int(float(img_path.split(".jpg")[0].split("_")[-1]))
        if idx % 2 == 0:
            page_number = 1
        else:
            page_number = 2

    doc_template_id = "{}_page_{}".format(doc_id, page_number)

    return doc_template_id  #page_1 / page_2

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--doc_id", help="pos01/pos02", default='pos01')
    args = parser.parse_args()

    line_parser = TemplateBoxParser()


    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    img_paths = glob.glob(args.img_dir + "/*")
    print("Len imgs: ", len(img_paths))

    template_info = read_json(template_path_dict[args.doc_id])

    crop_metadata = {}
    pages = ['page_']
    for page in pages:
        metadata = {"boxes": [], "box_types": []}

        crop_metadata[page] = metadata

    count = 0
    for idx, img_path in tqdm.tqdm(enumerate(img_paths)):
        aligned_images = cv2.imread(img_path)

        doc_template_id = get_doc_id_with_page(img_path, args.doc_id)
        # print(img_path, doc_template_id, aligned_images)

        
        cropped_images = line_parser.run(
            [aligned_images], 
            metadata=[{
                'boxes': template_info[doc_template_id]['fields']
            }]
        )
        
        count += len(cropped_images[0])
        for id_img, crop_img in enumerate(cropped_images[0]):
            out_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(img_path))[0] + "_" + str(id_img) + ".jpg")
            cv2.imwrite(out_path, crop_img)

    print("Total: ", count)