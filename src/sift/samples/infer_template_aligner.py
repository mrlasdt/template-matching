
from src.config.sift_based_aligner import config
from src.modules.sift_based_aligner import SIFTBasedAligner
from src.utils.common import read_json, get_doc_id_with_page
from argparse import ArgumentParser
import os 
import cv2
import time 


num_pages_dict = {
    'pos01': 1,
    'pos04': 2
}

exception_files = {
    'pos01': ["SKM_458e Ag22101217490_0.png"]   #only page_1
}

template_path_dict = {
    'pos01': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos01.json',
    'pos04': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos04.json',
    'pos02': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos02.json',
    'pos03': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos03.json',
    'pos08': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos08.json',
    'pos05': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos05.json',
    'pos06': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos06.json',
    'cmnd' : '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/CMND/json/cmnd.json'
}

def reformat(config, doc_id):
    template_path = template_path_dict[doc_id]
    template_info = read_json(template_path)
    config['template_info'] = template_info
    return config

# def get_doc_id_with_page(img_path):
#     if "_0.jpg" in img_path:
#         doc_page = '{}_page_1'.format(doc_id)
#     elif "_1.jpg" in img_path:
#         doc_page = '{}_page_2'.format(doc_id)
#     else:
#         idx = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
#         # idx = int(float(img_path.split(".jpg")[0].split("_")[-1]))
#         if idx % 2 == 0:
#             doc_page = '{}_page_1'.format(doc_id)
#         else:
#             doc_page = '{}_page_2'.format(doc_id)

#     return doc_page

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir")
    parser.add_argument("--output")
    parser.add_argument("--doc_id", help="pos01/pos02", default='pos01')
    parser.add_argument("--show", action='store_true')
    
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    error_dir = args.output + "_error"
    if not os.path.exists(error_dir):
        os.makedirs(error_dir, exist_ok=True)

    doc_id = args.doc_id
    print("DOC_ID: ", doc_id)

    img_paths = [os.path.join(args.img_dir, filename) for filename in os.listdir(args.img_dir)]
    
    print("total samples: ", len(img_paths))
    
    
    # init aligner
    
    config = reformat(config, doc_id=doc_id)
    # print(config)
    aligner = SIFTBasedAligner(
        **config
    )

    
    # create metadata for images
    metadata = []
    for img_path in img_paths:
        
        doc_with_page = get_doc_id_with_page(img_path, doc_id=doc_id)
        metadata.append(
            {
                'doc_type': doc_with_page,
                'img_path': img_path
            }
        )
    
    images = [cv2.imread(img_path) for img_path in img_paths]
    t1 = time.time()
    transformed_images = aligner.run(images, metadata)
    print("Total time align: ", time.time() - t1)
    
    print(len(img_paths), len(transformed_images))


    error_count = 0
    for idx in range(len(transformed_images)):
        img_name = os.path.basename(img_paths[idx])
        img_outpath = os.path.join(args.output, img_name)
        img_out = transformed_images[idx]
        doc_type = metadata[idx]['doc_type']
        

        if args.show:
            field_boxes = config['template_info'][doc_type]['fields']
            for bbox in field_boxes:
                x, y = bbox["position"]["left"], bbox["position"]["top"]
                w, h = bbox["size"]["width"], bbox["size"]["height"]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(img_out, (x1,y1), (x2,y2), color=(255, 0, 0), thickness=2)
        
        # print("Writing image...")
        if img_out is not None:
            # print("Write to: ", img_outpath)
            cv2.imwrite(img_outpath, img_out)
        else:
            error_count += 1 
            # print("Image None: ", img_paths[idx])
            cv2.imwrite(os.path.join(error_dir, img_name), images[idx])


    print("Done. Num error cases: ", error_count)
