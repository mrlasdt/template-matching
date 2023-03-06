
from src.config.sift_based_aligner import config
from src.modules.sift_based_aligner import SIFTBasedAligner
from src.utils.common import read_json
from argparse import ArgumentParser
import os 
import cv2

num_pages_dict = {
    'pos01': 2,
    'pos04': 2
}

exception_files = {
    'pos01': ["SKM_458e Ag22101217490_0.png"]   #only page_1
}

template_path_dict = {
    'pos01': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos01.json',
    'pos04': '/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/data/json/pos04.json',
}

def reformat(config, doc_id):
    # template_infos = config['template_info']

    template_path = template_path_dict[doc_id]
    template_info = read_json(template_path)

    # num_page = num_pages_dict[doc_id]

    config['template_info'] = template_info
    return config




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir")
    parser.add_argument("--output")
    parser.add_argument("--doc_id", help="pos01/pos04", default='pos01')
    args = parser.parse_args()

    # make dir 
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    error_dir = args.output + "_error"
    if not os.path.exists(error_dir):
        os.makedirs(error_dir, exist_ok=True)

    doc_id = args.doc_id
    print("DOCID: ", doc_id)

    # load img paths
    img_paths = [args.img_dir]
    images = [cv2.imread(img_path) for img_path in img_paths]
    print("total samples: ", len(img_paths))
    
    #reformat config 
    config = reformat(config, doc_id=args.doc_id)
    
    # aligner init 
    aligner = SIFTBasedAligner(
        **config
    )

    
    
    metadata = [
        {
            'doc_type': '{}_1'.format(doc_id),
            'img_path': img_paths[0]
        }
    ]
    # metadata = [{'doc_type': 'edit_form_1_1' if "_0.jpg" in img_path else 'edit_form_1_2', 'img_path': img_path} for img_path in img_paths] 
    transformed_images = aligner.run(images, metadata)
    
    print(len(img_paths), len(transformed_images))


    error_count = 0
    for idx in range(len(transformed_images)):
        img_name = os.path.basename(img_paths[idx])
        img_outpath = os.path.join(args.output, img_name)
        img_out = transformed_images[idx]
        doc_type = metadata[idx]['doc_type']
        field_boxes = config['template_info'][doc_type]['fields']

        for bbox in field_boxes:
            x, y = bbox["position"]["left"], bbox["position"]["top"]
            w, h = bbox["size"]["width"], bbox["size"]["height"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(img_out, (x1,y1), (x2,y2), color=(255, 0, 0), thickness=2)

        if img_out is not None:
            print("Write to: ", img_outpath)
            cv2.imwrite(img_outpath, img_out)
        else:
            error_count += 1 
            print("Image None: ", img_paths[idx])
            cv2.imwrite(os.path.join(error_dir, img_name), images[idx])


    print("Num error cases: ", error_count)
