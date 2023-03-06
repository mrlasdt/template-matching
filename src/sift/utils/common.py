import os 
import json 


def get_doc_id_with_page(img_path, doc_id):
    if "_0." in img_path:
        doc_page = '{}_page_1'.format(doc_id)
    elif "_1." in img_path:
        doc_page = '{}_page_2'.format(doc_id)
    else:
        idx = int(os.path.splitext(os.path.basename(img_path))[0].split("_")[-1])
        # idx = int(float(img_path.split(".jpg")[0].split("_")[-1]))
        if idx % 2 == 0:
            doc_page = '{}_page_1'.format(doc_id)
        else:
            doc_page = '{}_page_2'.format(doc_id)

    return doc_page

def read_json(json_path):
    with open(json_path, 'r', encoding='utf8') as f: 
        data = json.load(f)
    return data 

