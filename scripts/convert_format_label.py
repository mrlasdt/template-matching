import os
import json 
from xml.etree import ElementTree as ET
import argparse

"""

format config :

config:
    pos01.json
    pos02.json 


"""


mapping = {
    "anchor": "anchors",
    "field": "fields"
}


def read_xml(xml_path):
    """_summary_

    Args:
        xml_path (_type_): _description_

    Returns:
        (list): [
            {
                'filename': xxx.jpg,
                'metadata': {
                    'label_1': [[xmin, ymin, xmax, ymax], ...],
                    'label_2': ...
                }
            }
        ]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for item in root.iter("image"):
        image_info = {}
        filename = item.attrib['name']  # img_path
        
        metadata = {}

        anchor_boxes = []
        fields_boxes = []
        for obj in item.iter('box'):
            label = obj.attrib['label']

            for att in obj.iter('attribute'):
                text = att.text
                label = label + "_" + text
            if label in mapping:
                label = mapping[label]

            xmin = int(float(obj.attrib['xtl']))
            ymin = int(float(obj.attrib['ytl']))
            xmax = int(float(obj.attrib['xbr']))
            ymax = int(float(obj.attrib['ybr']))
            box = [xmin, ymin, xmax, ymax]

            if label == 'anchors':
                anchor_boxes.append(box)
            else:
                fields_boxes.append(
                    {
                        'box': box,
                        'label': label
                    }
                )


            # box_info = {
            #     'box': box,
            #     'label': 
            # }
            
            # if label not in metadata:
            #     metadata[label] = [box]
            # else:
            #     metadata[label].append(box)
            # labels.append(label)

        metadata = {
            'anchors': anchor_boxes,
            'fields': fields_boxes
        }

        image_info['filename'] = filename
        image_info['metadata'] = metadata

        data.append(image_info)
    return data

def write_json(data, out_path):
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(data, f)



class Converter:
    def __init__(self):
        pass 
    
    @staticmethod
    def cvat2json(in_path, out_path, template_dir=None):
        """convert cvat output format to json format 

        image name in cvat should have page number, example: filename_page_1.jpg

        json format for one document type  belike:
        {
            "page_1": {
                "anchors": [[xmin, ymin, xmax, ymax], ...]
                "fields": ...
            },
            "page_2": {
                "anchors": [[xmin, ymin, xmax, ymax], ...]
                "fields": ...
            }
        }    
        --> pos01.json

        Args:
            in_path (str): path of .xml file
            out_path (_type_): path of output, .json 
        """
        assert ".xml" in in_path and ".json" in out_path

        outdir = os.path.dirname(out_path)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        out_data = {}

        data = read_xml(in_path)
        """ data format 
        [
            {
                'filename': xxx.jpg,
                'metadata': {
                    'label_1': [[xmin, ymin, xmax, ymax], ...],
                    'label_2': ...
                }
            }
        ]
        """
        doc_id = os.path.splitext(os.path.basename(out_path))[0]
        
        for item in data:
            filename = item['filename']
            metadata = item['metadata']
            page_number = int(float(os.path.splitext(filename)[0].split("_")[-1]))
            doc_id_with_page  = "{}_page_{}".format(doc_id, page_number+1)
            metadata['image_path'] = os.path.join(template_dir, filename)
            out_data[doc_id_with_page] = metadata
            

        write_json(out_data, out_path)
        print("Done!!!")


    def cvat2json_with_fields(in_path, out_path, template_dir=None):
        """convert cvat output format to json format 

        image name in cvat should have page number, example: filename_page_1.jpg

        json format for one document type  belike:
        {
            "page_1": {
                "anchors": [[xmin, ymin, xmax, ymax], ...]
                "fields": ...
            },
            "page_2": {
                "anchors": [[xmin, ymin, xmax, ymax], ...]
                "fields": ...
            }
        }    
        --> pos01.json

        Args:
            in_path (str): path of .xml file
            out_path (_type_): path of output, .json 
        """
        assert ".xml" in in_path and ".json" in out_path

        outdir = os.path.dirname(out_path)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        out_data = {}

        data = read_xml(in_path)
        """ data format 
        [
            {
                'filename': xxx.jpg,
                'metadata': {
                    'label_1': [[xmin, ymin, xmax, ymax], ...],
                    'label_2': ...
                }
            }
        ]
        """
        doc_id = os.path.splitext(os.path.basename(out_path))[0]
        
        for item in data:
            filename = item['filename']
            metadata = item['metadata']
            page_number = int(float(os.path.splitext(filename)[0].split("_")[-1]))
            doc_id_with_page  = "{}_page_{}".format(doc_id, page_number+1)
            metadata['image_path'] = os.path.join(template_dir, filename)
            out_data[doc_id_with_page] = metadata
            

        write_json(out_data, out_path)
        print("Done!!!")

            

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True, help="in path")
    parser.add_argument("--out_path", type=str, required=True, help="out path")
    parser.add_argument("--template_dir", type=str, required=True)

    args = parser.parse_args()
    converter = Converter()
    converter.cvat2json(
        in_path=args.in_path,
        out_path=args.out_path,
        template_dir=args.template_dir
    )