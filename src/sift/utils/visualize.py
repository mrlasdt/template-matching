import os
import glob
import math
import json
import random
from sys import prefix

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont



def visualize_ocr_output(inputs,
                         image,
                         vis_dir,
                         prefix_name='img_visualize',
                         font_path='./times.ttf',
                         is_vis_kie=False
                         ):
    
    """
    Visualize ocr output (box + text) and kie output (optional)
    params:
        inputs (dict/list[list,list]): keys {ocr, kie}
            - ocr value format: list of item (polygon box, label, prob/kie_label)
            - kie value format: not implemented
        image (np.ndarray): BGR image
        vis_dir (str): save directory 
        name_vis_image (str): prefix name of save image
        font_path (str): path of font
        is_vis_kie (bool): if True, third item is kie label
    return: 
    
    """
    # table_reconstruct_result = ehr_res['table_reconstruct_result']
    # assert 'ocr' in inputs, "not found 'ocr' field in inputs"

    # identity input format 
    if len(inputs) == 2 and isinstance(inputs[1][0], str):
        ocr_result = [
            [box if isinstance(box[0], list) else box2poly(box), text, 1.0]
            for box, text in zip(inputs[0], inputs[1])
        ]
    else:
        ocr_result = inputs['ocr']

    if not os.path.exists(vis_dir):
        print("Creating {} dir".format(vis_dir))
        os.makedirs(vis_dir)

        
    img_visual = draw_ocr_box_txt(image=image,
                                  annos=ocr_result,
                                  font_path=font_path,
                                  table_boxes=None,
                                  cell_boxes=None,
                                  para_boxes=None,
                                  is_vis_kie=is_vis_kie
                                  )

    paths = sorted(glob.glob(vis_dir + "/" + prefix_name + "*"),
                   key=lambda path: int(path.split(".jpg")[0].split("_")[-1]))
    if len(paths) == 0:
        idx_name = '1'
    else:
        idx_name = str(int(paths[-1].split(".jpg")[0].split("_")[-1]) + 1)
    cv2.imwrite(os.path.join(vis_dir, prefix_name +
                "_" + idx_name + ".jpg"), img_visual)


def export_to_csv(table_reconstruct_text, vis_dir, csv_name='table_text_reconstruct'):
    paths = sorted(glob.glob(vis_dir + "/" + csv_name + "*"),
                   key=lambda path: int(path.split(".csv")[0].split("_")[-1]))
    if len(paths) == 0:
        idx_name = '1'
    else:
        idx_name = str(int(paths[-1].split(".csv")[0].split("_")[-1]) + 1)
    df = pd.DataFrame(table_reconstruct_text)
    df.to_csv(os.path.join(vis_dir, csv_name +
              "_" + idx_name + ".csv"), index=False)


def save_json(data, vis_dir, json_name='ehr_result'):
    """ save dictionary to json file
    Args:
        data (dict): 
        vis_dir (str):  path to save json 
        json_name (str, optional): json name. Defaults to 'ehr_result'.
    """
    paths = sorted(glob.glob(vis_dir + "/" + json_name + "*"),
                   key=lambda path: int(path.split(".json")[0].split("_")[-1]))
    if len(paths) == 0:
        idx_name = '1'
    else:
        idx_name = str(int(paths[-1].split(".json")[0].split("_")[-1]) + 1)
    outpath = os.path.join(vis_dir, json_name + "_" + idx_name + ".json")
    with open(outpath, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)


def draw_ocr_box_txt(image,
                     annos,
                     scores=None,
                     drop_score=0.5,
                     font_path="test/fonts/latin.ttf",
                     table_boxes=None,
                     cell_boxes=None,
                     para_boxes=None,
                     is_vis_kie=False):
    """
    Args:
        image (np.ndarray / PIL): BGR image or PIL image 
        annos (list): (box, text, label/prob)
        scores (list, optional): probality. Defaults to None.
        drop_score (float, optional): . Defaults to 0.5.
        font_path (str, optional): Path of font. Defaults to "test/fonts/latin.ttf".
    Returns:
        np.ndarray: BGR image
    """
    
    if is_vis_kie:
        kie_labels = set([item[2] for item in annos])
        colors = {
            label: (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
            for label in kie_labels
        }
    
    color_vis = {
        'table': (255, 192, 70),
        'cell': (218, 66, 15),
        'paragraph': (0, 187, 148)
    }

    random.seed(0)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    

    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt, meta_data) in enumerate(annos):
        if scores is not None and scores[idx] < drop_score:
            continue
        
        if is_vis_kie:
            color = colors[meta_data]
        else:
            color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.6), 20)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)

    if table_boxes is not None:
        img_left = draw_rectangle_pil(
            img_left,
            table_boxes,
            color=color_vis['table'],
            width=6,
            label='table'
        )
    if cell_boxes is not None:
        img_left = draw_rectangle_pil(
            img_left,
            cell_boxes,
            color=color_vis['cell'],
            width=5,
            label='cell'
        )
    if para_boxes is not None:
        img_left = draw_rectangle_pil(
            img_left,
            para_boxes,
            color=color_vis['paragraph'],
            width=2,
            label='para'
        )

    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    img_show = cv2.cvtColor(np.array(img_show), cv2.COLOR_RGB2BGR)
    return img_show


def draw_rectangle_pil(pil_image,
                       boxes,
                       color,
                       width=1,
                       label=None,
                       font_path="test/fonts/latin.ttf"
                       ):
    """
    Args:
        pil_image ([type]): [description]
        boxes (list): list of [xmin, ymim, xmax, ymax]
        color (list): list of (R, G, B)
    """
    drawer = ImageDraw.Draw(pil_image)
    color = tuple((int(color[0]), int(color[1]), int(color[2])))
    for box in boxes:
        drawer.rectangle([(int(box[0]), int(box[1])), (int(
            box[2]), int(box[3]))], outline=color, width=width)
        
        if label:
            font_size = 35
            font = ImageFont.truetype(font_path, size=32, encoding="utf-8")
            drawer.text([int(box[0]) + 5, int(box[1]) - font_size - 5],
                        label, fill=color, font=font)
    return pil_image

def box2poly(box):
    """
    Convert box format to polygon format: xyxy to xyxyxyxy
    """
    xmin, ymin, xmax, ymax = box 
    poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    return poly