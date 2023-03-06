import fitz  # PyMuPDF, imported as fitz for backward compatibility reasons
import os
import glob 
from tqdm import tqdm
import argparse
import cv2
from PIL import Image 
from pathlib import Path

def convert_pdf2image(file_path, outdir, img_max_size=None):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    doc = fitz.open(file_path)  # open document
    # dpi = 300  # choose desired dpi here
    zoom = 2  # zoom factor, standard: 72 dpi
    magnify = fitz.Matrix(zoom, zoom) 
    for idx, page in enumerate(doc):
        form_name = Path(file_path).parts[-2]
        print(form_name)
        out_with_form_name_dir = Path(outdir) / form_name
        if not out_with_form_name_dir.exists():
            out_with_form_name_dir.mkdir(exist_ok=True)
        pix = page.get_pixmap(matrix=magnify)  # render page to an image
        outpath = out_with_form_name_dir / Path(os.path.splitext(os.path.basename(file_path))[0] + "_" + str(idx) + ".png")
        pix.save(outpath)

        # img = Image.open(outpath )
        # img = img.convert("L")
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # img.save(outpath)
    # if status:
    #     print("OK")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()
    # pdf_dir = "/home/sds/hoanglv/FWD_Raw_Data/Form POS01"
    # outdir = "/home/sds/hoanglv/Projects/FWD/assets/test/test_image_transformer/template_aligner/pdf2image"
    

    pdf_paths = sorted(glob.glob(args.pdf_dir + "/*/*.pdf"))
    print(pdf_paths[:5])
    skip_dirs = ["BMH_IL",  "BMH_UL",   "POS01",   "POS02",   "POS03",   "POS04",   "POS05",   "POS06",   "POS08", "TXN"]

    for pdf_path in tqdm(pdf_paths):
        if pdf_path.split("/")[-2] in skip_dirs:
            continue
        convert_pdf2image(pdf_path, args.out_dir)