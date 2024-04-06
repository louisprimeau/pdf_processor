# may be useful
# find /some/path -type f -name '.*' -execdir sh -c 'mv -i "$0" "./${0#./.}"' {} \;

import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

import layoutparser as lp
from PIL import Image
from pix2text import Pix2Text, merge_line_texts
from pdf2image import convert_from_path
from pypdf import PdfWriter, PdfReader

from text_processor import format_math_whitespace,\
replace_hyphen_spaces, replace_common_unicode
from util import *
from better_ocr import analyze_pdf

#matplotlib.use('TkAgg')

layout_model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
"""
text_model = Pix2Text(language='en',
               analyzer_config=dict(model_name='mfd',
                                    model_type='yolov7',
                                    model_fp='/home/louis/research/pdf_processor/models/mfd-yolov7-epoch224-20230613.pt'),
               formula_config=dict(model_fp='/home/louis/research/pdf_processor/models/p2t-mfr-20230702.pth'),
               device='cuda')"""

text_model = Pix2Text(model_backend='onnx', device='cuda')

image_cleaning_pipeline = [pad_paragraph_image]
text_cleaning_pipeline = [merge_line_texts, format_math_whitespace, replace_hyphen_spaces, replace_common_unicode]
write_directory = 'output/'
try:
    os.mkdir(write_directory)
except(FileExistsError):
    pass

root_path = "data/magnet/"
for pdf_path, directories, files in os.walk(root_path):
    for file in files:
        pdffile_name = os.fsdecode(file)
        if pdffile_name.endswith('.pdf'):
            print(pdffile_name)
            current_dir = os.path.join(write_directory, os.path.splitext(pdffile_name)[0])
            try:
                os.mkdir(current_dir)
            except(FileExistsError):
                print("this path already exists, skipping...")
                continue
            try:
                output_text, output_images, figure_captions = analyze_pdf(os.path.join(pdf_path, pdffile_name), layout_model, text_model, image_cleaning_pipeline, text_cleaning_pipeline)

                with open(os.path.join(current_dir, 'text.txt'), 'w') as f:
                    f.write(' '.join(output_text))

                with open(os.path.join(current_dir, 'captions.txt'), 'w') as f:
                    f.write('\n'.join(figure_captions))

                for i, im in enumerate(output_images):
                    Image.fromarray(im).save(os.path.join(current_dir, 'figure{}.png'.format(i)))

                shutil.copy(os.path.join(pdf_path, pdffile_name), os.path.join(current_dir, pdffile_name))
            except KeyboardInterrupt:
                shutil.rmtree(current_dir)
                with open('failed.txt', 'a') as f:
                    f.write(pdffile_name)
                sys.exit() # lol
            except:
                print("{} failed.".format(pdffile_name))
                shutil.rmtree(current_dir)
                with open('failed.txt', 'a') as f:
                    f.write(pdffile_name)

        else:
            continue

text_cleaning_pipeline = [merge_line_texts, format_math_whitespace, replace_hyphen_spaces, replace_common_unicode]

