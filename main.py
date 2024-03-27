import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt

import layoutparser as lp
from PIL import Image
from pix2text import Pix2Text, merge_line_texts
from pdf2image import convert_from_path
from pypdf import PdfWriter, PdfReader


from text_processor import format_math_whitespace,\
replace_hyphen_spaces, replace_common_unicode
from util import *
from better_ocr import analyze_pdf

matplotlib.use('TkAgg')


file_path = "data/published_papers_2"
#file_name = "PhysRevB.54.R3760.pdf"
file_name = "Ghosh_2011_J._Phys.__Condens._Matter_23_164203-SrHo2O4.pdf"

layout_model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

text_model = Pix2Text(language='en',
               analyzer_config=dict(model_name='mfd',
                                    model_type='yolov7',
                                    model_fp='/home/louis/research/pdf_processor/models/mfd-yolov7-epoch224-20230613.pt'),
               formula_config=dict(model_fp='/home/louis/research/pdf_processor/models/p2t-mfr-20230702.pth'),
               device='cuda')
#text_model = Pix2Text(model_backend='onnx', device='cuda')
image_cleaning_pipeline = [pad_paragraph_image]
text_cleaning_pipeline = [merge_line_texts, format_math_whitespace, replace_hyphen_spaces, replace_common_unicode]


output_text, output_images, figure_captions = analyze_pdf(os.path.join(file_path, file_name), layout_model, text_model, image_cleaning_pipeline, text_cleaning_pipeline)
