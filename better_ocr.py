import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt

import layoutparser as lp
from PIL import Image
from pix2text import Pix2Text, merge_line_texts
from pdf2image import convert_from_path
from pypdf import PdfWriter, PdfReader

import sklearn.mixture

from text_processor import format_math_whitespace,\
replace_hyphen_spaces, replace_common_unicode

#matplotlib.use('TkAgg')

# pad paragraph image with white margin, improves inference on edges
def pad_paragraph_image(paragraph_image):
    H, W, C = paragraph_image.shape
    pad_prop = 0.1
    pad_H, pad_W, pad_C = int(pad_prop * H), int(pad_prop * W), 0       
    return np.pad(paragraph_image,
                  ((pad_H, pad_H),(pad_W, pad_W),(pad_C, pad_C)),
                  'constant',
                  constant_values=((255, 255),(255, 255),(255, 255))
                  )

# decide reading order of paragraphs
def reorganize_layout(paragraph_layout):
    x_coords = np.array([t.block.x_1 for t in paragraph_layout._blocks])
    y_coords = np.array([t.block.y_1 for t in paragraph_layout._blocks]) / 10
    coords = np.vstack((x_coords, y_coords)).T
    idxs = np.arange(len(paragraph_layout._blocks))
    model = sklearn.mixture.BayesianGaussianMixture(n_components=2).fit(coords)
    membership = model.predict(coords).reshape(-1)
    clusters = []
    x_centroids = []
    for m in np.unique(membership):
        x_centroids.append(np.mean(x_coords[membership == m]))
        clusters.append(idxs[membership == m][np.argsort(y_coords[membership == m])])
    clusters = [clusters[i] for i in np.argsort(np.array(x_centroids))]
    reordered_idxs = np.hstack(clusters)
    print(coords[reordered_idxs, :])
    paragraph_layout._blocks = [paragraph_layout._blocks[i] for i in reordered_idxs]
    return paragraph_layout

def extract_text_from_images(paragraph_layout, page_image, image_cleaning_pipeline=[], text_cleaning_pipeline=[]):
    output_strings = []
    for block in paragraph_layout:
        if block.type=='Text':

            x1, y1 = int(block.block.x_1), int(block.block.y_1)
            x2, y2 = int(block.block.x_2), int(block.block.y_2)
            paragraph_image = page_image[y1:y2, x1:x2]

            for func in image_cleaning_pipeline: paragraph_image = func(paragraph_image)
            
            paragraph_text = text_model(Image.fromarray(paragraph_image))
            for func in text_cleaning_pipeline:
                paragraph_text = func(paragraph_text)
            
            output_strings.append(paragraph_text)
    return output_strings

def extract_image_from_image(image_layout, page_image, image_cleaning_pipeline=[]):
    output_images = []
    for block in image_layout:
        if block.type='Image':
            x1, y1 = int(block.block.x_1), int(block.block.y_1)
            x2, y2 = int(block.block.x_2), int(block.block.y_2)
            figure_image = page_image[y1:y2, x1:x2]
            for func in image_cleaning_pipeline:
                figure_image = func(figure_image)
            output_image.append(figure_image)
    return output_images

def associate_captions(page_paragraphs, paragraph_layout, page_images, image_layout):
    labels = []
    for paragraph in page_paragraphs:
        if any(good_string in paragraph[:7].lower() for good_string in ['table', 'figure', 'fig']):
            labels.append(True)
        else:
            regular.append(False)

    caption_coordinates = [(block.block.x_1, block.block.y_1) for is_caption, block in zip(labels, paragraph_layout) if is_caption]
    image_coordinates = [(block.block.x_1, block.block.y_1) for block in image_layout]

file_path = "data"
file_name = "PhysRevB.54.R3760.pdf"

page_images = convert_from_path(os.path.join(file_path, file_name), dpi=400)
page_images = [np.array(page_image) for page_image in page_images]

layout_model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

text_model = Pix2Text(language='en',
               analyzer_config=dict(model_name='mfd',
                                    model_type='yolov7',
                                    model_fp='/home/louis/research/llms/models/mfd-yolov7-epoch224-20230613.pt'),
               formula_config=dict(model_fp='/home/louis/research/llms/models/p2t-mfr-20230702.pth'),
               device='cuda')
#text_model = Pix2Text(model_backend='onnx', device='cuda')

image_cleaning_pipeline = [pad_paragraph_image]
text_cleaning_pipeline = [merge_line_texts, format_math_whitespace, replace_hyphen_spaces, replace_common_unicode]

output_text = []
output_images = []
figure_captions = []
for i, page_image in enumerate(page_images[1:]):
    print("page", i)
    paragraph_layout = layout_model.detect(page_image)
    figure_layout = [block for block in paragraph_layout._blocks if block.type=='Figure']
    paragraph_layout = [block for block in paragraph_layout._blocks if block.type=='Text']
    paragraph_layout = reorganize_layout(paragraph_layout)

    page_images = extract_image_from_image(figure_layout, page_image)
    page_text = extract_text_from_image(paragraph_layout, page_image, image_cleaning_pipeline, text_cleaning_pipeline)

    page_captions, output_text = associate_captions(page_images, page_text)

    output_text += page_text
    output_images += page_images
    figure_captions += page_captions

    


"""
reader = PdfReader(os.path.join(file_path, file_name))
writer = PdfWriter()

for page in reader.pages:
    new_page = writer.add_page(page)
    new_page.mediabox.upper_right = (
        new_page.mediabox.right,
        new_page.mediabox.top * 0.945,
    )
    write_path = os.path.join(file_path, "{0}_{2}{1}".format(*os.path.splitext(file_name), "CROPPED"))
    with open(write_path, "wb") as fp:
        writer.write(fp)
"""
