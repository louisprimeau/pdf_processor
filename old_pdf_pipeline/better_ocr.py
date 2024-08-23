

import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt

import layoutparser as lp
from PIL import Image
from pix2text import Pix2Text, merge_line_texts
from pdf2image import convert_from_path

from util import *

def analyze_pdf(path, layout_model, text_model,
                image_cleaning_pipeline=[], text_cleaning_pipeline=[]):

    # convert pdf to image arrays
    page_images = [np.array(page_image) for page_image in convert_from_path(path, dpi=500)]

    # for each page
    output_text, output_images, figure_captions, output_tables = [], [], [], []
    for i, page_image in enumerate(page_images[0:]):

        # get paragraph, figure layout
        raw_layout = layout_model.detect(page_image)

        figure_layout, paragraph_layout, table_layout = [], [], []
        for block in raw_layout:
            if block.type=='Figure': figure_layout.append(block)
            elif block.type=='Text': paragraph_layout.append(block)
            elif block.type=='Table': table_layout.append(block)

        # decide the column layout
        paragraph_layout = reorganize_layout(paragraph_layout)

        # associate captions with the detected figures
        caption_layout, paragraph_layout = associate_captions(paragraph_layout, figure_layout)

        # extract the figures
        figure_images = extract_image_from_image(figure_layout, page_image)

        # run text extraction on paragraph regions
        page_text = extract_text_from_image(text_model,
                                            paragraph_layout,
                                            page_image,
                                            image_cleaning_pipeline,
                                            text_cleaning_pipeline)

        # run text extraction on caption regions
        page_captions = extract_text_from_image(text_model,
                                                caption_layout,
                                                page_image,
                                                image_cleaning_pipeline,
                                                text_cleaning_pipeline)


        # run text extraction on table regions
        tables = extract_text_from_image(text_model,
                                         table_layout,
                                         page_image,
                                         image_cleaning_pipeline,
                                         )
        
        output_text += page_text
        output_images += figure_images
        figure_captions += page_captions
        output_tables += tables

    return output_text, output_images, figure_captions, output_tables

def extract_text(path, layout_model, text_model, image_cleaning_pipeline=[], text_cleaning_pipeline=[]):

    # convert pdf to image arrays
    page_images = [np.array(page_image) for page_image in convert_from_path(path, dpi=500)]

    # for each page
    output_text  = []
    for i, page_image in enumerate(page_images[0:]):

        # get paragraph, figure layout
        raw_layout = layout_model.detect(page_image)

        paragraph_layout = [block for block in raw_layout if block.type=='Text']
        
        # run text extraction on paragraph regions
        page_text = extract_text_from_image(text_model,
                                            paragraph_layout,
                                            page_image,
                                            image_cleaning_pipeline,
                                            text_cleaning_pipeline)
        
        output_text += page_text

    return output_text
