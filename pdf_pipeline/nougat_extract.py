# Shamelessly taken from jupyter notebook demo for nougat ocr
import sys
import os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch
from huggingface_hub import hf_hub_download
from typing import Optional, List
import io
import fitz
from pathlib import Path
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
from PIL import Image, ImageDraw
from gmft import CroppedTable, TableDetector, AutoTableFormatter
from gmft.pdf_bindings import PyPDFium2Document
from util import RunningVarTorch, StoppingCriteriaScores

processor = AutoProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
table_detector = TableDetector()
table_formatter = AutoTableFormatter()
    
def rasterize_paper(pdf: Path, outpath: Optional[Path] = None, dpi: int = 96, return_pil=False, pages=None) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
        dpi (int, optional): The output DPI. Defaults to 96.
        return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
    """

    pillow_images = []
    if outpath is None:
        return_pil = True
    try:
        if isinstance(pdf, (str, Path)):
            pdf = fitz.open(pdf)
        if pages is None:
            pages = range(len(pdf))
        for i in pages:
            page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
            if return_pil:
                pillow_images.append(io.BytesIO(page_bytes))
            else:
                with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                    f.write(page_bytes)
    except Exception:
        pass
    if return_pil:
        return pillow_images

write_directory = '/home/louis/data/processed_data/text_data_2/'
try:
    os.mkdir(write_directory)
except(FileExistsError):
    pass

# walk through all subdirectories of root_path
root_path = "/home/louis/data/raw_data/pdf/10.1103"
for pdf_path, directories, files in os.walk(root_path):
    for file in files:
        pdffile_name = os.fsdecode(file)
        if pdffile_name.endswith('.pdf'):
            print(pdffile_name)
            current_dir = os.path.join(write_directory, os.path.splitext(pdffile_name)[0])

            #check if pdf has already been processed
            try:
                os.mkdir(current_dir)
            except(FileExistsError):
                print("this path already exists, skipping...")
                continue

            # run extraction
            try:  
                this_pdf_text = []
                doc = PyPDFium2Document(os.path.join(pdf_path, pdffile_name))
                
                images = rasterize_paper(pdf=os.path.join(pdf_path, pdffile_name), return_pil=True)
                for pymupdfpage, image_file in zip(doc, images):
                    tables = table_detector.extract(pymupdfpage)
                    image = Image.open(image_file)
                    if len(tables) > 0:
                        draw = ImageDraw.Draw(image)
                        for table in tables:
                            draw.rectangle(list(np.array(table.bbox) * 96 / 72), fill=(255, 255, 255))

                    pixel_values = processor(images=image, return_tensors="pt").pixel_values
                    
                    # autoregressively generate token with custom stopping criteria (as defined by the Nougat authors)
                    outputs = model.generate(pixel_values.to(device),
                          min_length=1,
                          max_length=3584,
                          bad_words_ids=[[processor.tokenizer.unk_token_id]],
                          return_dict_in_generate=True,
                          output_scores=True,
                          stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
                                             )
                    generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
                    generated = processor.post_process_generation(generated, fix_markdown=True)
                    this_pdf_text.append(generated)

                    if len(tables) > 0:
                        for table in tables:
                            this_pdf_text.append(table_formatter.extract(tables[0]).df().to_markdown())

                output_text = "\n".join(this_pdf_text)
                    
                # output text
                with open(os.path.join(current_dir, 'text.txt'), 'w') as f:
                    f.write(output_text)

                # copy pdf to target directory
                shutil.copy(os.path.join(pdf_path, pdffile_name), os.path.join(current_dir, pdffile_name))
                doc.close()

            # If an exception occurs, delete the working output dir
            
            # quit if keyboard interrupt
            except KeyboardInterrupt:
                shutil.rmtree(current_dir)
                with open('failed.txt', 'a') as f:
                    f.write(pdffile_name)
                sys.exit() # lol

            # keep going if other kind of error, write filename to failed.txt
            except:
                print(sys.exc_info()[0])
                print("{} failed.".format(pdffile_name))
                shutil.rmtree(current_dir)
                with open('failed.txt', 'a') as f:
                    f.write(pdffile_name)

        else:
            continue
