import os, io
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image, ImageDraw
import numpy as np
from gmft.pdf_bindings import PyPDFium2Document
from .util import RunningVarTorch, StoppingCriteriaScores, rasterize_paper

def extract_pdf_text(pdf_path, model, processor, table_detector, table_formatter, device):
    
    this_pdf_text = []
    #if isinstance(pdf_path, io.BytesIO): pdf_path = pdf_path.read()
    #doc = PyPDFium2Document(pdf_path)
    images = rasterize_paper(pdf=pdf_path, return_pil=True)
    for image_file in images:
        #tables = table_detector.extract(pymupdfpage)
        image = Image.open(image_file)
        #if len(tables) > 0:
        #    draw = ImageDraw.Draw(image)
        #    for table in tables:
        #        draw.rectangle(list(np.array(table.bbox) * 96 / 72), fill=(255, 255, 255))

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
    
    return "\n".join(this_pdf_text)