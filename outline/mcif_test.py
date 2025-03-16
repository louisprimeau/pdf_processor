
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import datasets
import transformers
import torch
import json
import gc
from enum import Enum
from pydantic import BaseModel, constr, PositiveFloat, conint
from typing import Optional, Sequence, Tuple
import outlines
import datetime

class ElementMoment(BaseModel):
    element: constr(max_length=2)
    moment_x: float
    moment_y: float
    moment_z: float

class MCIF(BaseModel):
    moments: Sequence[ElementMoment]
    
model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

outline_model = outlines.models.Transformers(model, tokenizer)
generator = outlines.generate.json(outline_model, Material)

prompt = "You are a world class large language model tasked with extracting material information from passages of scientific papers. Please describe the material described on the following passage: {text} and extract the chemical formula, superconducting critical temperature, upper critical field, the Neel temperature, the a lattice constant, the b lattice constant, and the c lattice constant. If you can't find the answer in the provided text, skip it."

data = datasets.load_dataset('/lustre/isaac/proj/UTK0254/lp/MDR-Supercon-questions/', 'val', split='validation', streaming=True, trust_remote_code=True)

iterable_dataset = iter(data)

model.eval()
test_with = 'validation'

out_file_name = f'{model_id.split("/")[1]}_{test_with}_{datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S")}.jsonl'

#with open(out_file_name, 'a') as f:
for i, message in enumerate(iterable_dataset):
    out = generator(prompt.format(text=message['messages'][0]['content']))
    print(out.json())
    breakpoint()
    with open(out_file_name, "a") as f:
        f.write(out.json() + "\n")
        
    torch.cuda.empty_cache()
    gc.collect()
