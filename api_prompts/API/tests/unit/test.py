# Author : Jackson Dendy 
# Last Update : 12/16/2024
# Description : General Test file to just run some one line tests. Nothing here besides junk code.

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import WhitespaceTokenizer
from rouge_score import rouge_scorer
import os
import json
from rake_nltk import Rake
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


'''
llm = LLM(model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
samp = SamplingParams(temperature=0,max_tokens = 1000, n = 1)
outputs = llm.generate("Hello, my name is", samp)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

'''


import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
one_under_path_root = str(Path(__file__).parents[0])
root = str(path_root)
sys.path.insert(1, root)
import Model

'''
model = Model.Model("http://127.0.0.1:7777", "This is just a test for functionality. When I say one you say two.")
model.request("one")
print(eval(model.getmessages())[-1]['content'])
'''

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id)

#llm = LLM(model=model_id, tokenizer = tokenizer)
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id)

model = AutoModelForCausalLM.from_pretrained(
model_id,
torch_dtype=torch.bfloat16,
quantization_config=bnb_config,
device_map="auto",
)

terminators = [
tokenizer.eos_token_id,
tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipe_call = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", eos_token_id=terminators)

print(pipe_call([{"role": "user", "content": "List the number 1000 40 times"}], max_new_tokens=1000)[0]['generated_text'][-1]['content'])