# Author : Jackson Dendy 
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


model = Model.Model("http://127.0.0.1:7777", "This is just a test for functionality. When I say one you say two. Only respond with the word two and do not include any other tokens in your response.")
model.request("one")
print(model.getmessages())



