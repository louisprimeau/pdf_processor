from flask import *
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import numpy as np
import json, sys, os


app = Flask(__name__)
app.secret_key = b'oi_Q))o{7Q,<lH0zf*fndNJUjf>.Uk4J)s0aDawB$!YZhGciacj-~?B4/hUSt..'
app.permanent = True


model_sig='8B'
model_id = "meta-llama/Meta-Llama-3.1-{}-Instruct".format(model_sig)

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

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

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", eos_token_id=terminators)

messages = []

@app.route('/')
def home():

    return "Model Initalized"

@app.route('/activate_model/<prompt>', methods = ['GET', 'POST'])
def activate(prompt):
    
    messages.append({"role": "system", "content": prompt})

    return "True"

@app.route('/request/<request>', methods = ['GET', 'POST'])
def request(request):
    messages.append({"role": "user", "content": request})
    response = pipe(messages, max_new_tokens=1000)[0]['generated_text'][-1]
    messages.append(response)

    return  response['content']

@app.route('/getmessages', methods = ['GET'])
def getmessages():
    return messages

@app.route('/clear', methods = ['GET', 'POST'])
def clear():
    
    x = len(messages)
    for i in range(x):
        
        messages.pop(-1)
        
    
    return "True"

@app.route("/clearish")
def clearish():
    new = []
    for i, ob in enumerate(messages):
        if "system" == ob['role']:
            new.append(ob)
    x = len(messages)
    for i in range(x):
        messages.pop(-1)

    for i in new:
        messages.append(i)

    return "True"

if __name__=='__main__':
    app.run(port=7777)