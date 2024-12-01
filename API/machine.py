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

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id,
                                          padding = True,
                                          pad_token= "[PAD]")


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

    x = len(messages)
    for i in range(x):
        
        messages.pop(-1)
    
    messages.append({"role": "system", "content": prompt})

    return "True"

@app.route('/upload/<request>', methods = ["GET", "POST"])
def upload(request):
    request = request.replace("uquq", "/")
    state = os.path.exists(request)
    if state:
        file = open(request, 'r').read()
        messages.append({'role': "user", "content": str(file)})
        return "True"
    else:
        return "False"
    

@app.route('/request/<request>', methods = ['GET', 'POST'])
def request(request):
    request = request.replace("uquq", "/")
    messages.append({"role": "user", "content": request})
    response = pipe(messages, max_new_tokens=1000)[0]['generated_text'][-1]
    messages.append(response)

    return  response['content']

@app.route('/zero_shot/<question>', methods = ['GET', 'POST'])
def zero_shot(question):
    zero = [{"role": "sys", "content": "You are machine to compare two strings. Compare how similar the information contatined in them is. the higher the better. Penalize wordy responese. Only return one integer between 0 and 100. Do not return instructions, decriptions, or code. Only return one integer"}]
    zero.append({"role": "user", "content": "Return the comparison integer only of: " + question})
    response = pipe(zero, max_new_tokens=1000)[0]['generated_text'][-1]
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

@app.route("/clear_sys")
def clear_sys():
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

@app.route("/clear_chain/<length>")
def clear_chain(length):
    new = messages[0:(int(length))*2 + 2]
    
    x = len(messages)
    for i in range(x):
        messages.pop(-1)

    for i in new:
        messages.append(i)

    return "True"

@app.route("/E2E/<str1>/<str2>")
def E2E(str1, str2):
    str1 = str1.replace("uquq", "/")
    str2 = str2.replace("uquq", "/")
    tok = tokenizer([str1,str2], padding="longest")
    vec1 = torch.tensor(tok['input_ids'][0]).unsqueeze(0)
    vec2 = torch.tensor(tok['input_ids'][1]).unsqueeze(0)
    emb1 = model(vec1)[0].squeeze()
    emb2 = model(vec2)[0].squeeze()
    #cos = torch.nn.CosineSimilarity(dim=1)

    return ""#str(cos(emb1, emb2))

if __name__=='__main__':
    app.run(port=7777)