from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import json

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Don't make up an answer."""

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

# use quantization to lower GPU usage                                                
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

NUM_CHARS = 4000

#formula_question = "What is the material studied in this paper? Format the answer as MATERIAL: {Chemical Formula}. If there are multiple materials, separate them with &. Just give a formula and do not provide any explanation. Here are some example outputs: 'MATERIAL: Ga3As4.5 & Al0.6Fe0.4 & TexS1-x & UF6', 'MATERIAL: PrOs4Sb12 & PrOs3Sb13'"
temp_question = "What is the critical temperature at zero-field of {MATERIAL}? Just give a number and do not provide any explanation. Format the answer as CRITICAL TEMPERATURE: {Number} K. Do not write a sentence before the output. Here are some example outputs: 'CRITICAL TEMPERATURE: 3 K', 'CRITICAL TEMPERATURE: 15.6 K'"
#field_question = "What is upper critical field of {MATERIAL}? Just give a number and do not provide any explanation. Format the answer as MAGNETIC FIELD: {Number} T."

def format_prompt(prompt, paper_text):
  PROMPT = f"Question: {prompt}\nContext: " + paper_text
  return PROMPT

def generate(formatted_prompt):
  formatted_prompt = formatted_prompt[:NUM_CHARS] # to avoid GPU OOM                      
  messages = [{"role":"system","content":SYS_PROMPT}, {"role":"user","content":formatted_prompt}]
  # tell the model to generate                                                       
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)
  print('{} characters = {} tokens'.format(len(formatted_prompt), input_ids.shape[1]))
  outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=terminators,
      do_sample=False, #temperature=0.6, top_p=0.9,
  )
  response = outputs[0][input_ids.shape[-1]:]
  return tokenizer.decode(response, skip_special_tokens=True)

import os
with open('/lustre/isaac/proj/UTK0254/lp/pdf_processor/extraction/database.json') as f:
    database = json.load(f)

def format_material_string(string, splitchar):
    string = string.replace('MATERIAL:', '')
    string_list = "".join(string.split()).split(splitchar)
    return string_list
    
def format_temperature_string(temp_list):
    temp_list = "".join(temp_list.replace('K', '').split()).split(',')
    temp_list = [temp_string[:-1] if temp_string[-1] == '.' else temp_string for temp_string in temp_list]
    temp_list_float = []
    for temp in temp_list:
        try:
            temp_list_float.append(float(temp))
        except ValueError:
            temp_list_float.append(-100.0)
    return temp_list_float

def standard_chem_formula(dictionary):
    return ''.join(char for char in ''.join(key for key in sorted(dictionary.keys())) if char.isalpha() or char.isnumeric())

processed_database_output = {}
for key in database.keys():
    materials = format_material_string(database[key][0], ',')
    temperatures = format_temperature_string(database[key][1].replace('CRITICAL TEMPERATURE:', ''))
    if len(materials) > len(temperatures):
        temperatures = temperatures + [-100.0] * (len(materials) - len(temperatures))
    processed_database_output[key] = (materials, temperatures)

database = processed_database_output

answers = {}
answers['questions'] = [temp_question]
paper_source_directory = '/lustre/isaac/proj/UTK0254/lp/pdf_processor/processed_data/superconductivity_processed/'
file_name = 'text.txt'

for i, directory in enumerate(os.listdir(paper_source_directory)):
    if directory not in database.keys(): continue
    if len(database[directory][0]) != 1: continue
    object_path = os.path.join(paper_source_directory, directory)
    print(i, object_path)
    if os.path.isdir(object_path) and not directory.startswith("."):
        
        paper_textfile = os.path.join(paper_source_directory, directory, 'text.txt')
        with open(paper_textfile) as f:
            paper_text = f.read()
            answers[directory] = []
            materials = database[directory][0]
            answers[directory].append('&'.join(materials))
            q1list = []
            for material in materials:
                q1list.append(generate(format_prompt(temp_question.replace('{MATERIAL}', material), paper_text)))
            print(q1list)
            try:
                answers[directory].append(",".join(q.split(":")[-1] for q in q1list))
            except:
                answers[directory].append(q1list)
            print(answers[directory])

with open('/lustre/isaac/proj/UTK0254/lp/pdf_processor/run_70B_{}chars.json'.format(NUM_CHARS), 'w', encoding='utf-8') as f:
    json.dump(dict(sorted(answers.items())), f, ensure_ascii=False, indent=4)
