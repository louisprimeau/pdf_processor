from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import json

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Don't make up an answer."""

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# use quantization to lower GPU usage                                                
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
  cache_dir="/lustre/isaac/proj/UTK0254/lp",
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

formula_question = "What is the material studied in this paper? Format the answer as MATERIAL: {Chemical Formula}. If there are multiple materials, separate them with &.  Just give a formula and do not provide any explanation."
temp_question = "What is the critical temperature at zero-field of {MATERIAL}? Just give a number and do not provide any explanation. The critical temperature is sometimes expressed as Tc, T_c, $T_c$, or $T_{c}$. Format the answer as CRITICAL TEMPERATURE: {Number} K."
field_question = "What is upper critical field of {MATERIAL}? Just give a number and do not provide any explanation. Format the answer as MAGNETIC FIELD: {Number} T."

def format_prompt(prompt, paper_text):
  PROMPT = f"Question: {prompt}\nContext: " + paper_text
  return PROMPT

def generate(formatted_prompt):
  formatted_prompt = formatted_prompt[:4000] # to avoid GPU OOM                      
  messages = [{"role":"system","content":SYS_PROMPT}, {"role":"user","content":formatted_prompt}]
  # tell the model to generate                                                       
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)
  outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )
  response = outputs[0][input_ids.shape[-1]:]
  return tokenizer.decode(response, skip_special_tokens=True)

import os

answers = {}
answers['questions'] = [formula_question, temp_question, field_question]

paper_source_directory = './processed_data/superconductivity_processed/'
file_name = 'text.txt'

for i, directory in enumerate(os.listdir(paper_source_directory)):
    object_path = os.path.join(paper_source_directory, directory)
    print(i, object_path)
    if os.path.isdir(object_path) and not directory.startswith("."):
        
        paper_textfile = os.path.join(paper_source_directory, directory, 'text.txt')
        with open(paper_textfile) as f:
            paper_text = f.read()
            answers[directory] = []
            material_string = generate(format_prompt(formula_question, paper_text))
            print(material_string)
            answers[directory].append(material_string)
            materials = [out.strip() for out in material_string.split(":")[1].split("&")]
            q1list = []
            q2list = []
            for material in materials:
                q1list.append(generate(format_prompt(temp_question.replace('{MATERIAL}', material), paper_text)))
                q2list.append(generate(format_prompt(field_question.replace('{MATERIAL}', material), paper_text)))
                print(q1list, q2list)
            answers[directory].append(",".join(q.split(":")[1] for q in q1list))
            answers[directory].append(",".join(q.split(":")[1] for q in q2list))

with open('run4.json', 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)
