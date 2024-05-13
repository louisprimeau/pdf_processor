import os

#paper_source_directory = '/home/louis/research/pdf_processor/processed_data/superconductivity_processed/physrevb.71.134526'
#paper_source_directory = '/home/louis/research/pdf_processor/processed_data/superconductivity_processed/physrevb.88.144511'
#paper_source_directory = '/home/louis/research/pdf_processor/processed_data/superconductivity_processed/physrevb.31.1329'
paper_source_directory = '../processed_data/superconductivity_processed/physrevb.28.1389'
#paper_source_directory = '/home/louis/research/pdf_processor/processed_data/superconductivity_processed/physrevb.85.214519'
#paper_source_directory = '/home/louis/research/pdf_processor/processed_data/superconductivity_processed/physrevb.98.094505'

file_name = 'text.txt'

with open(os.path.join(paper_source_directory, file_name)) as f:
    paper_text = f.read()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch

SYS_PROMPT = """You are a condensed matter physicist. You are given the extracted parts of a long document and a question. Read the document and don't make up an answer."""

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

def format_prompt(prompt, paper_text):
  PROMPT = f"Question: {prompt}\nContext: " + paper_text
  return PROMPT

def generate(formatted_prompt):
  formatted_prompt = formatted_prompt[:10000] # to avoid GPU OOM                      
  messages = [{"role":"system","content":SYS_PROMPT}, {"role":"user","content":formatted_prompt}]
  # tell the model to generate                                                       
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)
  print("sequence length:", input_ids.shape)
  outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=terminators,
      do_sample=False, #temperature=0.6, top_p=0.9,
  )
  response = outputs[0][input_ids.shape[-1]:]
  return tokenizer.decode(response, skip_special_tokens=True)

print(generate(format_prompt("What is the material studied in this paper? Format the answer as MATERIAL: {Chemical Formula}. If there are multiple materials, separate them with &. Just give a formula and do not provide any explanation. Here are some example outputs: 'MATERIAL: Ga3As4.5 & Al0.6Fe0.4 & TexS1-x & UF6', 'MATERIAL: PrOs4Sb12 & PrOs3Sb13'", paper_text)))
        
