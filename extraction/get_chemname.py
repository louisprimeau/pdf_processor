from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
"""
pipeline = transformers.pipeline(
  "text-generation",
  model="meta-llama/Meta-Llama-3-8B-Instruct",
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="cuda",
)
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

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
    quantization_config=bnb_config
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]



def format_prompt(prompt,retrieved_documents,k):
"""using the retrieved documents we will prompt the model to generate our responses"""
  PROMPT = f"Question:{prompt}\nContext:"
  for idx in range(k) :
    PROMPT+= f"{retrieved_documents['text'][idx]}\n"
  return PROMPT

def generate(formatted_prompt):
  formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
  messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
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

def rag_chatbot(prompt:str,k:int=2):
  scores , retrieved_documents = search(prompt, k)
  formatted_prompt = format_prompt(prompt,retrieved_documents,k)
  return generate(formatted_prompt)
