import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
import json
import gc
import GPUtil

def memory_usage():
    memory_usage = []
    for device in range(torch.cuda.device_count()):
        memory_usage.append(torch.cuda.memory_allocated(device) / 1024 ** 3)
    return memory_usage

def generate(tokenizer, model, messages, max_in_tokens=1024, max_new_tokens=32):
  GPUs = GPUtil.getGPUs()
  # tell the model to generate                                                
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  )[:, :max_in_tokens].to(model.device)
  
  print("token shape", input_ids.shape)
  print("total memory usage:", sum(gpu.memoryUsed for gpu in GPUs))
  outputs = model.generate(
      input_ids,
      max_new_tokens=max_new_tokens,
      eos_token_id=terminators,
      do_sample=False, #temperature=0.6, top_p=0.9,
      pad_token_id=tokenizer.eos_token_id,
  )
  print("total memory usage:", sum(gpu.memoryUsed for gpu in GPUs))
  gpu_usage.append(sum(gpu.memoryUsed for gpu in GPUs))

  response = outputs[0][input_ids.shape[-1]:]

  #torch.cuda.empty_cache()
  
  return tokenizer.decode(response, skip_special_tokens=True)

model_id = "meta-llama/Meta-Llama-3.1-405B-Instruct"

# use quantization to lower GPU usage                                                
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
    #device_map={"": PartialState().process_index},
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
)

model.generation_config.pad_token_id = tokenizer.pad_token_id

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

data = datasets.load_dataset('../finetune/superconductivity_dataset_cot', 'val', split='validation', streaming=True, trust_remote_code=True)

iterable_dataset = iter(data)

model.eval()
skip_to = 0
counter = 0
test_with = 'validation'

summarize_question = "Please summarize the material properties studied in this paper. Put the material properties studied in a table format along with their values. Then comment on why this paper is useful and interesting."
message_lengths = [len(message['messages'][0]['content']) for message in iter(data)]
longest_example = max(range(len(message_lengths)), key=lambda x: message_lengths[x])
message = [m for i, m in enumerate(iter(data)) if i==longest_example][0]
gpu_usage = []; sequence_lengths = list(range(1000, 22000, 5000))
GPUs = GPUtil.getGPUs()
with torch.no_grad():
    message = message['messages']
    sys_prompt = message[0]    
    llm_summarize_chat = [sys_prompt]
    llm_summarize_chat.append({'role': 'user', 'content': summarize_question})

    for sequence_length in sequence_lengths:
        out = generate(tokenizer, model, llm_summarize_chat, max_in_tokens=sequence_length, max_new_tokens=1024)
        #gpu_usage.append(sum(gpu.memoryUsed for gpu in GPUs))
        print(gpu_usage)
        torch.cuda.empty_cache()
        gc.collect()