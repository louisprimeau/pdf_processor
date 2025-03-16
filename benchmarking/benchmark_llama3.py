from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import datasets
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import transformers
import torch
import json
import gc

import GPUtil
GPUtil.showUtilization()

#from accelerate import PartialState
with torch.profiler.profile() as profiler:
    pass

def memory_usage():
    memory_usage = []
    for device in range(torch.cuda.device_count()):
        memory_usage.append(torch.cuda.memory_allocated(device) / 1024 ** 3)
    return memory_usage

def generate(tokenizer, model, messages, max_new_tokens=32):

  # tell the model to generate                                                       
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)
  
  print("token shape", input_ids.shape)
  
  outputs = model.generate(
      input_ids,
      max_new_tokens=max_new_tokens,
      eos_token_id=terminators,
      do_sample=False, #temperature=0.6, top_p=0.9,
      pad_token_id=tokenizer.eos_token_id,
  )

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

finetuned_name = 'Meta-Llama-3.1-405B-Instruct_test_4b'
#finetuned_name = 'llama3.18B-finetunedlp'
#model = PeftModel.from_pretrained(model, finetuned_name)
#breakpoint()

model.eval()
skip_to = 0
counter = 0
test_with = 'validation'
with torch.no_grad(), open(finetuned_name + '_{}.jsonl'.format(test_with), 'a') as f, open(finetuned_name + f'_{test_with}_memory.txt', 'a') as g:
    """
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=3, repeat=2  # Adjust based on your profiling needs
        ),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as profiler:"""
    
    for i, message in enumerate(iterable_dataset):
        message = message['messages']
        if counter + len(message) < skip_to:
            counter = counter + len(message)
            continue

        sys_prompt = message[0]
        sys_prompt['content'] = sys_prompt['content'][:8000]
        
        llm_summarize_chat = [sys_prompt]
        
        # summarize the paper first, usually helps output
        summarize_question = "Please summarize the material properties studied in this paper. Put the material properties studied in a table format along with their values. Then comment on why this paper is useful and interesting."            
        llm_summarize_chat.append({'role': 'user', 'content': summarize_question})
        out = generate(tokenizer, model, llm_summarize_chat, max_new_tokens=1024)
        llm_summarize_chat.append({'role': 'assistant', 'content': out})
        
        for j in range(1, len(message), 2):
            counter += 1
            if counter < skip_to: continue
            
            llm_message = llm_summarize_chat.copy()
            llm_message.append(message[j])
            
            words = llm_message[-1]['content'].split(" ")
            question_type = " ".join(words[3:words.index("for")])

            print("example {}".format(i))
            print(question_type)
            for device in range(torch.cuda.device_count()):
                print("memory allocated on device {}: {}GB".format(device, torch.cuda.memory_allocated(device)/1024**3))
            
            g.write(",".join(str(i) for i in memory_usage()) + "\n")
            GPUtil.showUtilization()
            out = generate(tokenizer, model, llm_message, max_new_tokens=32)
            out_dict = {'question_type': question_type, 'response': out, 'answer': message[j+1]['content']}
            
            g.write(",".join(str(i) for i in memory_usage()) + "\n")
            GPUtil.showUtilization()
            print(out_dict)
            
            f.write(json.dumps(out_dict) + "\n")


            torch.cuda.empty_cache()
            gc.collect()
        #profiler.step()
        
        #if i > 2: break
