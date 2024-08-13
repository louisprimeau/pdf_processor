from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import datasets
#from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import transformers
import torch
import json
#from accelerate import PartialState


def generate(tokenizer, model, messages):

  # tell the model to generate                                                       
  input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(model.device)
  
  
  outputs = model.generate(
      input_ids,
      max_new_tokens=32,
      eos_token_id=terminators,
      do_sample=False, #temperature=0.6, top_p=0.9,
      pad_token_id=tokenizer.eos_token_id,
  )

  response = outputs[0][input_ids.shape[-1]:]

  #torch.cuda.empty_cache()
  
  return tokenizer.decode(response, skip_special_tokens=True)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
)

model.generation_config.pad_token_id = tokenizer.pad_token_id

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

data = datasets.load_dataset('/home/louis/research/pdf_processor/finetune/superconductivity_dataset', 'val', split='validation', streaming=True)

iterable_dataset = iter(data)

finetuned_name = 'llama3.1-8B'
#finetuned_name = 'llama3.1-8B-finetunedlp'
# model = PeftModel.from_pretrained(model, finetuned_name
test_with = 'validation'
with torch.no_grad(), open(finetuned_name + '_{}.json'.format(test_with), 'a') as f:

    for i, message in enumerate(iterable_dataset):
        message = message['messages']
        llm_message = message[0:2]
        llm_message[1]['content'] = llm_message[1]['content'] + ' If there are multiple materials studied, list the properties for them in a comma separated list, e.g. X, Y. No need to put additional punctuation at the end of your message.'

        llm_message[0]['content'] = llm_message[0]['content'][:28000]

        print("example {}".format(i))
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024**3))
        print("torch.cuda.max_memory_allocated: %fGB"%(torch.cuda.max_memory_allocated(0)/1024**3))

        out = generate(tokenizer, model, llm_message)
        out_dict = {'question': message[1]['content'],'response': out, 'answer': message[2]['content']}
        del message
        f.write(json.dumps(out_dict) + "\n")

        gc.collect()
        torch.cuda.empty_cache()
