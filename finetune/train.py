
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import datasets
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import transformers
import torch
import json

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Don't make up an answer."""

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# use quantization to lower GPU usage                                                
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)

model.resize_token_embeddings(len(tokenizer))

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def format_prompt(prompt, paper_text):
    PROMPT = f"Question: {prompt}\nContext: " + paper_text
    return PROMPT

def generate(example_dict):

    sys_prompt = "You are a helpful assistant. You will answer questions about the following paper: {}".format(example_dict['paper_dict'])

    formatted_prompt = "{} Just answer the question separated by commas. Do not attempt to explain your answer. If you do not know the answer, write NA.".format(example_dict['question'])
    
    messages = [{"role":"system","content": sys_prompt},
                {"role":"user","content": formatted_prompt}]
    # tell the model to generate                                                       
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=False, #temperature=0.6, top_p=0.9,
    )

    #response = outputs[0][input_ids.shape[-1]:]
    response = outputs[0]
    
    return tokenizer.decode(response, skip_special_tokens=True)

data = datasets.load_dataset('/home/louis/research/pdf_processor/finetune/superconductivity_dataset', 'train')

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )


model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

output_model="llama3.18B-finetunedlp"


sft_config = SFTConfig(packing=False,
                       max_seq_length=512,
                       output_dir="/tmp",
                       per_device_train_batch_size=4,
                       gradient_accumulation_steps=16,
                       optim="paged_adamw_32bit",
                       learning_rate=2e-4,
                       lr_scheduler_type="cosine",
                       save_strategy="epoch",
                       logging_steps=10,
                       num_train_epochs=3,
                       max_steps=250,
                       fp16=True,
                       push_to_hub=False,
                       report_to="none",
                       )


trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        peft_config=peft_config,
        args=sft_config,
        tokenizer=tokenizer,
    )

trainer.train()
