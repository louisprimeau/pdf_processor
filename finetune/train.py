
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import datasets
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import transformers
import torch
import json
from accelerate import Accelerator, PartialState

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Don't make up an answer."""

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# use quantization to lower GPU usage                                                
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": PartialState().process_index},
    quantization_config=bnb_config,
    attn_implementation="eager",
)

model.resize_token_embeddings(len(tokenizer))

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

data = datasets.load_dataset('/lustre/isaac/proj/UTK0254/lp/pdf_processor/finetune/superconductivity_dataset/', 'train', trust_remote_code=True)

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

output_model="llama3.18B-finetunedlp"

sft_config = SFTConfig(packing=False,
                       max_seq_length=1024,
                       output_dir=output_model,
                       per_device_train_batch_size=4,
                       per_device_eval_batch_size=1,
                       gradient_accumulation_steps=4,
                       optim="paged_adamw_32bit",
                       learning_rate=2e-4,
                       lr_scheduler_type="cosine",
                       save_strategy="steps",
                       save_steps=0.1,
                       logging_steps=10,
                       max_steps=250,
                       fp16=False,
                       bf16=True,
                       push_to_hub=False,
                       report_to="none",
                       )


trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        peft_config=peft_config,
        args=sft_config,
        tokenizer=tokenizer,
    )

trainer.train()
trainer.model.save_pretrained(output_model)
tokenizer.save_pretrained(output_model)
