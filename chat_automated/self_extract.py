from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import json, sys, os

# Setup Model
MODEL_SIG = '8B'
MODEL_ID = f'meta-llama/Meta-Llama-3.1-{MODEL_SIG}-Instruct'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
     MODEL_ID,
     torch_dtype=torch.bfloat16,
     quantization_config=bnb_config,
    device_map="auto",
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                eos_token_id=terminators)



PAPER_TEXT_PATH = "/home/louis/data/superconductivity_dataset_cot/train/physrevb.8.3479/out.txt"
PAPER_TEXT = open(PAPER_TEXT_PATH, encoding='utf-8').read()
SYS_PROMPT_CHATBOT = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. You cannot reply with explanations unless they are inside the JSON document. Don't make up an answer. The paper text is: """
SYS_PROMPT_INTERROGATER = """You are helpful assistant in navigating prompt engineering techniques for chatbots. """

messages_chatbot = [{"role": "system", "content": SYS_PROMPT_CHATBOT + PAPER_TEXT}]
messages_interrogater = [{"role": "system", "content": SYS_PROMPT_INTERROGATER},
                         {"role": "user", "content": "I want to extract superconductivity materials parameters from a scientific paper by having a chatbot read it. I would like to know also about the specific materials studied in the paper. Can you give me a prompt that will do that? Just provide the prompt, and no additional text."}]

num_messages = 2
for i in range(num_messages):
    out = pipe(messages_interrogater, max_new_tokens=512)[0]['generated_text'][-1]['content']
    print("Interrogator:")
    print(out + "\n")
    print("--------------------------------")
    messages_chatbot.append({"role": "user", "content": out})
    messages_interrogater.append({"role": "assistant", "content": out})

    out = pipe(messages_chatbot, max_new_tokens=512)[0]['generated_text'][-1]['content']
    print("Chatbot:")
    print(out + "\n")
    print("-------------------------------")
    messages_chatbot.append({"role": "assistant", "content": out})
    messages_interrogater.append({"role": "user", "content": "This is what the chatbot said: " + out + "Can you help me by writing my next prompt? Try to have the chatbot write JSON output. Just provide the prompt, and no additional text."})