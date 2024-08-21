import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import json, sys, os

sys_prompt = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Here is the document: """

@st.cache_resource()
def load_model():
  model_sig='8B'
  model_id = "meta-llama/Meta-Llama-3.1-{}-Instruct".format(model_sig)

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
    quantization_config=bnb_config,
    device_map="auto",
  )

  terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", eos_token_id=terminators)

  return pipe

st.title("📝 File Q&A with LLama3.1-8B")

if "model" not in st.session_state:
  st.session_state["model"] = load_model()

st.session_state.file_name = st.file_uploader("Upload an article", type=("txt"))
  
if "messages" not in st.session_state:
  st.session_state.messages = []
  
if "messages" in st.session_state and st.session_state.file_name is not None and len(st.session_state.messages) == 0:
  print("initializing sys prompt")
  st.session_state.messages = [{"role":"system", "content": sys_prompt + st.session_state.file_name.read().decode()[:18000]}]

print("file_name" in st.session_state)
print(st.session_state.file_name)
print("messages" in st.session_state)
  
# Display chat messages from history on app rerun
if "messages" in st.session_state:
  print(st.session_state.messages)
  print(st.session_state.file_name)
  for message in st.session_state.messages:
    if message["role"] != "system":
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "messages" in st.session_state:
  if question := st.chat_input("ask a question"):
    messages = st.session_state.messages
    pipe = st.session_state["model"]

    messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
      st.markdown(question)

    with st.chat_message("assistant"):
      response = pipe(messages, max_new_tokens=64)[0]['generated_text'][-1]['content']
      st.markdown(response)
      
    messages.append({"role": "assistant", "content": response})
