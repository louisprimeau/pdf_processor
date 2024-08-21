import streamlit as st
import transformers

sys_prompt = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer."""
temp_question = "What is the critical temperature at zero-field of {MATERIAL}? Just give a number and do not provide any explanation. Format the answer as CRITICAL TEMPERATURE: {Number} K. Do not write a sentence before the output. Here are some example outputs: 'CRITICAL TEMPERATURE: 3 K', 'CRITICAL TEMPERATURE: 15.6 K'"
source_question = "Can you reproduce the exact sentence where you got this information? Do not make up a sentence or attempt to explain."

def format_prompt(prompt, paper_text):
  PROMPT = f"Question: {prompt}\nContext: " + paper_text
  return PROMPT

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

st.title("📝 File Q&A with LLama3.1-8B")

uploaded_file = st.file_uploader("Upload an article", type=("txt"))
question = st.text_input(
    "Ask something about the article",
    placeholder=temp_question,
    disabled=not uploaded_file,
)

if uploaded_file and question:
    article = uploaded_file.read().decode()

    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": format_prompt(question, article)}
    
    ]
    response = pipe(messages, max_new_tokens=32)[0]['generated_text'][-1]['content']

    st.write("### Answer")
    st.write(response.completion)