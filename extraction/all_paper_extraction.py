from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import transformers
import torch
import json, sys, os

def format_prompt(prompt, paper_text):
  PROMPT = f"Question: {prompt}\nContext: " + paper_text
  return PROMPT
  
with open('/lustre/isaac/proj/UTK0254/lp/pdf_processor/extraction/output/database.json') as f:
    database = json.load(f)

def format_material_string(string, splitchar):
    string = string.replace('MATERIAL:', '')
    string_list = "".join(string.split()).split(splitchar)
    return string_list
    
def format_temperature_string(temp_list):
    temp_list = "".join(temp_list.replace('K', '').split()).split(',')
    temp_list = [temp_string[:-1] if temp_string[-1] == '.' else temp_string for temp_string in temp_list]
    temp_list_float = []
    for temp in temp_list:
        try:
            temp_list_float.append(float(temp))
        except ValueError:
            temp_list_float.append(-100.0)
    return temp_list_float

def standard_chem_formula(dictionary):
    return ''.join(char for char in ''.join(key for key in sorted(dictionary.keys())) if char.isalpha() or char.isnumeric())

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Don't make up an answer."""

temp_question = "What is the critical temperature at zero-field of {MATERIAL}? Just give a number and do not provide any explanation. Format the answer as CRITICAL TEMPERATURE: {Number} K. Do not write a sentence before the output. Here are some example outputs: 'CRITICAL TEMPERATURE: 3 K', 'CRITICAL TEMPERATURE: 15.6 K'"

source_question = "Can you reproduce the exact sentence where you got this information? Do not make up a sentence or attempt to explain."

#formula_question = "What is the material studied in this paper? Format the answer as MATERIAL: {Chemical Formula}. If there are multiple materials, separate them with &. Just give a formula and do not provide any explanation. Here are some example outputs: 'MATERIAL: Ga3As4.5 & Al0.6Fe0.4 & TexS1-x & UF6', 'MATERIAL: PrOs4Sb12 & PrOs3Sb13'"

#field_question = "What is upper critical field of {MATERIAL}? Just give a number and do not provide any explanation. Format the answer as MAGNETIC FIELD: {Number} T."


# Setup Model

model_sig='405B'
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

NUM_CHARS = int(sys.argv[1])


# Read in superconductivity dataset

processed_database_output = {}
for key in database.keys():
    materials = format_material_string(database[key][0], ',')
    temperatures = format_temperature_string(database[key][1].replace('CRITICAL TEMPERATURE:', ''))
    if len(materials) > len(temperatures):
        temperatures = temperatures + [-100.0] * (len(materials) - len(temperatures))
    processed_database_output[key] = (materials, temperatures)

database = processed_database_output

# Run extraction

answers = {}
answers['questions'] = [temp_question, source_question]

paper_source_directory = '/lustre/isaac/proj/UTK0254/lp/superconductivity_dataset/'
file_name = 'out.txt'

for i, directory in enumerate(os.listdir(paper_source_directory)):
    if directory not in database.keys(): continue
    if len(database[directory][0]) != 1: continue

    object_path = os.path.join(paper_source_directory, directory)
    if os.path.isdir(object_path) and not directory.startswith("."):
        
        paper_textfile = os.path.join(paper_source_directory, directory, file_name)
        with open(paper_textfile) as f:

            print("Parsing {} ".format(paper_textfile))
            paper_text = f.read()
            answers[directory] = [[] for i in range(len(answers['questions']))]

            materials = database[directory][0]

            answers[directory][0].append('&'.join(materials))

            q1list, q2list = [], []
            for material in materials:
              messages = [{"role":"system", "content":SYS_PROMPT}]

              question_prompt = format_prompt(temp_question.replace('{MATERIAL}', material), paper_text)
              messages.append({"role":"user","content":question_prompt[:NUM_CHARS]})

              response = pipe(messages, max_new_tokens=32)[0]['generated_text'][-1]['content']
              q1list.append(response)
              messages.append({"role":"assistant", "content":response})

              messages.append({"role":"user","content":source_question})
              response = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
              q2list.append(response)

            # attempt to parse temperature output
            try:
                answers[directory][0].append(",".join(q.split(":")[-1] for q in q1list))
            except:
                answers[directory][0].append(q1list)

            answers[directory][1] = q2list

output_path = '/lustre/isaac/proj/UTK0254/lp/pdf_processor/extraction/output/'
with open(os.path.join(output_path, 'run_llama3_1_{}_{}chars_qs.json'.format(model_sig, NUM_CHARS)), 'w', encoding='utf-8') as f:
    json.dump(dict(sorted(answers.items())), f, ensure_ascii=False, indent=4)
