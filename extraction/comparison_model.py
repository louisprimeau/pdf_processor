import json
import chemparse
import os

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="nasa-impact/nasa-smd-ibm-st-v2", breakpoint_threshold_type="percentile"))

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining

model = SentenceTransformer("nasa-impact/nasa-smd-ibm-st-v2")

with open('/lustre/isaac/proj/UTK0254/lp/pdf_processor/extraction/output/run_llama3_1_8B_28000chars_qs.json') as f:
    model_output_1 = json.load(f)

with open('/lustre/isaac/proj/UTK0254/lp/pdf_processor/extraction/output/run_llama3_1_70B_28000chars_qs.json') as f:
    model_output_2 = json.load(f)

def format_temperature_string(temp_list):
    temp_list = "".join(temp_list.replace('K', '').split()).split(',')
    temp_list = [t for t in temp_list if len(t) > 0]
    temp_list = [temp_string[:-1] if temp_string[-1] == '.' else temp_string for temp_string in temp_list]
    temp_list_float = []
    for temp in temp_list:
        try:
            temp_list_float.append(float(temp))
        except ValueError:
            print(temp)
            temp_list_float.append(-100.0)
    return temp_list_float


num_checked = 0
num_temps_same = 0
num_sentences_same = 0
num_correct_sentences_1 = 0
num_correct_sentences_2 = 0
for key in model_output_1.keys():
    if key == 'questions': continue
    if key in model_output_2.keys():

        paper_source_directory = '/lustre/isaac/proj/UTK0254/lp/superconductivity_dataset/'
        file_name = 'out.txt'
        paper_textfile = os.path.join(paper_source_directory, key, file_name)
        with open(paper_textfile) as f:
            paper_text = f.read()


        breakpoint()
        temp_1, sentence_1 = model_output_1[key]
        temp_2, sentence_2 = model_output_2[key]
        sentence_1 = sentence_1[0]
        sentence_2 = sentence_2[0]
        temp_1_raw = temp_1[1]
        temp_2_raw = temp_2[1]
        
        temp_1 = format_temperature_string(temp_1[1])[0]
        temp_2 = format_temperature_string(temp_2[1])[0]
        
        if '"' in sentence_1: sentence_1 = sentence_1.split('"')[1]
        if '"' in sentence_2: sentence_2 = sentence_2.split('"')[1]

        print("---------------------------------")
        print(key)
        print("(",temp_1_raw, temp_1,") (", temp_2_raw, temp_2, ")", sentence_1 in paper_text, sentence_2 in paper_text)
        print(sentence_1)
        print(sentence_2)
        print("---------------------------------")
        if temp_1 == temp_2: num_temps_same += 1
        if sentence_1 == sentence_2: num_sentences_same += 1
        if sentence_1 in paper_text: num_correct_sentences_1 +=1
        if sentence_2 in paper_text: num_correct_sentences_2 +=1
        num_checked += 1


print("matching temps:", num_temps_same / num_checked)
print("matching sentences:", num_sentences_same / num_checked)
print("8B sentences correct:",  num_correct_sentences_1 / num_checked)
print("70B sentences correct:",  num_correct_sentences_2 / num_checked)
