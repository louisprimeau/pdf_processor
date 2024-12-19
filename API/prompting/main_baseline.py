# Author : Jackson Dendy 
# Last Update : 12/18/2024
# Description : Running this file will intake a jsonl file of questions and chains along with a folder ot text files and 
# Ask each paper the corresponding questions with each chain it will then save the results to a folder as a jsonl file
# This file is made to establish a baseline for the model

from API import Louis, makedir, jsonl_read
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import WhitespaceTokenizer
from rouge_score import rouge_scorer


import os, json

## Extract file data
path_root = Path(__file__).parents[0]
root = str(Path(__file__).parents[0])

# Assigns Root Directory
directory = root

## Dfine where certain files are and system prompt
sys = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Be very concise and avoid wordy responses. """
chain_file="gpt4ochain.jsonl"
#chain_file="testchains.jsonl"
question_file="test.jsonl"

## Initialize File Structure
test_name = "Test_1"
results_path = f"{directory}/results/{test_name}"
results_path = makedir(results_path)

print("Files Initialized Contacting Louis")
# Records system prompt
with f open(f"{results_path}/sys.txt", "a"):
    f.write(sys)
    f.close()

# Intialize instance of Louis Class
API = Louis("http://127.0.0.1:7777", sys)

# Extracts a list of dictionarys from jsonl files
questions = jsonl_read(f"{directory}/questions/{question_file}")
chains = jsonl_read(f"{directory}/chains/{chain_file}")

# Querys API with questions
for i, val in enumerate(questions):

    # Opens Files again to prevent atomic bomb (total data loss)
    qa = open(f'{results_path}/results.jsonl', "a")
    error = open(f'{results_path}/errors.txt', 'a')

    # Clears all prompts except system
    print(f"Beginning with {val['doi']}")
    API.clear_sys()

    qs = val['messages']

    # Intializes the data point in jsonl
    entry = {"paper": val['doi'], "logs": []}
    filepath = f"{directory}/questions/test/{val['doi']}"
    
    # Uploads file to API
    state = API.upload(f"{filepath}/text_converted.txt")
    if state == "True":
            print("Uploaded Succesfully")
            paper = open(state, "r")
    else:
        state = API.upload(f"{filepath}/text.txt")
        if state == "True":
            print("File Succesfully uploaded")
            paper = open(state, "r")
        else:
            error.write(filepath)
            print("File Failed")
    paper_string = paper.read()
    paper.close()
    
    # Large nested loops of order Chains(prompts -> questions(points))
    for c in chains:

        # Clears everything except the first 2 entrys (paper and sys) from API cache
        API.clear_chain(1)
        prompts = c['prompts']
        chain_num = c["chain"]
        point = {"num": c["chain"], "chain": [], "requests": []}
        print(f"Initializing Chain {chain_num}")

        # Asks prompts and stores data
        for p in prompts:
            response = API.request(p)
            chain_point = {"prompt": p, "response": response}
            point['chain'].append(chain_point)

        # Asks questions and stores data
        for q in qs:

            # Clears previous question for API cache
            API.clear_chain(len(prompts))
            print("Asking Question")

            response = API.request(q['question']) 
            print("Question Answered")
            answer = q['answer']

            # Evaluates String Matching Score
            tk = WhitespaceTokenizer()

            # Rouge Score
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(q["answer"], response)
            rouge2 = scores['rouge2'].fmeasure
            rougeL = scores['rougeL'].fmeasure

            # reference: https://arxiv.org/abs/2308.04624
            #TODO: E2E
            
            # LLM Score
            llm_assesment = API.zero_shot(f"String 1 is {answer} and String 2 is {response}")

            # Sentence Retrival
            sentence = API.request("Return the sentence you retrieved the answer to the question from. Only display the sentence and no other tokens in the response")
            sentence_bool = sentence in paper_string
            point['requests'].append({"question": q['question'], "answer": q['answer'], "response": response, 'rouge2':  rouge2, 'rougeL': rougeL, "LLM": llm_assesment, "sentence": sentence, "sentence_prescense": sentence_bool})
            
        entry['logs'].append(point)

    qa.write(json.dumps(entry) + "\n")
    qa.close()
    error.close()

    print(f"Finished with {val['doi']}\n")
    
        
