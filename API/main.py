from Louis import Louis
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import WhitespaceTokenizer
from rouge_score import rouge_scorer
from utility import jsonl_read, makedir

import os

path_root = Path(__file__).parents[0]
root = str(Path(__file__).parents[0])

#TODO: Tasks
'''
- Add support for PDF interpretation with the OCR
'''
# Assigns Root Directory
directory = root

sys = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Here is the document: """
chain_file="gpt4ochain.jsonl"
question_file="test.jsonl"

# Initialize File Structure
test_name = "Test_1"
results_path = f"{directory}/results/{test_name}"
makedir(results_path)

print("Files Initialized Contacting Louis")

# Calls the API I created 
API = Louis("http://127.0.0.1:7777", sys)

questions = jsonl_read(f"{directory}/questions/{question_file}")
chains = jsonl_read(f"{directory}/chains/{chain_file}")

# Querys API with questions
for i, val in enumerate(questions):

    # Opens Files again to prevent atomic bomb
    qa = open(f'{results_path}/results.txt', "a")
    error = open(f'{results_path}/errors.txt', 'a')

    # Clears all prompts except system
    print(f"Beginning with {val['doi']}")
    API.clear_sys()

    qs = val['messages']

    # Intializes the data point in jsonl
    entry = {'paper': val['doi'], 'logs': []}
    filepath = f"{directory}/questions/test/{val['doi']}"
    
    # Uploads file to API
    try:
        if os.path.isfile(f"{filepath}/text_converted.txt"):
            API.upload(f"{filepath}/text_converted.txt")
            print("File Succesfully Uploaded")
        else:
            API.upload(f"{filepath}/text.txt")
            print("File Succesfully Uploaded")
    except:
        error.write(f"{filepath}\n")  
        print("File Upload Error")
    
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
            chain_point = {'prompt': p, 'response': response}
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
            reference = tk.tokenize(q["answer"])
            prediction = tk.tokenize(response)

            # Bleu Score
            bleu_score = sentence_bleu([reference], prediction)

            # Rouge Score
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(q["answer"], response)
            rouge1 = scores['rouge1'].fmeasure
            rougeL = scores['rougeL'].fmeasure

            point['requests'].append({'question': q['question'], 'answer': q['answer'], 'response': response, 'bleu': bleu_score, 'rouge1':  rouge1, 'rougeL': rougeL})
        entry['logs'].append(point)
    qa.write(f"{entry}\n")
    qa.close()
    error.close()

    print(f"Finished with {val['doi']}\n")
    
        
