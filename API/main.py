from Louis import Louis
from pathlib import Path
from utility import jsonl_read, makedir

import os

path_root = Path(__file__).parents[0]
root = str(Path(__file__).parents[0])

#TODO: Tasks
'''
- Add support for Chains
- Add support for Quantitatize loss to respond to answers
- Add support for PDF interpretation with the OCR
- BLEU score
- make 40 for prompts
'''
# Assigns Root Directory
directory = root

sys = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Here is the document: """

# Initialize File Structure
test_name = "Test_1"
results_path = f"{directory}/results/{test_name}"
makedir(results_path)

print("Files Initialized Contacting Louis")

# Calls the API I created 
API = Louis("http://127.0.0.1:7777", sys)

questions = jsonl_read(f"{directory}/questions/test.jsonl")
chains = jsonl_read(f"{directory}/chains/chains.jsonl")

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

            point['requests'].append({'question': q['question'], 'answer': q['answer'], 'response': response})
        entry['logs'].append(point)
    qa.write(f"{entry}\n")
    qa.close()
    error.close()

    print(f"Finished with {val['doi']}\n")
    
        
