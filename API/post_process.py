import matplotlib.pyplot as plt  
import numpy as np
from utility import *

results_path = "/home/jdendy/pdf_processor/API/results/Test_2/results.jsonl"
paper_analyzed = "physrevb.50.4144"

q = jsonl_read(results_path)

for val in q:
    
    if val['paper'] == paper_analyzed:
        paper = val
        print("Entry Found")

num_questions = len(paper['logs'][0]['requests'])
num_chains = len(paper['logs'])
#Entry structure : {paper: "", logs:[{num:"",chain: [], requests[{question, answer, response, scores}]}]}
for i in range(num_questions):
    plt.figure(figsize= (7,5))
    plt.title(f"Questions {i} LLM Scores")
    plt.xlabel("Chain Number")
    plt.ylabel("LLM Score 1-100")   
    score_array = [j['requests'][i]['LLM'] for j in paper['logs']]
    print(score_array)
    chain_array = [j['num'] for j in paper['logs']]    
    plt.plot(chain_array , score_array)
    plt.savefig(f"/home/jdendy/pdf_processor/API/results/Test_2/Question_{i}.png")
    
    




