# Author : Jackson Dendy 
# Last Update : 12/16/2024
# Description : Creates all plots and evaluation metrics to assess the efficacy the main.py file

import matplotlib.pyplot as plt  
import numpy as np
from utility import *
test = 2
paper_analyzed = "physrevb.50.4144"
results_path = f"/home/jdendy/pdf_processor/API/results/Test_{test}/results.jsonl"


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
    chain_array = [j['num'] for j in paper['logs']]    
    plt.plot(chain_array , score_array)
    plt.savefig(f"/home/jdendy/pdf_processor/API/results/Test_{test}/{paper_analyzed}_Question_{i}.png")
    
    




