# Author : Jackson Dendy
# Last Updated : 12/20/2024
# Description : Script to Retrieve papers based on critical temperature to test a certain sentence retirival accuracy metric

from pathlib import Path
import os, json, sys

## Extract file data
root = str(Path(__file__).parents[1])
#sys.path.insert(root, 1)
directory = f"{root}/questions/test"

# Iterate over files in directory
i = 0
positive_papers = []

for name in os.listdir(directory):
    papers = os.path.join(directory, name)

    if os.path.isdir(papers):
        for ppr in os.listdir(papers):

            if ".txt" in ppr: 
                with open(os.path.join(papers, ppr)) as f:
                    content = f.read()
                    if "critical temperature" or "Critical Temperature" or "Critical temperature" or "Crit Temp" or "Critical Temp" in content:
                        i+=1
                        terms = ["critical temperature", "Critical Temperature", "Critical temperature", "Crit Temp", "Critical Temp"]
                        for k in terms:
                            num = content.find(k)
                            if not(num == -1):
                                paragraph = content[num-100:num+100]
                                long = content[num-1000:num+1000]
                                positive_papers.append({"doi": name, "paragraph": paragraph, "long_paragraph": long})
unique_papers = []
Nan_papers = []
for i in positive_papers:
    if not(i in unique_papers):
        unique_papers.append(i)   
        Nan_papers.append({"doi": i["doi"], "messages": [{"question": "What is the critical temperature of in the paper", "answer": 0}]})

with open("crit_temp_paper/papers.jsonl", "a") as p:
    for i in unique_papers:
        p.write(json.dumps(i) + "\n") 
   


            