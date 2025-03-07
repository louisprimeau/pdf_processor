# Author : Jackson Dendy 
# Description : Supporting functions for files in this folder

import torch
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import matplotlib.pyplot as plt


def makedir(path):
    '''Creates self counting directorys in the form test_1 -> test_2 -> test_3 -> ...
        
        Parameters
        ----------
        
        path : str
            directory you want to create
        
        Returns
        -------
        
        path : str
            Unique path of directory to store test results'''
    while os.path.exists(path):
        index = path.split('_')
        bit = index[-1]
        index.pop(-1)
        path = [a + '_' for a in index]
        path = "".join(path)
        path += str(int(bit) + 1)
    os.mkdir(path)

    return path

# Creates the list of questions
def jsonl_read(file):
    '''Reads jsonl files
    
        Parameters
        ----------
        
        file : str
            jsonl file path

        Returns
        -------
        data : list
            list of dictionarys corresponding to the jsonl file
        '''
    j = open(file, "r")

    data = []
    for i in j.readlines():

        x = json.loads(i)
            
        data.append(x)

    j.close()
    return data

def batch_paper_pp(results_path, threshold_dict):
    '''
    tp = correct identified as correct
    tn = correctly identified as incorrect
    fp = incorrectly identified as correct
    fn = incorrectly identified as incorrect
    precision - tp/fp+tp
    recall - tp/tp+fn
    ROC-AUC
    PR-AUC
    '''
    results_file = f"{results_path}/clean_results.jsonl"
    results_path = makedir(results_path + "/plot_1")
    with open(results_file, "r") as r:
        
        # Read File
        results = jsonl_read(results_file)
        chain_number = len(results[0]["logs"])

        #Evaluate Which metrics want to be evaluated
        metrics_dict = {}
        for metric in threshold_dict:
            if metric in results[0]['logs'][0]['requests'][0]:
                metrics_dict[metric] = threshold_dict[metric]
            else:
                metrics_dict[metric] = [0,0]


        for i in range(chain_number):

                # Adjust responses for threshholds and plot PR Curves
            for key, val in metrics_dict.items():
        
                precision = []
                recall = []
                for j in range(int(val[0]), int(val[1])):
                    # Evaluate the actual answer and respones
                        # Evaluate the results 
                    for paper in results:
                        qa_list = paper["logs"][i]["requests"]
                        for q in qa_list:
                            answer = q["answer"]
                            response = q['response']
                            threshhold = float(q[key])
                            tp = 0
                            tn = 0
                            fp = 0
                            fn = 0

                            try:

                                if "NA" in answer and "NA" in response:
                                    tn+=1   

                                elif "NA" in response:
                                    if threshhold < j:
                                        tn +=1
                                    else:
                                        fp+=1

                                elif "NA" in answer:
                                    if threshhold < j:
                                        tn += 1
                                    else:
                                        fn+=1

                                elif "," in response:
                                    l_response = [float(i) for i in response.split(",")].sort()
                                    l_answer = [float(i) for i in answer.split(",")].sort()

                                    if l_answer == l_response:

                                        if threshhold < j:
                                            tp += 1
                                        else:
                                            fn+=1
                                    else:

                                        if threshhold < j:
                                            tn +=1
                                        else:
                                            fp+=1
                                else:

                                    if float(response) in [float(i) for i in answer.split(",")]:
                                        if threshhold < j:
                                            fn +=1
                                        else:
                                            tp+=1
                                    elif not(float(response) in [float(i) for i in answer.split(",")]):
                                        if threshhold < j:
                                            tn +=1
                                        else:
                                            fp+=1

                            except ValueError:
                                raise f"Answer:{answer} or Response{response}: is not a valid for this evaluation metric"

                    if fp > 0 or tp > 0: 
                        precision.append(tp/(fp+tp))
                        recall.append(tp/(tp+fn))
            

                fig, ax = plt.subplots(figsize = (7,7))
                ax.plot(precision, recall)
                ax.set_title(f"Precsion Recall Plot: {key}")
                ax.set_xlabel("Precision")
                ax.set_ylabel("Recall")
                fig.savefig(f"{results_path}/batch_PR_Plot_{key}.png")
                plt.close()


class File_handler():
    '''Helper Class created for neat file handling

        Attributes
        ----------

        File_handler.file : file
            The file object to be used with file.write and file.read
        
        File_handler.open : func
            Opens the file given in the instance
        
        File_handler.close : func
            Closes file for given instance

    '''
    def __init__(self, path):
        self.path = path
        self.isfileopen = False

    def open(self, open_type):
        self.file = open(self.path, open_type)

    def close(self):
        if self.isfileopen:
            self.file.close()
        self.file.flush()