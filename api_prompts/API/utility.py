# Author : Jackson Dendy 
# Last Update : 12/16/2024
# Description : Supporting functions for files in this folder

import torch
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


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

def E2E(str1, str2, model):
    '''!*Deprecated and moved to machine.py*! Beginning implemtation of cosine simlarity
        
        Paramters
        ---------
        
        str1 : str
            string wanting to be compared
        
        str2 : str
            string wanting to be compared
        
        model : UNKOWN
            Pipline for model 
            
        Returns
        -------
        None
        '''
    model_sig='8B'
    model_id = "meta-llama/Meta-Llama-3.1-{}-Instruct".format(model_sig)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    str1 = tokenizer.encode(str1)
    str2 = tokenizer.encode(str2)
    print(str1)

    output = model(str1)[0].squeeze()
    # only grab output of CLS token (<s>), which is the first token
    print(output[0])
    '''
    cos = torch.nn.CosineSimilarity(dim=1)
    a = 0
    print(cos(emb1.mean(axis=a), emb2.mean(axis=a)))'''


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