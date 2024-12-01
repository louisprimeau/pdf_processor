import torch
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


def makedir(path):
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
    j = open(file, "r")

    questions = []
    for i in j.readlines():
        try:
            x = json.loads(i)
            x['doi'] = x['doi'][8:]
        except:
            x = json.loads(i)
        questions.append(x)

    j.close()
    return questions

def E2E(str1, str2, model):
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