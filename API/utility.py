import os, json


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