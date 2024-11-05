from Louis import Louis
import json, os

#TODO: Tasks
'''
- Add support for Chains
- Add support for Quantitatize loss to respond to answers
- Add support for PDF interpretation with the OCR
- BLEU score
- make 40 for prompts
'''

firectory = 'neifn'
sys = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Here is the document: """

# Calls the API I created 
API = Louis("http://127.0.0.1:7777", sys)

def makedir(path):
    while os.path.exists(path):
        index = path.split('_')
        bit = index[-1]
        index.pop(-1)
        path = [a + '_' for a in index]
        path = "".join(path)
        path += str(int(bit) + 1)
    os.mkdir(path)
    
# Creates the list of questions
def jsonl_read(file):
    j = open(file, "r")

    questions = []
    for i in j.readlines():
        x = json.loads(i)
        x['doi'] = x['doi'][8:]
        questions.append(x)

    j.close()
    return questions

# Initialize File Structure
test_name = "Test_1"
results_path = f"{firectory}/results/{test_name}"
makedir(results_path)

print("Files Initialized Contacting Louis")

questions = jsonl_read(f"{directory}/questions/test.jsonl")

# Querys API with questions
for i, val in enumerate(questions):

    qa = open(f'{results_path}/results.txt', "a")
    error = open(f'{results_path}/errors.txt', 'a')

    print(f"Beginning with {val['doi']}")
    API.clearish()

    qs = val['messages']

    entry = {'paper': val['doi'], 'chain': [] 'requests': []}
    filepath = f"{directory}questions/test/{val['doi']}"

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

    
    for q in qs:

        print("Asking Question")
        API.clearish()

        response = API.request(q['question']) 
        print("Question Answered")
        answer = q['answer']

        entry['requests'].append({'question': q['question'], 'answer': q['answer'], 'response': response})
    print(f"Finished with {val['doi']}\n")
    
    qa.write(f"{entry}\n")
    qa.close()
    error.close()