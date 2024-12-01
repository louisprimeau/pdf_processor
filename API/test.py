from Louis import Louis
from pathlib import Path
from utility import jsonl_read, makedir, E2E
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import WhitespaceTokenizer
from rouge_score import rouge_scorer
import os
'''
# Download necessary NLTK data
nltk.download('punkt_tab')

reference1 = "blah is the name of the game"
candidate1 = "blah blah blah"

tk = WhitespaceTokenizer()

reference = tk.tokenize(reference1)
candidate = tk.tokenize(candidate1)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference1, candidate1)
rouge1 = scores['rouge1'].fmeasure
rougeL = scores['rougeL'].fmeasure

print(reference)
print(candidate)
print(float(sentence_bleu([reference], candidate, weights =[1])))

print(rouge1)
print(rougeL)'''

sys = '''You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Here is the document: '''
chain_file="testchains.jsonl"
question_file="test.jsonl"
'''
# Initialize File Structure
test_name = "Test_1"

print("Files Initialized Contacting Louis")

# Calls the API I created 
API = Louis("http://127.0.0.1:7777", sys)

print(API.upload("/home/jdendy/pdf_processor/API/questions/test/physrevb.29.2664/text.txt"))

print(API.getmessages())'''

#print(os.path.exists("erverv"))


# Calls the API I created 
API = Louis("http://127.0.0.1:7777", sys)

print(API.E2E("yes", "no this is coming together now"))