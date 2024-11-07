from Louis import Louis
from pathlib import Path
from utility import jsonl_read, makedir
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

print(len(os.listdir("/home/jdendy/pdf_processor/API/questions/test")))
print(167 * 30)