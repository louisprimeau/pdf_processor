from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
root = str(path_root)
sys.path.insert(1, root)

from API import Prompter, Louis




sys = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Be very concise and avoid wordy responses. """

P = Prompter(
            home = "http://127.0.0.1:7777",
            sys = sys, 
            question_file = '/home/jdendy/pdf_processor/API/questions/crit_temp_questions.jsonl', 
            chain_file ='/home/jdendy/pdf_processor/API/chains/testchains.jsonl',
            results_path = 'results/CT_results_1',
            paper_dir =  '/home/jdendy/pdf_processor/API/questions/test',
            E2E = False,
            r_scores = False
            )

P.query()