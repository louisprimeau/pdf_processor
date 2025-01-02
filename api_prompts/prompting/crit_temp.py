from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
root = str(path_root)
sys.path.insert(1, root)

from API import Prompter




sys = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Be very concise and avoid wordy responses. Give all responses in a single number with no other charecters."""

P = Prompter(
            home = "http://127.0.0.1:7777",
            sys = sys, 
            question_file = '/home/jdendy/pdf_processor/api_prompts/questions/crit_temp_questions.jsonl', 
            results_path = '/home/jdendy/pdf_processor/api_prompts/prompting/results/crit_temp_data_1',
            paper_dir =  '/home/jdendy/pdf_processor/api_prompts/questions/test',
            E2E = False,
            r_scores = False
            )

P.query()