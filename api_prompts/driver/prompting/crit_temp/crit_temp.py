from pathlib import Path
import sys

path_root = Path(__file__).parents[3]
root = str(path_root)
sys.path.insert(1, root)

from API import Prompter




sys = """You are given the extracted parts of a long documents and are expected to find the critical temperature of the requested material. Don't make up an answers and respond with NA if the paper contains insufficient information. Put your chain of thought in between you can be wordy with your thoughts but in your final answer provide one number. It is very important for your final answer to be one number with the proper unit."""

P = Prompter(
            home = "http://127.0.0.1:7777",
            sys = sys, 
            question_file = '/home/jdendy/pdf_processor/api_prompts/data/questions/crit_temp_questions.jsonl', 
            results_path = '/home/jdendy/pdf_processor/api_prompts/data/results/crit_temp_results/results/crit_temp_data_1',
            paper_dir =  '/home/jdendy/pdf_processor/api_prompts/data/papers/test',
            E2E = True,
            r_scores = False,
            s_retrieval = True,
            LLM_zeroshot = True,
            )

P.query()

