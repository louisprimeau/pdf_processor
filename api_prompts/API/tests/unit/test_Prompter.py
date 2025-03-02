# Author: Jackson Dendy
# Description: Unit Tests for the Prompter class.

import pytest, sys, torch, shutil
from pathlib import Path
path_root = Path(__file__).parents[2]
one_under_path_root = str(Path(__file__).parents[0])
root = str(path_root)
sys.path.insert(1, root)
import Prompter
from utility import jsonl_read

ob = Prompter.Prompter( home = "http://127.0.0.1:7777", 
                        sys = "This is just a test for functionality. When I say one you say two. Only respond with the word two and do not include any other tokens in your response.", 
                        question_file = f"{root}//tests//unit//help_files//test_question.jsonl", 
                        results_path = f"{root}//tests//unit//help_files//results//test_data_1", 
                        paper_dir = f"{root}//tests//unit//help_files//test_papers", 
                        chain_file = f"{root}//tests//unit//help_files//test_chain.jsonl")
class Test_Prompter:

    def test_query(self):
        ob.query()
        data = jsonl_read(f"{root}//tests//unit//help_files//results//test_data_1//results.jsonl")
        state = False
        key_list = ["paper", "logs", "num", "chain", "prompt", "requests", "answer", "response", "LLM", "rouge2", "rougeL", "sentence_bool", "confidence", "sentence", "E2E"]
        if len(data) == 0:
            assert state
        else:
            for i in data[0].keys():
                if i not in key_list:
                    assert state
                if i == "logs":
                    for j in data[0].i[0].keys():
                        if j not in key_list:
                            assert state
                        if j == "requests":
                            for k in data.i[0].j[0].keys():
                                if k not in key_list:
                                    assert state
                        elif i == "chain":
                            for k in data[0].i[0].j[0].keys():
                                if j not in key_list:
                                    assert state
            
        shutil.rmtree(f"{root}//tests//unit//help_files//results//test_data_1")
        assert not(state)