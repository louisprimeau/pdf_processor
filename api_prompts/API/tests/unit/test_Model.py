# Author: Jackson Dendy
# Description: Unit Tests for the Model class.

import pytest
import sys
import torch
from pathlib import Path
path_root = Path(__file__).parents[2]
one_under_path_root = str(Path(__file__).parents[0])
root = str(path_root)
sys.path.insert(1, root)
import Model



'''Tests for Machine.py and Model.py'''
model = Model.Model("http://127.0.0.1:7777", "This is just a test for functionality. When I say one you say two. Only respond with the word two and do not include any other tokens in your response.")

class Test_Model:

    def test_getmessages(self):
        message = eval(model.getmessages())
        assert len(message) == 1

    def test_upload(self):
        model.upload(f"{root}//tests//unit//help_files//test_paper.txt")
        assert len(eval(model.getmessages())) == 2

    def test_request(self):
        model.request("one")
        assert len(eval(model.getmessages())) == 4

    def test_E2E(self):
        num = model.E2E("one", "one")
        assert "1.0" == num

    def test_zero_shot(self):
        response = model.zero_shot("The first string: Bruh and second string: Bruh")
        assert type(int(response)) == type(100)

    def test_clear_sys(self):
        model.clear_sys()
        assert eval(model.getmessages()) == []

    def test_clear(self):
        model.request(" ")
        model.clear()
        assert len(eval(model.getmessages())) == 1

    def test_clear_chain(self):
        model.request(" ")
        model.clear_chain(2)
        assert len(eval(model.getmessages())) == 1
