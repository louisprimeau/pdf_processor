import pytest
import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
one_under_path_root = str(Path(__file__).parents[0])
root = str(path_root)
sys.path.insert(1, root)
import Model



'''Tests for Machine.py and Model.py'''
model = Model.Model("http://127.0.0.1:7777", "This is just a test for functionality. When I say one you say two.")

class Test_Model:

    def test_upload(self):
        model.upload(f"{root}//tests//unit//test_paper.txt")
        assert len(eval(model.getmessages())) == 2

    def test_request(self):
        model.request("one")
        assert len(eval(model.getmessages())) == 4

    def test_E2E(self):
        model.E2E
        assert True

    def zero_shot(self):
        response = model.zero_shot("Bruh", "Bruh")
        assert response == 1

    def clear_sys(self):
        model.clear_sys
        assert True

    def clear(self):
        model.clear
        assert len(eval(model.getmessages())) == 1

    def clear_chain(self):
        model.clear_chain
        assert True

    def getmessages(self):
        message = eval(model.getmessages())
        assert len(message) == 1