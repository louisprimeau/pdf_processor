
from Louis import Louis
import time
import json, os

sys = """You are an assistant for answering questions. You are given the extracted parts of a long document and a question. Don't make up an answer. Here is the document: """

# Calls the API I created 
API = Louis("http://127.0.0.1:7777", sys)
API.request("Who are you")
#time.sleep(1)
API.clearish()

#time.sleep(1)
print(API.getmessages())
