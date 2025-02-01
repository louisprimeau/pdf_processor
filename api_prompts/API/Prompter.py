from API.utility import jsonl_read, makedir, E2E, File_handler
from API.Model import Model
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import WhitespaceTokenizer
from rouge_score import rouge_scorer
import os, json

class Prompter(Model):
    '''Child class of Model. Prompts the model with chains and questions based on dictionarys

        Attributes
        ----------

        Prompter.query : func
            runs the evaluation for an LLM
    '''

    def __init__(self, home, sys, question_file, results_path, paper_dir, r_scores = True, s_retrieval = True, LLM_zeroshot = True, E2E = True, chain_file = None):
        '''Initializes the Prompter class
            
            Parameters
            ----------
            
            home : str
                Local URL that the API is running on
            system : str
                System Prompt to assign a purpose to the model
            chain_file : str
                filepath to a .jsonl file containing all chains in the form {chain : "#", prompts : [list, of, questions, in , order]}
            question_file : str
                filepath to a .jsonl file containing all questions in the form 
                {"doi": "paper_relative_path", "journal_name": "", "messages": [{"question": "", "answer": ""}, {"question": "", "answer": ""}]}
            results_path : str
                path of an empty directory where results will be stored
            paper_dir : str
                path to all papers named in the dois in the questions_dict
            r_scores : bool
                if true adds rouge scores to results
            s_retrieval : bool
                if true adds sentence retrival metric to results
            LLM_zeroshot : bool
                if true adds LLM eval to results
            E2E : bool
                if true adds E2E evaluation to results
        '''
        L = Model(home, sys)
        self.__dict__ = L.__dict__.copy()

        if chain_file == None:
            self.chains = [{"chain": 0, "prompts": []}]
        else:
            self.chains = chains = jsonl_read(chain_file)
            
        self.questions= questions = jsonl_read(question_file)
        self.paper_dir = paper_dir

        results_path = makedir(results_path)

        with open(f"{results_path}/sys.txt", "a") as f:
            f.write(self.system)
            f.close()

        self.results = File_handler(f"{results_path}/results.jsonl")
        self.errors = File_handler(f"{results_path}/errors.txt")
        print(f"Results can be found at {results_path}")
        self.r_score_bool = r_scores
        self.s_retrieval_bool = s_retrieval
        self.LLM_zeroshot_bool = LLM_zeroshot
        self.E2E_bool = E2E

    def dynamic_sentence_retrival_chain(self):
        '''Recursivly asks the model for an answer until it can find the sentence the answer came from'''
        None

    def query(self):
        '''Runs the questions and chains in to the model against the papers provided'''

        # Querys API with questions
        for i, val in enumerate(self.questions):

            # Opens Files again to prevent atomic bomb (total data loss)
            self.results.open("a")
            self.errors.open("a")

            # Clears all prompts except system
            print(f"Beginning with {val['doi']}")
            self.clear_sys()

            qs = val['messages']

            # Intializes the data point in jsonl
            entry = {"paper": val['doi'], "logs": []}
            filepath = f"{self.paper_dir}/{val['doi']}"
            
            # Uploads file to API
            state = self.upload(f"{filepath}/text_converted.txt")
            if state == "True":
                    print("Uploaded Succesfully")
                    paper = open(f"{filepath}/text_converted.txt", "r")
                    paper_string = paper.read()
            else:
                state = self.upload(f"{filepath}/text.txt")
                if state == "True":
                    print("File Succesfully uploaded")
                    paper = open(f"{filepath}/text.txt", "r")
                    paper_string = paper.read()
                else:
                    self.errors.file.write(filepath)
                    raise "File Failed"
            
            paper.close()
            
            # Large nested loops of order Chains(prompts -> questions(points))
            for c in self.chains:

                # Clears everything except the first 2 entrys (paper and sys) from API cache
                self.clear_chain(1)
                prompts = c['prompts']
                chain_num = c["chain"]
                point = {"num": c["chain"], "chain": [], "requests": []}
                print(f"Initializing Chain {chain_num}")

                # Asks prompts and stores data
                for p in prompts:
                    response = self.request(p)
                    chain_point = {"prompt": p, "response": response}
                    point['chain'].append(chain_point)

                # Asks questions and stores data
                for q in qs:

                    # Clears previous question for API cache
                    self.clear_chain(len(prompts))
                    print("Asking Question")

                    response = self.request(q['question']) 
                    print("Question Answered")
                    answer = q['answer']

                    result_point = {"question": q['question'], "answer": q['answer'], "response": response,}
                    
                    # Direct Rating Confidence Score
                    confidence = self.request("On a scale of 0-100 evaluate your confidence in the accuracy of your response. Only return an integer.")
                    result_point['confidence'] = confidence
                    
                    # Sentence Retrival
                    if self.s_retrieval_bool:
                        sentence = self.request("Return the sentence you retrieved the answer to the question from. Only display the sentence and no other tokens in the response. Return the exact string.")
                        try:
                            paper_string.index(sentence)
                            sentence_bool = True
                        except ValueError:
                            sentence_bool = False
                        result_point['sentence_bool'] = sentence_bool
                        result_point['sentence'] = sentence

                    # Rouge Score
                    if self.r_score_bool:
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                        scores = scorer.score(q["answer"], response)
                        rouge2 = scores['rouge2'].fmeasure
                        rougeL = scores['rougeL'].fmeasure
                        result_point['rouge2'] = rouge2
                        results_point['rougeL'] = rougeL

                    # reference: https://arxiv.org/abs/2308.04624
                    #E2E
                    if self.E2E_bool:
                        E2E_score = self.E2E(answer, response)
                        result_point['E2E'] = E2E_score

                    # LLM Score
                    if self.LLM_zeroshot_bool:
                        llm_assesment = self.zero_shot(f"String 1 is {answer} and String 2 is {response}")
                        result_point['LLM'] = llm_assesment
                        

                    point['requests'].append(result_point)
                    
                entry['logs'].append(point)

            self.results.file.write(json.dumps(entry) + "\n")
            self.results.close()
            self.results.close()

            print(f"Finished with {val['doi']}\n")

        
    
    