

General Workflow For Using the API package
------------------------------------------
------------------------------------------

1. Make sure to run machine.py on a seperate screen locally. This acts the API to call from the Model Class.

2. Initialize the Prompter Class and run the desired questions and chains against the desired papers.
    - The Workflow within the prompter class is as follows.
        1. All post-data storage is allocated along with loading the questions and chains
        2. In a nested loop fashion the class checks each chain against each paper. Asking the listed questions for each chain.
        This allows to test the efficacy of different chains.
        3. Query Comparison-AI and run evaluations on the model responses.
3. All post-processing is done outside the class and functions are located in the utility.py file. There is a batch post-processing that compares responses across many papers
and there is a single_paper processing that evaluates responses on one paper. (Neither of these currently work :|.)

Sub_Directory Descriptions
------------------------------
------------------------------

API/
-----------
The API package allows querying of the LLama 3.1 model in its current state (future versions may use deepseek and nemotron). The querying aims to extract material properties from a batch of papers.

    machine.py
    ----------
    When ran it loads the active model onto the system and allows the use of the Model class and its child classes.

    Model.py
    ---------
    Contains Model Class. Class that calls an API utlizing the LLamma 3.3 as a means to extract data from research papers
        
        Attributes
        ----------
        
        Model.home : str 
            Local URL that the API is running on
        Model.upload : function
            Uploads a local file text file to the API for analysis
        Model.request : function
            Uploads a string to the API in a chat bot style
        Model.E2E : function
            Evaluates the similarity of two strings using cosine similarity of embeded vectors
        Model.zero_shot : function
            Evaluates the similarity of two strings on a scale of 1-100 through prompting the model
        Model.clear_sys : function
            Clears all message history including system prompt in the model
        Model.clear : function
            Clears everything except the system prompt in the model
        Model.clear_chain : function
            Clears everything except the system prompt and prompt chain in the model
        Model.getmessages : function
            Return all message history of the model
        
    Promper.py
    -----------
    Child class of Model. Prompts the model with chains and questions based on dictionarys

        Attributes
        ----------

        Prompter.query : func
            runs the evaluation for an LLM

    utility.py
    -----------
    Contains helper functions and post processing functionality.

conditional_retrival/
----------------------
Creates test case data sets to assist in the implementation of model baselines. Currently the only test case data set it creates is the compilation of all critical temperature data 
for the materials in the set of papers. 

deprecated/
-----------------
Old Functions or versions of API.

prompting/
----------------
Contains the Driver code for running the prompt class. 

Ignored Directory
---------------------
---------------------

chains/
-----------
Filepath to a .jsonl file containing all chains in the form {chain : "#", prompts : [list, of, questions, in , order]}.

questions/
-------------
Filepath to a .jsonl file containing all questions in the form {"doi": "paper_relative_path", "journal_name": "", "messages": [{"question": "", "answer": ""}, {"question": "", "answer": ""}]}.


Future Plans
---------------
---------------

- Add dynamic chains for the direct sentence comparison metric. That is when the model says it found data from a sentence that doesnt exist it reprompts the model with the same chain + one question telling it that the sentence doesnt exist.
- Implement deepseek and nemotron to improve performance over llama
- Implement PR and AOC evaluation charts for the model. Also add other things for evaluation.
- Potentially use platt scaling to make the self evaluation confidence scores more meaningful: Need to do more research on what exactly platt scaling is and how to implement it.