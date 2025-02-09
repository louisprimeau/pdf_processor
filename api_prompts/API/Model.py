# Author : Jackson Dendy 
# Last Update : 12/16/2024
# Description : The File defines the class to be used when calling the API to do things.
# Desgined to streamline the use of the API.

import requests

class Model():
    '''Class that calls an API utlizing the LLamma 3.3 as a means to extract data from research papers
        
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
        '''
    def __init__(self, home, system):
        '''Initializes the API for the model in order to be prompted
        
            Parameters
            ----------
            home : str
                Local URL that the API is running on
            system : str
                System Prompt to assign a purpose to the model
            '''
        self.home = home
        self.system = system
        system = system.replace("\\", "vwvw")
        system = system.replace("/", "uquq")
        x = requests.get(f"{home}/activate_model/{system}").text
        if not(x == "True"):
            raise("Model not initilized properly")

    def upload(self, file):
        '''Uploads a local file text file to the API for analysis
            
            Parameters
            ----------
            
            file : str
                path of txt file you want to upload
                
            Returns
            ------
            r : str
                Answer/response to the file from the model
            '''
        file = file.replace("/", "uquq")
        r = requests.get(f"{self.home}/upload/{file}").text
        return r

    def request(self, message):
        '''Uploads a string to the API in a chat bot style
        
            Parameters
            ----------
            
            message : str
                Message you want to ask API
            
            Returns
            ------
            x : str
                Answer/response to the prompt from the model'''
        message = message.replace("\\", "vwvw")
        message = message.replace("/", "uquq")
        x = requests.get(f"{self.home}/request/{message}").text

        return x 

    def E2E(self, str1, str2):
        '''Evaluates the similarity of two strings using cosine similarity of embeded vectors

            Parameters
            ----------
            
            str1 : str
                string you want to be compared
            str2 : str
                string you want to be compared
            
            Returns
            ------
            score : str
                The cosine similarity of two strings
            '''
        str1 = str1.replace("/","uquq")
        str2 = str2.replace("/", "uquq")
        score = requests.get(f"{self.home}/E2E/{str1}/{str2}").text

        return score
    
    def zero_shot(self, message):
        '''Evaluates the similarity of two strings on a scale of 0-100 through prompting the model

            Parameters
            ----------
            
            message : str
                Each string you want compare in the form "String 1 is {answer} and String 2 is {response}"
            
            Returns
            ------
            x : int
                Score of similarity from 0-100, higher is more similar
            '''
        message = message.replace("/", "uquq")
        x = requests.get(f"{self.home}/zero_shot/{message}").text
        return x
        
    def clear_sys(self):
        '''Clears all message history including system prompt in the model
            
            Returns
            -------
            None
            '''
        requests.get(f"{self.home}/clear_sys")

    def clear(self):
        '''Clears everything except the system prompt in the model
            
            Returns
            -------
            None
            '''
        requests.get(f"{self.home}/clear")
    
    def clear_chain(self, length):
        '''Clears everything except the system prompt and prompt chain in the model

            Parameters
            ----------

            length : int
                how many prompts are in the chain that you want to avoid clearing
            
            Returns
            -------
            None
            '''
        requests.get(f"{self.home}/clear_chain/{length}")
            
    def getmessages(self):
        '''Return all message history of the model
            
            Returns
            -------
            None
            '''
        x = requests.get(f"{self.home}/getmessages").text
        return x
