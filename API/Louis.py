import requests



class Louis():
    def __init__(self, home, system):
        self.home = home
        x = requests.get(f"{home}/activate_model/{system}").text
        if not(x == "True"):
            raise("Model not initilized properly")

    def upload(self, file):
        file = open(file, "r").read()

        requests.get(f"{self.home}/request/{file}")

    def request(self, message):
        x = requests.get(f"{self.home}/request/{message}").text

        return x 

    def clearish(self):
        requests.get(f"{self.home}/clearish")

    def clear(self):
    
        requests.get(f"{self.home}/clear")
            
    def getmessages(self):
        x = requests.get(f"{self.home}/getmessages").text
        return x



