import requests



class Louis():
    def __init__(self, home, system):
        self.home = home
        x = requests.get(f"{home}/activate_model/{system}").text
        if not(x == "True"):
            raise("Model not initilized properly")

    def upload(self, file):
        file = file.replace("/", "uquq")
        r = requests.get(f"{self.home}/upload/{file}").text
        return r

    def request(self, message):
        x = requests.get(f"{self.home}/request/{message}").text

        return x 
    
    def zero_shot(self, message):
        message = message.replace("/", "uquq")
        x = requests.get(f"{self.home}/zero_shot/{message}").text
        return x
        
    def clear_sys(self):
        requests.get(f"{self.home}/clear_sys")

    def clear(self):
        requests.get(f"{self.home}/clear")
    
    def clear_chain(self, length):
        requests.get(f"{self.home}/clear_chain/{length}")
            
    def getmessages(self):
        x = requests.get(f"{self.home}/getmessages").text
        return x



