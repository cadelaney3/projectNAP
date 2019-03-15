import requests

class Azure_API:
    def __init__(self, headers, text):
        self.doc = { 'documents' : [
            { 'id' : 1, 'language' : 'en', 'text' : text },
        ] }
        self.headers = headers

    def sentiment(self):
        
        url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment'
        azure_response  = requests.post(url, headers=self.headers, json=self.doc)
        sentiment = azure_response.json()
        sentiment = sentiment['documents'][0]['score']

        sent_dict = {'sentiment': sentiment}
        return sent_dict

    def entities(self):
        
        url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.1-preview/entities'
        azure_response  = requests.post(url, headers=self.headers, json=self.doc)   
        entities = azure_response.json()

        ents = []
        for item in entities['documents']:
                for i in item['entities']:
                    ents.append(i['name'].lower())
        
        ents.sort()
        return ents

    def keyphrases(self):
        
        url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/keyPhrases'
        
        azure_response  = requests.post(url, headers=self.headers, json=self.doc)    
        azure_kps = azure_response.json()

        keyPhrases = []
        for phrase in azure_kps['documents'][0]['keyPhrases']:
                keyPhrases.append(phrase.lower())

        keyPhrases.sort()
        return keyPhrases