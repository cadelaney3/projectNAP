from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
    import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions, EmotionOptions, RelationsOptions, SemanticRolesOptions, SentimentOptions, CategoriesOptions
from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback, AudioSource
from os.path import join, dirname
import json

default_url = "https://stream.watsonplatform.net/speech-to-text/api"
web_socket = "wss://stream.watsonplatform.net/speech-to-text/api/v1/recognize"

class IBM_API:
    def __init__(self, headers, text):
        self.naturalLanguageUnderstanding = headers
        self.text = text

    def concepts(self):
        IBM_dict = {}
        IBM_response = self.naturalLanguageUnderstanding.analyze(
            text=self.text,
            features=Features(
                entities=EntitiesOptions(emotion=True, sentiment=True, limit=10),
                keywords=KeywordsOptions(emotion=True, sentiment=True,limit=10),
                sentiment=SentimentOptions(),
                categories=CategoriesOptions()
                )).get_result()

        sent_dict = {'sentiment': IBM_response['sentiment']['document']['score']}
        IBM_dict['sentiment'] = sent_dict
        
        ent_result = []
        ents = IBM_response['entities']
        for e in ents:
            ent_result.append(e['text'].lower())
        ent_result.sort()
        IBM_dict['entities'] = ent_result
        
        kws = []
        for keyword in IBM_response['keywords']:
            kws.append(keyword['text'].lower())
        kws.sort()
        IBM_dict['keywords'] = kws
        
        cats = []
        for category in IBM_response['categories']:
            cats.append(category['label'])
        IBM_dict['categories'] = cats
        
        return IBM_dict

class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)
    
    def on_data(self, data):
        print(json.dumps(data, indent=2))

    def on_error(self, error):
        print("Error received: {}".format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

class IBM_transcribe:
    def __init__(self, speech_to_text):
        self.speech_to_text = speech_to_text

    def transcribe(self, audio, content_type):
        try:
            response = self.speech_to_text.recognize(
                audio=audio,
                content_type='audio/' + content_type,
                max_alternatives=2
            ).get_result()
            return response['results'][0]['alternatives'][0]['transcript']
        except Exception as e:
            print(e)
