from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.widgets import TextArea
from flask_bootstrap import Bootstrap
import requests
import json
from pprint import pprint
import sys
import os
import boto3
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
# IBM imports
import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features #, CategoriesOptionsImport, json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
    import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions, EmotionOptions, RelationsOptions, SemanticRolesOptions, SentimentOptions, CategoriesOptions
# Imports the Google Cloud client library
import six
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


with open('./constants.json') as f:
    CONSTANTS = json.load(f)


AZURE_KEY = CONSTANTS['AZURE_CREDENTIALS']['AZURE_KEY']
GOOGLE_APPLICATION_CREDENTIALS = CONSTANTS['GOOGLE_CREDENTIALS']
IBM_APIKEY = CONSTANTS['IBM_CREDENTIALS']['IBM_APIKEY']
IBM_URL = CONSTANTS['IBM_CREDENTIALS']['IBM_URL']
AWS_ACCESS_KEY = CONSTANTS['AWS_CREDENTIALS']['AWSAccessKeyId']
AWS_SECRET_KEY = CONSTANTS['AWS_CREDENTIALS']['AWSSecretKey']
DEEP_AI_KEY = CONSTANTS['DEEP_AI_CREDENTIALS']['DEEP_AI_KEY']

AWSAccessKeyId=AWS_ACCESS_KEY
AWSSecretKey=AWS_SECRET_KEY

credential_path = "./google_creds.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

text_analytics_base_url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/'

azure_headers   = {"Ocp-Apim-Subscription-Key": AZURE_KEY, 'Content-Type': 'application/json', 'Accept': 'application/json',}

DEBUG = True
application = Flask(__name__)
#Bootstrap(app)

application.config.from_object(__name__)
application.config['SECRET_KEY'] = CONSTANTS['FLASK_SECRET_KEY']['SECRET_KEY']

# Instantiates a client
client = language.LanguageServiceClient()

comprehend = boto3.client(service_name='comprehend', aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY, region_name='us-west-2')


naturalLanguageUnderstanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey=IBM_APIKEY,
    url=IBM_URL
)

class ReusableForm(Form):
    textbox = TextAreaField('text:', validators=[validators.required()])


def g_sentiment(text):
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(
        content=text.encode('utf-8'),
        type=enums.Document.Type.PLAIN_TEXT)

    google_sentiment = client.analyze_sentiment(document).document_sentiment
    sent = {}
    sent['sentiment'] = google_sentiment.score
    sent['magnitude'] = google_sentiment.magnitude
    return sent

def g_entities(text):
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(
        content=text.encode('utf-8'),
        type=enums.Document.Type.PLAIN_TEXT)

    google_entities = client.analyze_entities(document).entities
    
    entities = []
    for entity in google_entities:
        entities.append(entity.name.lower())

    entities.sort()
    return entities


def g_entity_sentiment(text):
    """Detects entity sentiment in the provided text."""
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(
        content=text.encode('utf-8'),
        type=enums.Document.Type.PLAIN_TEXT)

    # Detect and send native Python encoding to receive correct word offsets.
    encoding = enums.EncodingType.UTF32
    if sys.maxunicode == 65535:
        encoding = enums.EncodingType.UTF16

    result = client.analyze_entity_sentiment(document, encoding)

    entities = {}
    for entity in result.entities:
        entity_str = ""
        entity_str += 'Mentions: '
        entity_str += (u'Name: "{}"'.format(entity.name))
        name = entity.name
        entities[name] = entity.sentiment

    return entities

def g_syntax(text):
    """Detects syntax in the text."""
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Instantiates a plain text document.
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects syntax in the document. You can also analyze HTML with:
    #   document.type == enums.Document.Type.HTML
    tokens = client.analyze_syntax(document).tokens

    # part-of-speech tags from enums.PartOfSpeech.Tag
    pos_tag = ('UNKNOWN', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM',
               'PRON', 'PRT', 'PUNCT', 'VERB', 'X', 'AFFIX')

    result = []
    for token in tokens:
        result.append((u'{}: {}'.format(pos_tag[token.part_of_speech.tag],
                               token.text.content)))
    
    #print("g syntax: ", result)
    return result

def g_categories(text):
    """Classifies content categories of the provided text."""
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(
        content=text.encode('utf-8'),
        type=enums.Document.Type.PLAIN_TEXT)

    categories = client.classify_text(document).categories

    result = []
    for category in categories:
        result.append(category.name)

    return result

def azure_sentiment(text):
    json_tbox = { 'documents' : [
        { 'id' : 1, 'language' : 'en', 'text' : text },
    ] }
    url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment'
    azure_response  = requests.post(url, headers=azure_headers, json=json_tbox)
    sentiment = azure_response.json()
    sentiment = sentiment['documents'][0]['score']

    sent_dict = {'sentiment': sentiment}
    return sent_dict

def azure_entities(text):
    json_tbox = { 'documents' : [
        { 'id' : 1, 'language' : 'en', 'text' : text },
    ] }
    url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.1-preview/entities'
    azure_response  = requests.post(url, headers=azure_headers, json=json_tbox)   
    entities = azure_response.json()

    ents = []
    for item in entities['documents']:
            for i in item['entities']:
                ents.append(i['name'].lower())
    
    ents.sort()
    return ents

def azure_keyphrases(text):
    json_tbox = { 'documents' : [
        { 'id' : 1, 'language' : 'en', 'text' : text },
    ] }
    url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/keyPhrases'
    
    azure_response  = requests.post(url, headers=azure_headers, json=json_tbox)    
    azure_kps = azure_response.json()

    keyPhrases = []
    for phrase in azure_kps['documents'][0]['keyPhrases']:
            keyPhrases.append(phrase.lower())

    #print("azure kps: ", keyPhrases)
    keyPhrases.sort()
    return keyPhrases

def aws_sentiment(text):
    sentiments = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    sent_dict = {}
    sent_dict['pos_sentiment'] = sentiments['SentimentScore']['Positive']
    sent_dict['neg_sentiment'] = sentiments['SentimentScore']['Negative']
    sent_dict['neut_sentiment'] = sentiments['SentimentScore']['Neutral']
    
    return sent_dict

def aws_entities(text):
    entities = comprehend.detect_entities(Text=text, LanguageCode='en')
    ents = []
    for entity in entities['Entities']:
        ents.append(entity['Text'].lower())
    
    ents.sort()
    return ents

def aws_keyphrases(text):
    keyphrases = comprehend.detect_key_phrases(Text=text, LanguageCode='en')
    kps = []
    for phrase in keyphrases['KeyPhrases']:
        kps.append(phrase['Text'].lower())

    kps.sort()
    return kps

def aws_syntax(text):
    syntax = comprehend.detect_syntax(Text=text, LanguageCode='en')
    batch = []
    for word in syntax['SyntaxTokens']:
        batch.append([word['Text'], word['PartOfSpeech']['Tag']])

    return batch

def IBM_sentiment(text):
    IBM_response = naturalLanguageUnderstanding.analyze(
        text=text,
        features=Features(
            sentiment=SentimentOptions()
            )).get_result()  
    sentiment = IBM_response['sentiment']['document']['score']
    sentiment_label = IBM_response['sentiment']['document']['label']

    sent_dict = {'sentiment': sentiment}
    return sent_dict

def IBM_entities(text):
    IBM_response = naturalLanguageUnderstanding.analyze(
    text=text,
    features=Features(
        entities=EntitiesOptions(emotion=True, sentiment=True, limit=10)
        )).get_result()

    result = []
    ents = IBM_response['entities']
    print(ents)

    for e in ents:
        result.append(e['text'].lower())
 
    result.sort()
    print(result)
    return result

def IBM_keywords(text):
    IBM_response = naturalLanguageUnderstanding.analyze(
        text=text,
        features=Features(
            keywords=KeywordsOptions(emotion=True, sentiment=True,limit=10),
            )).get_result()
    kws = []
    for keyword in IBM_response['keywords']:
        kws.append(keyword['text'].lower())
 
    kws.sort()
    return kws

def IBM_categories(text):
    IBM_response = naturalLanguageUnderstanding.analyze(
        text=text,
        features=Features(
            categories=CategoriesOptions()
            )).get_result()
    cats = []
    for category in IBM_response['categories']:
        cats.append(category['label'])
    #print("ibm cats: ", IBM_response)
    return cats

def IBM_concepts(text):
    IBM_response = naturalLanguageUnderstanding.analyze(
        text=text,
        features=Features(
            entities=EntitiesOptions(emotion=True, sentiment=True, limit=10),
            keywords=KeywordsOptions(emotion=True, sentiment=True,limit=10),
            concepts=ConceptsOptions(limit=10),
            semantic_roles=SemanticRolesOptions(keywords=True, entities=True),
            relations=RelationsOptions(),
            sentiment=SentimentOptions(),
            categories=CategoriesOptions()
            )).get_result()

def deep_ai_sum(text):
    import requests
    r = requests.post(
        "https://api.deepai.org/api/summarization",
        data={
            'text': text,
        },
        headers={'api-key': DEEP_AI_KEY}
    )

    output = r.json()
    summary = output['output']
    return summary


@application.route('/', methods=['GET', 'POST'])
def hello_world():

    form = ReusableForm(request.form)
    print(form.errors)

    init_dict = {'sentiment': {'sentiment': 0.0, 'magnitude': 0.0, 'neg_sentiment': 0.0,
                 'pos_sentiment': 0.0, 'neg_sentiment': 0.0}, 'entities': [],
                 'keyphrases': [], 'categories': [], 'syntax': [], 'summary': '',
                 'keywords': []}



    dummy_dict = {}
    dummy_dict['sentiment'] = {'sentiment': 100.00, 'magnitude': 100.00, 'pos_sentiment': 100.00,
                                'neg_sentiment': 100.00, 'neut_sentiment': 100.00}
    dummy_dict['entities'] = ['dummy', 'dimmy', 'dommy', 'dammy']
    dummy_dict['keyphrases'] = ['phrase is key', 'the key is phrasing']
    dummy_dict['categories'] = ['fake', 'data']
    dummy_dict['syntax'] = [['dummy', 'NOUN']]
    dummy_dict['summary'] = "This is a dummy summary"
    dummy_dict['keywords'] = ['dummy', 'words']

    google_dict = {}
    azure_dict = {}
    amazon_dict = {}
    ibm_dict = {}
    deep_ai_dict = {}

    if request.method == 'POST':
        
        textbox = request.form['textbox']
 
    if form.validate():
        
        google_document = types.Document(
            content=textbox,
            type=enums.Document.Type.PLAIN_TEXT)

        thread_dict = {}
        sub_dict = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            google_sub_dict = {}
            google_sub_dict['sentiment'] = executor.submit(g_sentiment, textbox).result()
            google_sub_dict['entities'] = executor.submit(g_entities, textbox).result()
            google_sub_dict['categories'] = executor.submit(g_categories, textbox).result()
            google_sub_dict['syntax'] = executor.submit(g_syntax, textbox).result()

            azure_sub_dict = {}
            azure_sub_dict['sentiment'] = executor.submit(azure_sentiment, textbox).result()
            azure_sub_dict['entities'] = executor.submit(azure_entities, textbox).result()
            azure_sub_dict['keyphrases'] = executor.submit(azure_keyphrases, textbox).result()

            amazon_sub_dict = {}
            amazon_sub_dict['sentiment'] = executor.submit(aws_sentiment, textbox).result()
            amazon_sub_dict['entities'] = executor.submit(aws_entities, textbox).result()
            amazon_sub_dict['keyphrases'] = executor.submit(aws_keyphrases, textbox).result()
            amazon_sub_dict['syntax'] = executor.submit(aws_syntax, textbox).result()
 
            ibm_sub_dict = {}
            ibm_sub_dict['sentiment'] = executor.submit(IBM_sentiment, textbox).result()
            ibm_sub_dict['entities'] = executor.submit(IBM_entities, textbox).result()
            ibm_sub_dict['keywords'] = executor.submit(IBM_keywords, textbox).result()
            ibm_sub_dict['categories'] = executor.submit(IBM_categories, textbox).result()
            
            deep_ai_sub_dict = {}
            deep_ai_sub_dict['summary'] = executor.submit(deep_ai_sum, textbox).result()

        thread_dict['google'] = google_sub_dict
        thread_dict['azure'] = azure_sub_dict
        thread_dict['amazon'] = amazon_sub_dict
        thread_dict['ibm'] =  ibm_sub_dict
        thread_dict['deep_ai'] = deep_ai_sub_dict

        google_dict = thread_dict['google']
        azure_dict = thread_dict['azure']
        amazon_dict = thread_dict['amazon']
        ibm_dict = thread_dict['ibm']
        deep_ai_dict = thread_dict['deep_ai']
 
    else:
        flash('Enter text to be processed:')

    if (google_dict == {} or azure_dict == {} or amazon_dict == {} or 
        ibm_dict == {} or deep_ai_dict == {}):
        return render_template('main.html', form=form, google_dict=init_dict, azure_dict=init_dict, amazon_dict=init_dict,
                                ibm_dict=init_dict, deep_ai_dict=init_dict)
    else:
        return render_template('main.html', form=form, google_dict=google_dict, azure_dict=azure_dict, amazon_dict=amazon_dict,
                           ibm_dict=ibm_dict, deep_ai_dict=deep_ai_dict)

if __name__ == "__main__":
    application.run(host='0.0.0.0')
