from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.widgets import TextArea
from flask_bootstrap import Bootstrap
import requests
import json
from pprint import pprint

subscription_key = 'ad9bd37163ed470cabc3d324f3d6ca5c'
subscription_key2 = '459ba66d7fe949cc8f38020a681ebfde'
assert subscription_key
assert subscription_key2

text_analytics_base_url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/'

azure_headers   = {"Ocp-Apim-Subscription-Key": subscription_key, 'Content-Type': 'application/json', 'Accept': 'application/json',}

#DEBUG = True
application = Flask(__name__)
#Bootstrap(app)

#application.config.from_object(__name__)
#application.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


import sys
import os

credential_path = "/home/cdswaggy/Downloads/My_First_Project-06f28cd27269.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# Imports the Google Cloud client library
import six
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

# Instantiates a client
client = language.LanguageServiceClient()

# [END language_sentiment_text]

AWSAccessKeyId='AKIAJEC4HYYBKZNCAMIA'
AWSSecretKey='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'

import boto3
comprehend = boto3.client(service_name='comprehend', aws_access_key_id=AWSAccessKeyId,
    aws_secret_access_key=AWSSecretKey, region_name='us-west-2')

####################################
IBM_APIKEY='fj6-S0o3vni_GPO9ARhV96ZAL_4YKf-D6c_XfRYROgmz'
IBM_URL='https://gateway.watsonplatform.net/natural-language-understanding/api'
######################################

import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features #, CategoriesOptionsImport, json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
    import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions, EmotionOptions, RelationsOptions, SemanticRolesOptions, SentimentOptions, CategoriesOptions

naturalLanguageUnderstanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey=IBM_APIKEY,
    url=IBM_URL
)

'''
response = naturalLanguageUnderstanding.analyze(
    text='IBM is an American multinational technology company '
    'headquartered in Armonk, New York, United States, '
    'with operations in over 170 countries.',
    features=Features(
        entities=EntitiesOptions(emotion=True, sentiment=True, limit=2),
        keywords=KeywordsOptions(emotion=True, sentiment=True,
                                 limit=2))).get_result()

print(json.dumps(response, indent=2))
'''

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
    #g_sent_score = google_sentiment.score
    #g_sent_mag = google_sentiment.magnitude
    #google_entity_sent = g_entity_sentiment_text(textbox)
    #google_syntax = g_syntax_text(textbox)
    #google_classify = g_classify_text(textbox)
    return google_sentiment

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
        entities.append(entity.name)

    #print("g ents: ", entities)
    return entities


def g_entity_sentiment_text(text):
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

    entities = []
    for entity in result.entities:
        entity_str = ""
        entity_str += 'Mentions: '
        entity_str += (u'Name: "{}"'.format(entity.name))
        for mention in entity.mentions:
            entity_str += (u'  Begin Offset : {}'.format(mention.text.begin_offset))
            entity_str += (u'  Content : {}'.format(mention.text.content))
            entity_str += (u'  Magnitude : {}'.format(mention.sentiment.magnitude))
            entity_str += (u'  Sentiment : {}'.format(mention.sentiment.score))
            entity_str += (u'  Type : {}'.format(mention.type))
        entity_str += (u'Salience: {}'.format(entity.salience))
        entity_str += (u'Sentiment: {}\n'.format(entity.sentiment))
        entities.append(entity_str)

    return entities

def g_syntax_text(text):
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

def g_classify_text(text):
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
        result_str = ""
        result_str += (u'{:<16}: {}'.format('name', category.name))
        result_str += (u'{:<16}: {}'.format('confidence', category.confidence))
        result.append(result_str)

    print("g categories: ", categories)
    return result

def azure_sentiment(text):
    json_tbox = { 'documents' : [
        { 'id' : 1, 'language' : 'en', 'text' : text },
    ] }
    url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment'
    azure_response  = requests.post(url, headers=azure_headers, json=json_tbox)
    sentiment = azure_response.json()
    sentiment = sentiment['documents'][0]['score']

    #print("azure sent: ", sentiment)
    return sentiment

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
                ents.append(i['name'])
    
    #print("azure ents: ", ents)
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
            keyPhrases.append(phrase)

    #print("azure kps: ", keyPhrases)
    return keyPhrases

def aws_entities(text):
    entities = comprehend.detect_entities(Text=text, LanguageCode='en')
    ents = []
    for entity in entities['Entities']:
        ents.append(entity['Text'])
    
    print(entities)
    return ents

def aws_keyphrases(text):
    keyphrases = comprehend.detect_key_phrases(Text=text, LanguageCode='en')
    kps = []
    for phrase in keyphrases['KeyPhrases']:
        kps.append(phrase['Text'])

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

    return sentiment

def IBM_entities(text):
    IBM_response = naturalLanguageUnderstanding.analyze(
    text=text,
    features=Features(
        entities=EntitiesOptions(emotion=True, sentiment=True, limit=10)
        )).get_result()
    ents = []
    
    #for entity in IBM_response['entities']:
     #   ents.append(entity['text'])
    entities = IBM_response['entities']

    print("ibm ents: ", entities)
    return ents

def IBM_keywords(text):
    IBM_response = naturalLanguageUnderstanding.analyze(
        text=text,
        features=Features(
            keywords=KeywordsOptions(emotion=True, sentiment=True,limit=10),
            )).get_result()
    kws = []
    for keyword in IBM_response['keywords']:
        kws.append(keyword['text'])
 
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

@application.route('/', methods=['GET', 'POST'])
def hello_world():

    form = ReusableForm(request.form)
    print(form.errors)
    google_dict = {}
    azure_dict = {}
    amazon_dict = {}
    ibm_dict = {}

    if request.method == 'POST':
        
        textbox = request.form['textbox']
        '''
        json_tbox = { 'documents' : [
            { 'id' : 1, 'language' : 'en', 'text' : textbox },
        ] }
        print(textbox)
        '''
    if form.validate():

        google_document = types.Document(
            content=textbox,
            type=enums.Document.Type.PLAIN_TEXT)

        # Detects the sentiment of the text
        #google_sentiment = client.analyze_sentiment(document=google_document).document_sentiment
        #g_sent_score = google_sentiment.score
        #g_sent_mag = google_sentiment.magnitude
        google_sentiment = g_sentiment(textbox)
        google_entities = g_entities(textbox)
        google_entity_sent = g_entity_sentiment_text(textbox)
        google_syntax = g_syntax_text(textbox)
        google_classify = g_classify_text(textbox)
        
        google_dict['sentiment'] = google_sentiment.score
        google_dict['magnitude'] = google_sentiment.magnitude
        google_dict['entities'] = google_entities
        google_dict['classify'] = google_classify
        google_dict['syntax'] = google_syntax

        #azure_response  = requests.post(azure_sentiment_url, headers=headers, json=json_tbox)
        azure_sent = azure_sentiment(textbox)
        # azure_sentiment_score = azure_sentiments['documents'][0]['score']
        azure_dict['sentiment'] = azure_sent
        #azure_ent_resp = requests.post(azure_entities_url, headers=headers, json=json_tbox)
        azure_ents = azure_entities(textbox)
        azure_dict['entities'] = azure_ents

        azure_key_phrases = azure_keyphrases(textbox)
        azure_dict['key_phrases'] = azure_key_phrases

        aws_sentiment = comprehend.detect_sentiment(Text=textbox, LanguageCode='en')
        amazon_dict['pos_sentiment'] = aws_sentiment['SentimentScore']['Positive']
        amazon_dict['neg_sentiment'] = aws_sentiment['SentimentScore']['Negative']
        amazon_dict['neut_sentiment'] = aws_sentiment['SentimentScore']['Neutral']

        aws_ents = aws_entities(textbox)
        amazon_dict['entities'] = aws_ents
        aws_key_phrases = aws_keyphrases(textbox)
        amazon_dict['keyphrases'] = aws_key_phrases
        aws_syn = aws_syntax(textbox)
        amazon_dict['syntax'] = aws_syn

        #print("aws entities: ", aws_entities)
        
        IBM_sent = IBM_sentiment(textbox)
        IBM_ents = IBM_entities(textbox)
        IBM_kws = IBM_keywords(textbox)
        IBM_cats = IBM_categories(textbox)

        ibm_dict['sentiment'] = IBM_sent
        ibm_dict['entities'] = IBM_ents
        ibm_dict['keywords'] = IBM_kws
        ibm_dict['categories'] = IBM_cats
        
    
    else:
        flash('All the form fields are required')

    #return render_template('home.html', form=form, google_dict=google_dict, azure_dict=azure_dict, amazon_dict=amazon_dict,
                           # ibm_dict=ibm_dict)
    return render_template('main.html', form=form, google_dict=google_dict, azure_dict=azure_dict, amazon_dict=amazon_dict,
                           ibm_dict=ibm_dict)

if __name__ == "__main__":
    application.run()
