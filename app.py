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
azure_sentiment_url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment'
azure_entities_url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.1-preview/entities'
azure_key_phrases_url = 'https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/keyPhrases'

headers   = {"Ocp-Apim-Subscription-Key": subscription_key, 'Content-Type': 'application/json', 'Accept': 'application/json',}

DEBUG = True
app = Flask(__name__)
Bootstrap(app)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


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
text = "It is raining today in Seattle"

print('Calling DetectSentiment')
test = json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='en'), sort_keys=True, indent=4)
sent_score = test[1]
print(sent_score)
print('End of DetectSentiment\n')

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

    return result

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    form = ReusableForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        textbox = request.form['textbox']
        json_tbox = { 'documents' : [
            { 'id' : 1, 'language' : 'en', 'text' : textbox },
        ] }
        print(textbox)
    if form.validate():

        google_document = types.Document(
            content=textbox,
            type=enums.Document.Type.PLAIN_TEXT)

        # Detects the sentiment of the text
        google_sentiment = client.analyze_sentiment(document=google_document).document_sentiment
        google_entities = client.analyze_entities(google_document).entities
        google_entity_sent = g_entity_sentiment_text(textbox)
        google_syntax = g_syntax_text(textbox)
        google_classify = g_classify_text(textbox)

        azure_response  = requests.post(azure_sentiment_url, headers=headers, json=json_tbox)
        azure_sentiments = azure_response.json()
        azure_sentiment_score = azure_sentiments['documents'][0]['score']
        azure_ent_resp = requests.post(azure_entities_url, headers=headers, json=json_tbox)
        azure_entities = azure_ent_resp.json()
        azure_keyPhrase_resp = requests.post(azure_key_phrases_url, headers=headers, json=json_tbox)
        azure_key_phrases = azure_keyPhrase_resp.json()

        aws_sentiment = comprehend.detect_sentiment(Text=textbox, LanguageCode='en')
        aws_entities = comprehend.detect_entities(Text=textbox, LanguageCode='en')
        aws_key_phrases = comprehend.detect_key_phrases(Text=textbox, LanguageCode='en')
        aws_syntax = comprehend.detect_syntax(Text=textbox, LanguageCode='en')

        print("aws entities: ", aws_entities)

        IBM_response = naturalLanguageUnderstanding.analyze(
            text=textbox,
            features=Features(
                entities=EntitiesOptions(emotion=True, sentiment=True, limit=10),
                keywords=KeywordsOptions(emotion=True, sentiment=True,limit=10),
                concepts=ConceptsOptions(limit=10),
                semantic_roles=SemanticRolesOptions(keywords=True, entities=True),
                relations=RelationsOptions(),
                sentiment=SentimentOptions(),
                categories=CategoriesOptions()
                )).get_result()
        print(json.dumps(IBM_response, indent=2))

        flash('Text: ' + textbox)
        flash('Google: Sentiment: {}, Magnitude: {}'.format(google_sentiment.score, google_sentiment.magnitude))
        flash('Azure: Sentiment: ' + str(azure_sentiment_score))
        flash('AWS: Positive Sentiment: ' + str(aws_sentiment['SentimentScore']['Positive']) + '\n' +
              'Negative Sentiment: ' + str(aws_sentiment['SentimentScore']['Negative']))
        flash('IBM Sentiment: ' + str(IBM_response['sentiment']['document']))
        
        for entity in google_entities:
            flash_string = ""
            entity_type = enums.Entity.Type(entity.type)
            flash_string += (u'{:<16}: {}; '.format('name', entity.name))
            flash_string += (u'{:<16}: {}; '.format('type', entity_type.name))
            flash_string += (u'{:<16}: {}; '.format('salience', entity.salience))
            flash_string += (u'{:<16}: {}; '.format('wikipedia_url',
                entity.metadata.get('wikipedia_url', '-')))
            flash_string += (u'{:<16}: {}; '.format('mid', entity.metadata.get('mid', '-')))
            flash("Google entity: " + flash_string)

        #print(azure_entities)
        
        for item in azure_entities['documents']:
            for i in item['entities']:
                #print(i)
                #print(i['matches'])
                if 'type' in i:
                    #print(i['type'])
                    flash("Azure Entity: " + i['name'] + " Type: " + i['type'])
                else:
                    flash("Azure Entity: " + i['name'])

        phrase_str = ""
        for phrases in azure_key_phrases['documents'][0]['keyPhrases']:
            phrase_str += phrases + "; "
        flash("Azure key phrases: " + phrase_str)

        for word in IBM_response['keywords']:
            flash("IBM keyword: " + str(word['text']) + "; relevance: " + str(word['relevance']))
        
        for ent in google_entity_sent:
            flash("Google entity sentiment: " + ent)
        
        for syn in google_syntax:
            flash("Google syntax: " + syn)

        for c in google_classify:
            flash("Google classification: " + c)

        for cat in IBM_response['categories']:
            flash("IBM category: " + str(cat['label']) + "; score: " + str(cat['score']))

    else:
        flash('All the form fields are required')

    return render_template('home.html', form=form)

if __name__ == "__main__":
    app.run()
