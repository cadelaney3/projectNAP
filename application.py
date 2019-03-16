from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.widgets import TextArea
from flask_bootstrap import Bootstrap
import requests
import json
import sys
import os
import boto3
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
# IBM imports
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features #, CategoriesOptionsImport, json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
    import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions, EmotionOptions, RelationsOptions, SemanticRolesOptions, SentimentOptions, CategoriesOptions

from application.google_api import Google_Cloud
from application.google_api import Google_ST
from application.google_api import MicStream
from application.azure_api import Azure_API
from application.aws_api import AWS_API
from application.ibm_api import IBM_API
from application.deep_ai_api import Deep_AI_API

with open('./constants.json') as f:
    CONSTANTS = json.load(f)

audio = os.path.join(
    os.path.dirname(__file__),
    './audio', 'meeting_15sec.wav'
)

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


AZURE_KEY = CONSTANTS['AZURE_CREDENTIALS']['AZURE_KEY']
IBM_APIKEY = CONSTANTS['IBM_CREDENTIALS']['IBM_APIKEY']
IBM_URL = CONSTANTS['IBM_CREDENTIALS']['IBM_URL']
AWS_ACCESS_KEY = CONSTANTS['AWS_CREDENTIALS']['AWSAccessKeyId']
AWS_SECRET_KEY = CONSTANTS['AWS_CREDENTIALS']['AWSSecretKey']
DEEP_AI_KEY = CONSTANTS['DEEP_AI_CREDENTIALS']['DEEP_AI_KEY']

AWSAccessKeyId=AWS_ACCESS_KEY
AWSSecretKey=AWS_SECRET_KEY

credential_path = "./google_creds.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

azure_headers = {"Ocp-Apim-Subscription-Key": AZURE_KEY, 'Content-Type': 'application/json', 'Accept': 'application/json',}
deep_ai_headers = {'api-key': DEEP_AI_KEY}

DEBUG = True
application = Flask(__name__)

application.config.from_object(__name__)
application.config['SECRET_KEY'] = CONSTANTS['FLASK_SECRET_KEY']['SECRET_KEY']

comprehend = boto3.client(service_name='comprehend', aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY, region_name='us-west-2')

naturalLanguageUnderstanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey=IBM_APIKEY,
    url=IBM_URL
)

class ReusableForm(Form):
    textbox = TextAreaField('text:', validators=[validators.required()])

@application.route('/', methods=['GET', 'POST'])
def analyze():

    form = ReusableForm(request.form)
    print(form.errors)
    google_speech = Google_ST(audio, RATE, CHUNK)
    google_speech.file_transcribe()

    init_dict = {'sentiment': {'sentiment': 0.0, 'magnitude': 0.0, 'neg_sentiment': 0.0,
                 'pos_sentiment': 0.0, 'neg_sentiment': 0.0, 'neut_sentiment': 0.0}, 'entities': [],
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
        '''
        google = Google_Cloud(textbox)
        azure = Azure_API(azure_headers, textbox)
        aws = AWS_API(comprehend, textbox)
        ibm = IBM_API(naturalLanguageUnderstanding, textbox)
        deep_ai = Deep_AI_API(deep_ai_headers, textbox)

        thread_dict = {}
        sub_dict = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            google_sub_dict = {}
            google_sub_dict['sentiment'] = executor.submit(google.sentiment).result()
            google_sub_dict['entities'] = executor.submit(google.entities).result()
            google_sub_dict['categories'] = executor.submit(google.categories).result()
            google_sub_dict['syntax'] = executor.submit(google.syntax).result()

            azure_sub_dict = {}
            azure_sub_dict['sentiment'] = executor.submit(azure.sentiment).result()
            azure_sub_dict['entities'] = executor.submit(azure.entities).result()
            azure_sub_dict['keyphrases'] = executor.submit(azure.keyphrases).result()

            amazon_sub_dict = {}
            amazon_sub_dict['sentiment'] = executor.submit(aws.sentiment).result()
            amazon_sub_dict['entities'] = executor.submit(aws.entities).result()
            amazon_sub_dict['keyphrases'] = executor.submit(aws.keyphrases).result()
            amazon_sub_dict['syntax'] = executor.submit(aws.syntax).result()
 
            ibm_sub_dict = executor.submit(ibm.concepts).result()
  
            deep_ai_sub_dict = {}
            deep_ai_sub_dict['summary'] = executor.submit(deep_ai.summary).result()

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
        '''
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
