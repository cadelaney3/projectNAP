from flask import Flask, url_for, redirect, render_template, flash, request, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, FileField
from wtforms.widgets import TextArea
import requests
import json
import sys
import os
import io
from io import BytesIO
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

from app.google_api import Google_Cloud
from app.google_api import Google_ST
from app.azure_api import Azure_API
from app.aws_api import AWS_API
from app.ibm_api import IBM_API
from app.deep_ai_api import Deep_AI_API

from google.cloud import storage
import six

import tempfile
import wave

with open('./constants.json') as f:
    CONSTANTS = json.load(f)

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

DEBUG = False
application = Flask(__name__)

application.config.from_object(__name__)
application.config['SECRET_KEY'] = CONSTANTS['FLASK_SECRET_KEY']['SECRET_KEY']
application.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

comprehend = boto3.client(service_name='comprehend', aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY, region_name='us-west-2')

naturalLanguageUnderstanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey=IBM_APIKEY,
    url=IBM_URL
)

class ReusableForm(Form):
    textbox = TextAreaField('text:', validators=[validators.required()])

class AudioForm(Form):
    audioFile = FileField('audio')

def upload_audio_file(filename, content, content_type):
    if not filename:
        return None

    audio_filename = filename.replace(" ", "")
    storage_client = storage.Client()
    bucket = storage_client.bucket('project-nap-bucket')
    blob = bucket.blob(filename)
    blob.upload_from_string(
        content,
        content_type=content_type
    )
    url = blob.public_url
    if isinstance(url, six.binary_type):
        url = url.decode('utf-8')
    
    print("Uploaded file %s as %s" % (filename, url))
    gcs_uri = 'gs://project-nap-bucket/' + audio_filename
    return gcs_uri    

def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('project-nap-bucket')
    blob = bucket.blob(source_blob_name)

    blob.download_to_file(destination_file_name)
    #content = blob.download_as_string(storage_client)
    #return content

def analyze(textbox):
    results_dict = {}

    init_dict = {'sentiment': {'sentiment': 0.0, 'magnitude': 0.0, 'neg_sentiment': 0.0,
                 'pos_sentiment': 0.0, 'neg_sentiment': 0.0, 'neut_sentiment': 0.0}, 'entities': [],
                 'keyphrases': [], 'categories': [], 'syntax': [], 'summary': '',
                 'keywords': []}
    try:
        google = Google_Cloud(textbox)
    except Exception:
        print("Error: problem with Google Cloud")
    try:
        azure = Azure_API(azure_headers, textbox)
    except Exception:
        print("Error: problem with Azure API")
    try:
        aws = AWS_API(comprehend, textbox)
    except Exception:
        print("Error: problem with AWS API")
    try:
        ibm = IBM_API(naturalLanguageUnderstanding, textbox)
    except Exception:
        print("Error: problem with IBM API")
    try:
        deep_ai = Deep_AI_API(deep_ai_headers, textbox)
    except Exception:
        print("Error: problem with Deep AI API")

    google_dict = {}
    azure_dict = {}
    amazon_dict = {}
    ibm_dict = {}
    deep_ai_dict = {}
    keywords_dict = {}
    thread_dict = {}
    sub_dict = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        google_sub_dict = {}
        try:
            google_sub_dict['sentiment'] = executor.submit(google.sentiment).result()
            google_sub_dict['entities'] = executor.submit(google.entities).result()
            google_sub_dict['categories'] = executor.submit(google.categories).result()
            google_sub_dict['syntax'] = executor.submit(google.syntax).result()
        except Exception:
            print("Error: Google API calls")
            google_sub_dict = init_dict 

        try:
            azure_sub_dict = {}
            azure_sub_dict['sentiment'] = executor.submit(azure.sentiment).result()
            azure_sub_dict['entities'] = executor.submit(azure.entities).result()
            azure_sub_dict['keyphrases'] = executor.submit(azure.keyphrases).result()
            keywords_dict['keywords'] = azure_sub_dict['keyphrases']
        except Exception:
            print("Error: Azure API calls")
            azure_sub_dict = init_dict

        try:
            amazon_sub_dict = {}
            amazon_sub_dict['sentiment'] = executor.submit(aws.sentiment).result()
            amazon_sub_dict['entities'] = executor.submit(aws.entities).result()
            amazon_sub_dict['keyphrases'] = executor.submit(aws.keyphrases).result()
            for i in range(0, len(amazon_sub_dict['keyphrases'])):
                if amazon_sub_dict['keyphrases'][i] not in keywords_dict['keywords']:
                    keywords_dict['keywords'].append(amazon_sub_dict['keyphrases'][i])
            amazon_sub_dict['syntax'] = executor.submit(aws.syntax).result()
        except Exception:
            print("Error: AWS API calls")
            amazon_sub_dict = init_dict

        try:
            ibm_sub_dict = executor.submit(ibm.concepts).result()
            for i in range(0, len(ibm_sub_dict['keywords'])):
                if ibm_sub_dict['keywords'][i] not in keywords_dict['keywords']:
                    keywords_dict['keywords'].append(ibm_sub_dict['keywords'][i])
        except Exception:
            print("Error: IBM API calls")
            ibm_sub_dict = init_dict
        
        deep_ai_sub_dict = {}
        try:
            deep_ai_sub_dict['summary'] = executor.submit(deep_ai.summary).result()
        except Exception:
            print("Error: problem with deep_ai")
        else:
            deep_ai_sub_dict['summary'] = 'No summary available'

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

    results_dict = {'google': google_dict, 'azure': azure_dict, 'amazon': amazon_dict, 'ibm': ibm_dict, 'deep_ai': deep_ai_dict, 'keywords': keywords_dict}
    return results_dict


@application.route('/', methods=['GET', 'POST'])
def index():
    
    form = ReusableForm(request.form)
    textbox = ''
    transcription = ''

    print(form.errors)

    init_dict = {'sentiment': {'sentiment': 0.0, 'magnitude': 0.0, 'neg_sentiment': 0.0,
                 'pos_sentiment': 0.0, 'neg_sentiment': 0.0, 'neut_sentiment': 0.0}, 'entities': [],
                 'keyphrases': [], 'categories': [], 'syntax': [], 'summary': '',
                 'keywords': []}

    analyze_dict = {'google': init_dict, 'azure': init_dict, 'amazon': init_dict, 'ibm': init_dict, 'deep_ai': init_dict, 'keywords': init_dict}


    if request.method == 'POST':
        
        if 'file' in request.files:
            f = request.files.get('file')
            content = f.read()
            #f.save(secure_filename(f.filename))

            filename = secure_filename(f.filename)
            filename.replace(" ", '')
            f.seek(0)
            print(filename)

            gcs_uri = upload_audio_file(filename, content, f.content_type)

            # fd, path = tempfile.mkstemp()
            # try:
            #     with os.fdopen(fd, 'w+b') as temp:
            #         download_blob(filename, temp)
            #         #temp.write(c)
            #         w = wave.open(temp, 'rb')
            #         print(w.getnchannels())
            # finally:
            #     os.remove(path)

            if f.filename.lower().endswith(('.wav', '.flac', '.mp3', '.m4a', '.mp4')):
                RATE = 44100
            else:
                RATE = 1600

            try:
                google_speech = Google_ST(gcs_uri, RATE, CHUNK)
                transcription = google_speech.transcribe_file(gcs_uri)
                form.textbox.data = transcription
            except Exception as e:
                print(e)

        else:
            if form.textbox.data: 
                textbox = form.textbox.data
                try:
                    print()
                    analyze_dict = analyze(textbox)
                except Exception:
                    print("Error: could not analyze text")
    
    else:
        flash('Enter text to be processed:')

    return render_template('main.html', form=form, google_dict=analyze_dict['google'], azure_dict=analyze_dict['azure'], amazon_dict=analyze_dict['amazon'],
                           ibm_dict=analyze_dict['ibm'], deep_ai_dict=analyze_dict['deep_ai'], keywords_dict=analyze_dict['keywords'])

if __name__ == "__main__":
    #application.jinja_env.cache = {}
    application.run(host='0.0.0.0')
