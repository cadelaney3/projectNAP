import boto3
import uuid
import time
import requests

class AWS_API:
    def __init__(self, client, text):
        self.comprehend = client
        self.text = text

    def sentiment(self):
        sentiments = self.comprehend.detect_sentiment(Text=self.text, LanguageCode='en')
        sent_dict = {}
        sent_dict['pos_sentiment'] = sentiments['SentimentScore']['Positive']
        sent_dict['neg_sentiment'] = sentiments['SentimentScore']['Negative']
        sent_dict['neut_sentiment'] = sentiments['SentimentScore']['Neutral']
        
        return sent_dict

    def entities(self):
        aws_entities = self.comprehend.detect_entities(Text=self.text, LanguageCode='en')
        ents = []
        for entity in aws_entities['Entities']:
            ents.append(entity['Text'].lower())
        
        ents.sort()
        return ents

    def keyphrases(self):
        aws_keyphrases = self.comprehend.detect_key_phrases(Text=self.text, LanguageCode='en')
        kps = []
        for phrase in aws_keyphrases['KeyPhrases']:
            kps.append(phrase['Text'].lower())

        kps.sort()
        return kps

    def syntax(self):
        aws_syntax = self.comprehend.detect_syntax(Text=self.text, LanguageCode='en')
        batch = []
        for word in aws_syntax['SyntaxTokens']:
            batch.append([word['Text'], word['PartOfSpeech']['Tag']])

        return batch

class AWS_transcribe:
    def __init__(self, client):
        self.transcribe = client

    def transcribe_audio(self, bucket_name, filename):
        try:
            audio_format = ''
            if filename.endswith('wav'):
                audio_format = 'wav'
            elif filename.endswith('mp3'):
                audio_format = 'mp3'
            elif filename.endswith('mp4'):
                audio_format = 'mp4'
            elif filename.endswith('flac'):
                audio_format = 'flac'
            else:
                return "Error: Amazon Transcribe does not work with this audio type."
            
            job_name = str(uuid.uuid4())
            job_uri = "https://s3-us-west-2.amazonaws.com/" + bucket_name + "/" + filename
            self.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': job_uri},
                MediaFormat=audio_format,
                LanguageCode='en-US'
            )
            while True:
                status = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                print("Not ready yet...")
                time.sleep(5)

            url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']

            temp = requests.get(url)
            data = temp.json()
            transcription = data['results']['transcripts'][0]['transcript']
            return transcription

        except Exception as e:
            print(e)
            return "Error: Amazon Transcribe could not transcribe this file"