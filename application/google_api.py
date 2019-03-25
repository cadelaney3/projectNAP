from __future__ import division

import re
import sys
import six
from six.moves import queue
import os
import io
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import speech
from google.cloud.speech import enums as enums2
from google.cloud.speech import types as types2


class Google_Cloud:

    def __init__(self, text):
        print(text)
        self.client = language.LanguageServiceClient()

        if isinstance(text, six.binary_type):
            text = text.decode('utf-8')

        self.document = types.Document(
                content=text.encode('utf-8'),
                type=enums.Document.Type.PLAIN_TEXT)

    def sentiment(self):
        google_sentiment = self.client.analyze_sentiment(self.document).document_sentiment
        sent = {}
        sent['sentiment'] = google_sentiment.score
        sent['magnitude'] = google_sentiment.magnitude
        return sent
    
    def entities(self):
        google_entities = self.client.analyze_entities(self.document).entities
        
        entities = []
        for entity in google_entities:
            entities.append(entity.name.lower())

        entities.sort()
        return entities

    def entity_sentiment(self):
        # Detect and send native Python encoding to receive correct word offsets.
        encoding = enums.EncodingType.UTF32
        if sys.maxunicode == 65535:
            encoding = enums.EncodingType.UTF16

        result = self.client.analyze_entity_sentiment(self.document, encoding)

        entities = {}
        for entity in result.entities:
            entity_str = ""
            entity_str += 'Mentions: '
            entity_str += (u'Name: "{}"'.format(entity.name))
            name = entity.name
            entities[name] = entity.sentiment

        return entities

    def syntax(self):
        """Detects syntax in the text."""

        # Detects syntax in the document. You can also analyze HTML with:
        #   document.type == enums.Document.Type.HTML
        tokens = self.client.analyze_syntax(self.document).tokens

        # part-of-speech tags from enums.PartOfSpeech.Tag
        pos_tag = ('UNKNOWN', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM',
                'PRON', 'PRT', 'PUNCT', 'VERB', 'X', 'AFFIX')

        result = []
        for token in tokens:
            result.append((u'{}: {}'.format(pos_tag[token.part_of_speech.tag],
                                token.text.content)))
        
        return result

    def categories(self):
        """Classifies content categories of the provided text."""
        categories = self.client.classify_text(self.document).categories

        result = []
        for category in categories:
            result.append(category.name)

        return result
    
class Google_ST:
    def __init__(self, file, rate, chunk):
        self.audio_file = file
        self.client = speech.SpeechClient()
        self.rate = rate
        self.chunk = chunk
        self.config = types2.RecognitionConfig(
            encoding=enums2.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=rate,
            language_code='en-US'
        )
        self.streaming_config = types2.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True
        )

    def printFields(self):
        print(type(self.audio_file))
        print(type(self.audio_file.read()))

    def transcribe_file(self):
        with io.open(self.audio_file, 'rb') as audio_file:
            content = audio_file.read()
            #print(type(content))
            audio = types2.RecognitionAudio(content=content)
        
        response = self.client.recognize(self.config, audio)
        result_str = ''
        for result in response.results:
            result_str += result.alternatives[0].transcript
            print('Transcript: {}'.format(result.alternatives[0].transcript))

        return result_str
    
    def transcribe_long_file(self, uri):
       # with io.open(self.audio_file, 'rb') as audio_file:
        #    content = audio_file.read()
        audio = types2.RecognitionAudio(uri=uri)
        
        operation = self.client.long_running_recognize(self.config, audio)
        print('Waiting for operation to complete')
        response = operation.result(timeout=1000)

        result_str = ''
        for result in response.results:
            result_str += result.alternatives[0].transcript
        
        return result_str

    
    def transcribe_mic(self):
        with MicStream(self.rate, self.chunk) as stream:
            audio_generator = stream.generator()
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = self.client.streaming_recognize(self.streaming_config, requests)
            stream.listen_print_loop(responses)
