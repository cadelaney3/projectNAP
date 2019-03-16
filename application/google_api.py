import six
import os
import io
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


class Google_Cloud:

    def __init__(self, text):
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
    def __init__(self, file):
        self.file = file
        self.client = speech.SpeechClient()

    def transcribe(self):
        with io.open(self.file, 'rb') as audio_file:
            content = audio_file.read()
            audio = types.RecognitionAudio(content=content)
        
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            #sample_rate_hertz=44100,
            language_code='en-US'
        )
        
        response = self.client.recognize(config, audio)

        for result in response.results:
            print('Transcript: {}'.format(result.alternatives[0].transcript))