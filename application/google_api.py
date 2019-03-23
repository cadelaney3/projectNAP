from __future__ import division

import re
import sys
import six
from six.moves import queue
import os
import io
import pyaudio
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
    def __init__(self, file, rate, chunk):
        self.audio_file = file
        self.client = speech.SpeechClient()
        self.rate = rate
        self.chunk = chunk
        self.config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=rate,
            language_code='en-US'
        )
        self.streaming_config = types.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True
        )

    def printFields(self):
        print(type(self.audio_file))
        print(type(self.audio_file.read()))

    def transcribe_file(self):
        with io.open(self.audio_file, 'rb') as audio_file:
            #content = self.audio_file.read()
            content = audio_file.read()
            print(type(content))
            audio = types.RecognitionAudio(content=content)
        
        response = self.client.recognize(self.config, audio)
        result_str = ''
        for result in response.results:
            result_str += result.alternatives[0].transcript
            print('Transcript: {}'.format(result.alternatives[0].transcript))

        return result_str
    
    def transcribe_mic(self):
        with MicStream(self.rate, self.chunk) as stream:
            audio_generator = stream.generator()
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = self.client.streaming_recognize(self.streaming_config, requests)
            stream.listen_print_loop(responses)

class MicStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        
        return self
    
    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True

        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

        while True:
            try:
                chunk = self._buff.get(block=False)
                if chunk is None:
                    return
                data.append(chunk)
            except queue.Empty:
                break

            yield b''.join(data)

    def listen_print_loop(self, responses):
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.
        """
        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue

            # The `results` list is consecutive. For streaming, we only care about
            # the first result being considered, since once it's `is_final`, it
            # moves on to considering the next utterance.
            result = response.results[0]
            if not result.alternatives:
                continue

            # Display the transcription of the top alternative.
            transcript = result.alternatives[0].transcript

            # Display interim results, but with a carriage return at the end of the
            # line, so subsequent lines will overwrite them.
            #
            # If the previous result was longer than this one, we need to print
            # some extra spaces to overwrite the previous result
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + '\r')
                sys.stdout.flush()

                num_chars_printed = len(transcript)

            else:
                print(transcript + overwrite_chars)

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r'\b(exit|quit)\b', transcript, re.I):
                    print('Exiting..')
                    break

                num_chars_printed = 0
