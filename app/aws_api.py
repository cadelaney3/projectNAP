import boto3

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