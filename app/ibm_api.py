from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
    import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions, EmotionOptions, RelationsOptions, SemanticRolesOptions, SentimentOptions, CategoriesOptions

class IBM_API:
    def __init__(self, headers, text):
        self.naturalLanguageUnderstanding = headers
        self.text = text

    def concepts(self):
        IBM_dict = {}
        IBM_response = self.naturalLanguageUnderstanding.analyze(
            text=self.text,
            features=Features(
                entities=EntitiesOptions(emotion=True, sentiment=True, limit=10),
                keywords=KeywordsOptions(emotion=True, sentiment=True,limit=10),
                sentiment=SentimentOptions(),
                categories=CategoriesOptions()
                )).get_result()

        sent_dict = {'sentiment': IBM_response['sentiment']['document']['score']}
        IBM_dict['sentiment'] = sent_dict
        
        ent_result = []
        ents = IBM_response['entities']
        for e in ents:
            ent_result.append(e['text'].lower())
        ent_result.sort()
        IBM_dict['entities'] = ent_result
        
        kws = []
        for keyword in IBM_response['keywords']:
            kws.append(keyword['text'].lower())
        kws.sort()
        IBM_dict['keywords'] = kws
        
        cats = []
        for category in IBM_response['categories']:
            cats.append(category['label'])
        IBM_dict['categories'] = cats
        
        return IBM_dict