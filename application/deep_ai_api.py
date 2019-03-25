import requests

class Deep_AI_API:
    def __init__(self, headers, text):
        self.headers = headers
        self.text = text
 
    def summary(self):
        r = requests.post(
            "https://api.deepai.org/api/summarization",
            data={
                'text': self.text,
            },
            headers=self.headers
        )

        output = r.json()
        print(output)
        deep_ai_summary = ''
        if 'output' in output:
            deep_ai_summary = output['output']
        else:
            deep_ai_summary = 'No summary available'
        return deep_ai_summary

