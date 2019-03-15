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
        deep_ai_summary = output['output']
        return deep_ai_summary