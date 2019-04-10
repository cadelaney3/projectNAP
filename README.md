Web app that is hooked up with different text analytics and transcription APIs

to run you will need to make a constanst.json that looks like:

{
    "AWS_CREDENTIALS": {
        "AWSAccessKeyId": "yourAWSkeyID",
        "AWSSecretKey": "yourAWSsecretKey"
    },
    "IBM_CREDENTIALS": {
        "IBM_APIKEY": "yourIBMkey",
        "IBM_URL": "https://gateway.watsonplatform.net/natural-language-understanding/api"
    },
    "AZURE_CREDENTIALS": {
        "AZURE_KEY": "yourAzureKey"
    },
    "DEEP_AI_CREDENTIALS": {
        "DEEP_AI_KEY": "yourDeepAiKey"
    },
    "FLASK_SECRET_KEY": {
        "SECRET_KEY": "yourFlaskSecretKey"
    }
}

You will also need to sign up for Google Cloud and download a credentials file, which
I named google_creds.json