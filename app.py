from fastapi import FastAPI
import logging
import email
from pydantic import BaseModel
from fuzzywuzzy import fuzz

import torch
import torch.nn.functional as F 
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
import spacy


# logging configuration
logging.basicConfig(filename='log_app.log',
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)


sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
nlp_sm = spacy.load("en_core_web_sm")


def sentence_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = F.softmax(logits, dim=1)
    negative_sentiment_score = float(probabilities[0][0])
    if negative_sentiment_score>0.6:
        return 'NEGATIVE'
    if negative_sentiment_score<0.4:
        return 'POSITIVE'
    return 'NEUTRAL'

def extract_per_org(text):
    doc = nlp_sm(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]

def detect_faud(text):
    # detect if email related to 'Enron's oil & gas business' using fuzzy matching (Levenshtein Distance)
    pattern_expression = "Enron's oil & gas business"
    matching_score = fuzz.partial_ratio(text, pattern_expression)
    if matching_score>0.5:
        return True, extract_per_org(text)
    return False, None


class QueryObject(BaseModel):
    email_string: str


app = FastAPI()

@app.post('/process_email')
def process_email_content(query: QueryObject):
    try:
        msg = email.message_from_string(query.email_string)
        message_body = msg.get_payload()
    except Exception as e:
        logging.error(e)
        return 'inputted email string format is not valid, review log file for error details', 500
    email_sentiment = sentence_sentiment(message_body)
    email_is_fraud, per_org_list = detect_faud(message_body)
    return {'sentiment': email_sentiment,
            'email-suspicious': email_is_fraud,
            'persons-organizations': per_org_list}
