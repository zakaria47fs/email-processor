# email-processor
install spacy small language model with "python -m spacy download en_core_web_sm"  
install project requirements with "pip install -r requirements.txt"  
run the app with "uvicorn app:app"  

## used technologies
We using 'distilbert-base-uncased-finetuned-sst-2-english' transformer from huggingface to detect email sentiment  
Using fuzzywuzzy with Levestein distance to detect whether the email is related to ""  
Using Spacy small pretrained model for NER (Named Entity Recognition), to extract persons and organizations from the email message
