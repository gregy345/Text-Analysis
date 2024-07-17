# pretrained sentiment

"""
This script expands upon the example in the 
model card for distilbert-base-uncased-finetuned-sst-2-english

"""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from functools import reduce
import pandas as pd 
import numpy as np
import os 

# load data
os.chdir("path/to/data/set")
print(os.getcwd())

data_df = pd.read_csv("dataset.txt",sep=",",header=1,names=['ID','openended','V1','V2'])
data_re = data_df.drop(['V1','V2'], axis=1)
data_prime = data_re.set_index('ID')['openended'].to_dict()

# load model
model_location = "path/to/model/models4distilbert"
tokenizer = DistilBertTokenizer.from_pretrained(model_location)
model = DistilBertForSequenceClassification.from_pretrained(model_location)
 
# funciton to apply model 
def sentscore(indat):
    inputs = tokenizer(indat, return_tensors="pt")  
    with torch.no_grad(): 
        logits = model(**inputs).logits 

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# data set with scores
sent_dict = {ii[0]: ii[1] for ii in data_prime.items()} 
df_sent = pd.DataFrame(sent_dict.items(), columns=['ID','score'])
sentiment_data = data_df.merge(df_sent, on='ID')
print(' \n dataset with scores: \n', sentiment_data)

# deidentified openended with scores 
sent_dict1 = {ii: sentscore(ii) for ii in data_prime.values()}
sent_df = pd.DataFrame(sent_dict1.items(), columns=['words','score'])
print(' \n openended with scores (sent_df): \n', sent_df)

