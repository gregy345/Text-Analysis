#rules_based_sentiment

"""
This script uses the vader module from nltk to calcualte 
sentiment scores of comments in a dataset
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from functools import reduce
import pandas as pd
import os  

os.chdir("path/to/data/set")
print(os.getcwd())

# read in the csv file
data_df = pd.read_csv("dataset.txt", sep=",")

# convert openended column to nested list then reduce to list 
remarks_object = data_df[["openended"]].values.tolist()
remarks_list = reduce(lambda x,y: x+y, remarks_object)

# calculate sentiment scores
sid = SentimentIntensityAnalyzer()
sent_score = []
for ii in remarks_list:
	sent = sid.polarity_scores(ii)
	sent_score.append(sent)

# display list of dictionaries of sentiment scores
print(" \n list of sentiment scores \n ", sent_score)

# list of compound scores
comp_score = [x['compound'] for x in sent_score]
print(" \n the list of compound scores: \n ", comp_score)

# join original df and single column df of sentiment
comp_df = pd.DataFrame(comp_score, columns = ["col_se"])
data_sentiment  = pd.concat([data_df,comp_df], axis=1)
print(" \n data frame with openended and scores: \n ", data_sentiment)



