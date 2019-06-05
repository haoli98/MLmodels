# cnbmodel.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

#getting data from the csv
tweet_df = pd.read_csv('train.csv', usecols=['text','label'], header=0,
            dtype={'text':str,'label':int})
split=  int(len(tweet_df['text'].values)*80/100)

#splitting data into training and validation
training_text = (tweet_df['text'].values)[:split]
training_labels = (tweet_df['label'].values)[:split]

#build pipeline for classifier
text_cnb = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', ComplementNB()), ])

#fit the classfier to data
text_cnb.fit(training_text, training_labels)

#testing the classifier on the validation data
validation_text = tweet_df['text'].values[split:]
validation_labels = tweet_df['label'].values[split:]
validation_predictions = text_cnb.predict(validation_text)

#getting validation error
accuracy = np.mean(validation_predictions == validation_labels)
print("Validation Accuracy: {}".format(accuracy))
#Testing Accuracy: 0.7798165137614679
