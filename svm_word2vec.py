from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import preprocessing

word_vecs = preprocessing.word_vectors_by_tweet
tweet_df = pd.read_csv('train.csv', names=['label'], header=0, dtype={'label':int})

#splitting data into training and validation
split=  int(len(word_vecs)*80/100)
train_vecs = (word_vecs)[:split]
train_labels = (tweet_df['label'].values)[:split]

#generate the classifier
SVMclassifier = svm.LinearSVC(penalty = "l2", loss = 'squared_hinge', C = 1.0)

#train the classifier
print("starting training")
SVMclassifier.fit(train_vecs, train_labels)
print("finished training")

#get validation data to train the classifier
valid_vecs = word_vecs[split:]
valid_labels = tweet_df['label'].values[split:]

#predict labels for validation data
valid_preds = SVMclassifier.predict(valid_vecs)

#calculate prediction accuracy
accuracy = np.mean(valid_preds == valid_labels)
print("Testing Accuracy: {}".format(accuracy))
