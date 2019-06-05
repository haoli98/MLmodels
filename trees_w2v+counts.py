#trees_tfidf.py

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import preprocessing
import pandas as pd
import numpy as np

word_vecs = np.array(preprocessing.word_vectors_by_tweet)
tweet_df = pd.read_csv('train.csv', usecols = [1,17], names=['text','label'], header=0,
                        dtype={'text':str,'label':int})

split=  int(len(tweet_df['text'].values)*80/100)
train_count_vecs = (tweet_df['text'].values)[:split]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(tweet_df['text'].values)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

train_count_vecs = ((X_train_tf)[:split]).todense()
train_labels = (tweet_df['label'].values)[:split]
valid_labels = (tweet_df['label'].values)[split:]
valid_count_vecs = ((X_train_tf)[split:]).todense()

#splitting data into training and validation
train_word_vecs = (word_vecs)[:split]
valid_word_vecs = (word_vecs)[split:]

train_vecs = np.hstack([train_count_vecs,train_word_vecs])
# print(train_vecs.shape)
valid_vecs = np.hstack([valid_count_vecs,valid_word_vecs])
# print(valid_vecs.shape)

abc = AdaBoostClassifier(n_estimators=100)
rfc = RandomForestClassifier(n_estimators=100)
etc = ExtraTreesClassifier(n_estimators=100)
vclf = VotingClassifier(estimators=[('abc', abc), ('rfc', rfc), ('etc', etc)], voting='hard')

for clf, label in zip([abc, rfc, etc, vclf], ['AdaBoost', 'Random Forest','Extra Trees', 'Voting']):
    print(label)
    clf.fit(train_vecs,train_labels)
    preds = clf.predict(valid_vecs)
    accuracy = np.mean(preds == valid_labels)
    print("Validation Accuracy: {}".format(accuracy))
