from sklearn import svm
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import skeleton

  #If IS_REAL_TEST = False, split data into training and validation.
  #   test classifier based on how well it does on validation data
  #If IS_REAL_TEST = True, train classifier on all training data
  #   test classifier and store result in submission.csv
  #classifier should be of the form:
  # svm.LinearSVC(penalty = "l2", loss = 'squared_hinge', C = 1)

def run_classifier(classifier, IS_REAL_TEST = False):


  def add_features(text_vector, additional_feature_list):
    #add additional features:

    training_feature_vector = text_vector

    for feature in additional_feature_list:
      additional_feature_vector = feature/ sum(feature)
      training_feature_vector =  hstack([training_feature_vector,
        additional_feature_vector[:,None]])

    return training_feature_vector


  def normalize_time(tweet_df):
    #magic to get time of day as a number between 0 and 1
    tweet_df['created'] = pd.to_datetime(tweet_df['created'].str.split().str[1])
    tweet_df['created'] = pd.to_timedelta(tweet_df['created'])
    tweet_df['created'] = (tweet_df['created']).dt.total_seconds()
    tweet_df['created'] = tweet_df['created'] - min(tweet_df['created'])
    tweet_df['created'] = tweet_df['created'] / max(tweet_df['created'])

    return tweet_df

  #get data from csv
  tweet_df = pd.read_csv('train.csv',
    usecols=['text','retweetCount', 'favoriteCount', 'label', 'created'],
    header=0,
    dtype={'text':str, 'retweetCount':int, 'favoriteCount':int, 'label':int,
      'created':str})
  tweet_df = normalize_time(tweet_df)


  #shuffle data
  if IS_REAL_TEST:
    split=  int(len(tweet_df['text'].values))
  else:
    #split data into training and validation
    split=  int(len(tweet_df['text'].values)*80/100)

  #shuffle data (I think)
  tweet_df = tweet_df.sample(frac=1, random_state = 42)


  training_text = (tweet_df['text'].values)[:split]
  training_retweets = (tweet_df['retweetCount'].values)[:split]
  training_favorited = (tweet_df['favoriteCount'].values)[:split]
  training_time = (tweet_df['created'].values)[:split]

  training_labels = (tweet_df['label'].values)[:split]

  additional_features = [training_retweets, training_favorited, training_time]


  #print("training tweets" + str(training_retweetss))

  count_vect = CountVectorizer()
  training_text_vector = count_vect.fit_transform(training_text)
  print("before adding metadata, training text vector shape:" + str(training_text_vector.shape))

  training_feature_vector = add_features(training_text_vector, additional_features)

  print("after adding metadata, feature vector shape:" + str(training_feature_vector.shape))

  # print("training_retweet_vector"+ str(training_retweet_vector))

  #train the classifier
  print("starting training")
  classifier.fit(training_feature_vector, training_labels)
  print("finished training")


  if IS_REAL_TEST:
    tweet_df = pd.read_csv('test.csv',
     usecols=['text','retweetCount', 'favoriteCount',  'created'],
              header=0, dtype={'text':str, 'retweetCount':int,
              'favoriteCount':int, 'label':int,'created':str})
    tweet_df = normalize_time(tweet_df)
    split = 0

  text = (tweet_df['text'].values)[split:]
  retweets = (tweet_df['retweetCount'].values)[split:]
  favorited = (tweet_df['favoriteCount'].values)[split:]
  time = (tweet_df['created'].values)[split:]

  additional_features = [retweets, favorited, time]

  text_vector = count_vect.transform(text)
  test_feature_vector = add_features(text_vector, additional_features)

  test_predictions = classifier.predict(test_feature_vector)

  if IS_REAL_TEST:
    n = test_predictions.shape[0]
    #print(n)
    indicies = np.asarray((range(n)))
    #print(indicies)
    #print(test_predictions)

    #data_to_write = np.concatenate((indicies, test_predictions))
    data_to_write = {'col2':test_predictions}
    df_to_write = pd.DataFrame(data_to_write)
    #np.savetxt('submission.csv', df_to_write, delimiter=',', header = 'ID, Label')
    df_to_write.to_csv('submission.csv', header = ['Label'])
      #need to add ID manually

  else:
    validation_labels = tweet_df['label'].values[split:]
    #calculate prediction accuracy
    accuracy = np.mean(test_predictions == validation_labels)
    print("Testing Accuracy: {}".format(accuracy))
    #accuracy = 0.77522 . no reshuffling
    #accuracy = 0.8256880733944955 reshuffling
    #accuracy = 0.8256880733944955 adding in metadata