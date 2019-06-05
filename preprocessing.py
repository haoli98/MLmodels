import nltk
import numpy as np
from collections import defaultdict
from collections import OrderedDict
import random
import math
import csv
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize import TweetTokenizer
import pickle
from scipy.sparse import hstack
from sklearn import svm
from sklearn import ensemble


# There are 4 different datasets to train on of different dimensions, 25 is smallest
TWITTER_PATH = "/glove.twitter.27B.25d.txt"
d = 25

class Tweet:
    """ Tweet Class """
    def __init__(self, id, text, favorited, favoriteCount, replyToSN, created, 
                truncated, replyToSID, id_t, replyToUID, statusSource, screenName, retweetCount, 
                isRetweet, retweeted, longitude, latitude, label):
        self.id = id
        self.text = text
        self.favorited = favorited
        self.favoriteCount = favoriteCount
        self.replyToSN = replyToSN
        self.created = created
        self.truncated = truncated
        self.replyToSID = replyToSID
        self.id_t = id_t
        self.replyToUID = replyToUID
        self.statusSource = statusSource
        self.screenName = screenName
        self.retweetCount = retweetCount
        self.isRetweet = isRetweet
        self.retweetCount = retweetCount
        self.retweeted = retweeted
        self.longitude = longitude
        self.latitude = latitude
        self.label = label

def training():
    Tweetlst = []
    with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                newtweet = Tweet(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15],row[16], row[17])
                # print(row)
                Tweetlst.append(newtweet)
                line_count += 1
        print(f'Processed {line_count} lines.')

    print(len(Tweetlst))
    # print(Tweetlst[0].text)
    return Tweetlst

def testing():
    Tweetlst = []
    with open('test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                newtweet = Tweet(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], None, row[10], row[11], row[12], row[13], row[14], row[15], None)
                # print(row)
                Tweetlst.append(newtweet)
                line_count += 1
        print(f'Processed {line_count} lines.')

    print(len(Tweetlst))
    # print(Tweetlst[0].text)
    return Tweetlst

# def load_tweet_pickle():
#     # load pickle
#     f = open('raw_tweets.txt', encoding="utf8")
#     y = pickle.load(f)
#     f.close
#     return y

    # x=None usecols=np.arange(0,17)
    # with open('train.csv') as csv_file:
    # arrays = np.genfromtxt('train.csv', dtype=None, delimiter=",")
    # arrays = [np.array(map(line.split(','))) for line in open('train.txt')]
    # print(arrays[0])

def create_glove_models(filepath):
    glove_input_file =  filepath
    if 'twitter' in filepath:
        output_file = 'twitter'
    else:
        output_file = 'wikipedia'
    output_file += "_word2vec.txt"
    glove2word2vec(glove_input_file, output_file)

def get_glove_model(filepath):
    return gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=False)

### Comment or uncomment - creates twitter embeddings and stores in file
# create_glove_models("D:/NLP" + TWITTER_PATH )

### Loads the model
# model = get_glove_model("twitter_word2vec.txt")

def calculate_tweet_rep(speech_tokens, model, d):
    # avg = np.zeros((d))
    avg = np.zeros(d)
    # print(np.shape(avg))
    for token in speech_tokens:
        try:
            vector = model.get_vector(token)
            avg = np.add(vector, avg)
        except:
            # print("<NOT IN EMBEDDING")
            # avg = np.add(np.zeros(300), avg)
            continue
    avg =  avg/len(speech_tokens)
    return avg

def get_word_vectors(xTr, model, d):
    # tweet_lst = training()
    tweet_lst = xTr
    # print(tweet_lst)
    # print(tweet_lst[3].id)

    tweet_vectors = []

    tknzr = TweetTokenizer()
    for tweet in tweet_lst:
        tokens = tknzr.tokenize(tweet.text)
        # print(tweet.text)
        # print(tokens)
        tweet_rep = calculate_tweet_rep(tokens, model, d)
        tweet_vectors.append(tweet_rep)
    # print(tweet_vectors)
    return tweet_vectors

# only need to run once, can comment out
# create_glove_models("D:/NLP" + TWITTER_PATH)

# model = get_glove_model("twitter_word2vec.txt")
# word_vectors_by_tweet = get_word_vectors(xTr, model, d)

# print(word_vectors_by_tweet[0:3])

# print("Success")

##### Features #####
#number of capital letter in a sentence
# sum(1 for character in tweet if c.isupper())

# number of capital words in a sentence
# sum(1 for word in tweet if c.isupper())

# add the word embeddings of specifically tagged people, without noise of words - emphasized separately

#if there is a link: contains http:
# if ("http" in word)
#     feature = True
# else:
#     feature=False

# xTr = training()

def sum_capitals(text):
    return sum(1 for c in text if c.isupper())

def num_capitals(text):
    return sum(1 for word in text if word.isupper())

def has_http(text):
    if ("http" in text):
        return True
    return False

def compute_at_tokens(tokens, model):
    # print(tokens)
    count=0
    avg=np.zeros(d)
    for token in tokens:
        try:
            if '@' in token:
                count+=1
                sub_token=token[1:]
                try:
                    vector = model.get_vector(sub_token)
                except:
                    vector = np.zeros(d)
                avg += vector
        except:
            print("<NOT IN EMBEDDING>")
            continue
    avg =  avg/len(tokens)
    return count, avg

def compute_hash_tokens(tokens, model):
    # print(tokens)
    count=0
    avg=np.zeros(d)
    for token in tokens:
        if '#' in token:
            count+=1
            sub_token=token[1:]
            try:
                vector = model.get_vector(sub_token)
            except:
                vector = np.zeros(d)
            avg += vector
    avg =  avg/len(tokens)
    return count, avg    

def get_text_features(xTr):
    # xTr = training()
    features = []
    for tweet in xTr:
        capital_letters = sum_capitals(tweet.text)
        capital_words = num_capitals(tweet.text)
        http = has_http(tweet.text)

        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(tweet.text)
        at_count, at_embeddings = compute_at_tokens(tokens, model)
        hash_count, hash_embeddings = compute_hash_tokens(tokens, model)

        new_feature = np.array([capital_letters, capital_words, http, at_count, hash_count])
        new_feature = np.hstack([new_feature, at_embeddings, hash_embeddings])
        features.append(new_feature)
    return np.asarray(features)
    
def add_ardys_features(xTr, word_vecs):
    # word_vecs = word_vectors_by_tweet()
    other_vecs = get_text_features(xTr)
    print("debugging")
    # print(word_vecs.shape)
    print(type(word_vecs))
    # print(other_vecs.shape)
    print(type(other_vecs))
    stack = np.hstack([word_vecs, other_vecs])
    return stack

def get_labels(tweets):
    labels=[]
    for tweet in tweets:
        labels.append(tweet.label) 
    return labels

# def get_yTr_labels(tweets):


#####################################################################
################## Run Classifier ###################################
#####################################################################

# create_glove_models("D:/NLP" + TWITTER_PATH)

model = get_glove_model("twitter_word2vec.txt")
print("Success")

# shuffle_index = random.sample(range(0,len(xTr)))

# xTrain = training() #a list of tweets
# split = int(len(xTrain)*80/100)

# shuffle_index = random.sample(range(0,len(xTrain)), len(xTrain))
# xTrain = np.asarray(xTrain)[shuffle_index]
# xTr_new = xTrain[:split]
# yTr_dev = xTrain[split:]
# shuffle_index1 = shuffle_index[:split]
# shuffle_index2 = shuffle_index[split:]

# xTr_new = xTr_new[shuffle_index]
# yTr_dev = yTr_dev[shuffle_index]

# word_vectors_by_tweet_train = get_word_vectors(xTr_new, model, d)
# word_vectors_by_tweet_test = get_word_vectors(yTr_dev, model, d)
#actually should say xTe


# # training feature data
# transformed_xTr = add_ardys_features(xTr_new, word_vectors_by_tweet_train)
# transformed_yTr = add_ardys_features(yTr_dev, word_vectors_by_tweet_test)

# xTe = get_labels(xTr_new)
# yTe = get_labels(yTr_dev)

# test_predictions = classifier.predict(test_feature_vector)

# print(transformed_data[0:3])

# SVMclassifier = svm.LinearSVC(penalty = "l2", loss = 'squared_hinge', C = 1)
# SVMclassifier.fit(transformed_xTr, xTe)
# test_predictions = SVMclassifier.predict(transformed_yTr)

# accuracy = np.mean(test_predictions == yTe)
# print("Testing Accuracy SVM: {}".format(accuracy))


# RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators = 500, oob_score = True)
# RandomForestClassifier.fit(transformed_xTr, xTe)
# test_predictions = RandomForestClassifier.predict(transformed_yTr)

# print("oob error: " + str(RandomForestClassifier.oob_score_))

# accuracy = np.mean(test_predictions == yTe)
# print("Testing Accuracy Forest: {}".format(accuracy))

def predict_test(model):
    #lists of tweets.
    original_training_data = training()
    actual_testing_data = testing()

    tweet_train = get_word_vectors(original_training_data, model, d)
    tweet_test = get_word_vectors(actual_testing_data, model, d)

    transformed_xTr = add_ardys_features(original_training_data, tweet_train)
    transformed_xTe = add_ardys_features(actual_testing_data, tweet_test)

    yTr = get_labels(original_training_data)
    # yTe = get_labels(actual_testing_data)

    RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators = 100, oob_score = True)
    RandomForestClassifier.fit(transformed_xTr, yTr)
    test_predictions = RandomForestClassifier.predict(transformed_xTe)
    print("oob error: " + str(RandomForestClassifier.oob_score_))
    # accuracy = np.mean(test_predictions == yTe)
    # print("Testing Accuracy Forest: {}".format(accuracy))

    return test_predictions

def toKaggle(model):
    with open('forest_submission2.csv', 'w') as fw:
        fw.write("ID" + "," + "Label")
        fw.write('\n')
        preds = predict_test(model)
        for i in range(len(preds)):
            fw.write(str(i) + "," + str(preds[i]))
            fw.write('\n')
        fw.write('\n')
        fw.close()

toKaggle(model)