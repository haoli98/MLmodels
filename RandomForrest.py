from sklearn import ensemble
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import run_classifier

IS_REAL_TEST = True

RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators = 1000,
  oob_score = True)

Trained_RandomForestClassifer = run_classifier.run_classifier(
  RandomForestClassifier, IS_REAL_TEST, no_split = False)

print("oob error: " + str(Trained_RandomForestClassifer.oob_score_))

