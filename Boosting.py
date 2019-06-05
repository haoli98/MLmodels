from sklearn import ensemble, tree
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import run_classifier

IS_REAL_TEST = True

AdaBoostClassifier = ensemble.AdaBoostClassifier(
  base_estimator = tree.DecisionTreeClassifier(), n_estimators = 1000)

Trained_AdaBoostClassifer = run_classifier.run_classifier(
  AdaBoostClassifier, IS_REAL_TEST, no_split = False)

