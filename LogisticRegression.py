#logregrssion_postgrid.py

from sklearn import ensemble
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
import run_classifier


IS_REAL_TEST = True

lr_clf = LogisticRegression(penalty='l1',C=0.2)

trained_lr_clf = run_classifier.run_classifier(lr_clf, IS_REAL_TEST)
