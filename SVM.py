from sklearn import svm
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import run_classifier

IS_REAL_TEST = False

SVMclassifier = svm.LinearSVC(penalty = "l2", loss = 'hinge', C = 1)

run_classifier.run_classifier(SVMclassifier, IS_REAL_TEST)