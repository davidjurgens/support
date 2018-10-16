from sklearn.linear_model import *
import sys
import time
import json
import datetime
from datetime import date, timedelta as td
import time
import re
import numpy
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from nltk.tokenize.punkt import PunktSentenceTokenizer as sentTokenizer
from os import listdir
import math
from collections import defaultdict
import pandas as pd
import sklearn
import random
import csv
import sys
import os
import pickle
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.linear_model import LogisticRegressionCV
#import theano
from sklearn import svm
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
from sklearn.metrics import *
from collections import Counter
from sklearn import preprocessing
import sys

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import *

import pickle
from sklearn.model_selection import cross_val_score
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn import metrics as sk_metrics
from sklearn.model_selection import validation_curve

import numpy as np

from imblearn.over_sampling import *
from imblearn.combine import *
from imblearn.pipeline import make_pipeline

from sklearn.dummy import *

def main():

    input_prefix = sys.argv[1]

    tag = ''
    if len(sys.argv) > 2:
        tag = sys.argv[2]
    
    social = ['agreement', 'offensiveness', 'politeness', 'support']
    
    y = [[], [], [], []]
    X = []   
       
    input_file = input_prefix + '.zero.tsv'
    
    with open(input_file) as f:
        for line_no, line in enumerate(f):
            cols = line.split('\t')
            
            for i in range(4):
                # Convert to classes
                val = float(cols[i])
                label = None
                if i == 1: # Offensive
                    if val >= 1.5:
                        label = 'Offensive'
                    else:
                        label = 'Neutral'
                else:
                    if val > 3.5:
                        label = 'Positive'
                    elif val < 2.5:
                        label = 'Negative'
                    else:
                        label = 'Neutral'
                    
                y[i].append(label)

            feat_vec = [float(x) for x in cols[4].split()]
            feat_vec = np.array(feat_vec)

            X.append(feat_vec)
            if line_no > 0 and line_no % 1000 == 0:
                print('Loaded %d lines' % line_no)
            
    X = np.array(X)

    for i, name in enumerate(social):
        if name != 'support':
            continue
        
        print('Training %s on %d instances' % (name, len(y[i])))
        clf = RandomForestClassifier(n_estimators=128, n_jobs=-1, min_samples_leaf=5,
                                     verbose=10, max_features=0.25)
        pipeline = make_pipeline(SMOTE(), clf)    
        pipeline.fit(X, y[i])
        with open(input_prefix + '.' + name + tag + '.classifier.pkl', 'wb') as outf:
            pickle.dump(clf, outf)
            
if __name__ == '__main__':
    main()
