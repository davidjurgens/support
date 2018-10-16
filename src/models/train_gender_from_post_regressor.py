from sklearn.linear_model import *
import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from os import listdir
import math
from collections import defaultdict
import sklearn
import random
import os

from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV

import pickle


def main():

    input_file = sys.argv[1]
    attr = sys.argv[2]
    outfile = sys.argv[3]
    
    y = []
    X = []   

    lr = LogisticRegression()
    clf = lr

    with open(input_file) as f:
        for line_no, line in enumerate(f):
            cols = line.split('\t')

            label = [x for x in cols[0].split(',') if attr in x][0]
            y.append(label)

            vec_length = int(cols[2])
            feat_vec = np.zeros(vec_length)

            for i in range(3, len(cols), 2):
                feat_vec[int(cols[i])] = float(cols[i+1])
                
            X.append(feat_vec)
            if line_no > 0 and line_no % 25000 == 0:
                print('Loaded %d lines from %s' % (line_no, input_file))

    print('Converting to np')
    X = np.array(X)
    y = np.array(y)

    print('Fitting classifier with data from ' + input_file)
    clf.fit(X,y)

    print('Saving')
    with open(outfile, 'wb') as outf:
        pickle.dump(lr, outf)
    
if __name__ == '__main__':
    main()
