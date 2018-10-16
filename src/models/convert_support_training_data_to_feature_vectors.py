from sklearn.linear_model import Ridge
import sys
from social_features import get_social_features
from social_features import get_reply_only_social_features
import time
from collections import *
import re
import pickle
import numpy as np
from sklearn.preprocessing import Imputer

def main():

    input_file = sys.argv[1]
    outfile = sys.argv[2]

    MIN_FREQ = 5
    if len(sys.argv) > 3:
        MIN_FREQ = int(sys.argv[3])

    reply_only = False
    if len(sys.argv) > 4:
        reply_only = True
        
    max_lines = -1
    if len(sys.argv) > 5:
        max_lines = int(sys.argv[5])
        
    y = []
    X_dicts = []

    time_sum = 0
    
    with open(input_file) as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue

            if max_lines > 0 and line_no >= max_lines:
                break
            
            cols = line[:-1].split('\t')
            # All four social scores
            
            post = normalize(cols[4])
            reply = normalize(cols[5])

            if len(post) == 0 or len(reply) == 0:
                continue
            
            start = time.time()
            if not reply_only:
                feats = get_social_features(post, reply)
            else:
                feats = get_reply_only_social_features(post, reply)
            end = time.time()
            time_sum += end-start

            if len(feats) == 0:
                continue
            
            y.append(cols[0:4])
            X_dicts.append(feats)
            if len(X_dicts) % 50 == 0:
                print(('Saw %d instances, average time: %f' \
                      % (len(X_dicts), time_sum / len(X_dicts))))
                # break


    # Count features to remove rare ones
    feature_counts = Counter()
    for d in X_dicts:
        for k in d.keys():
            feature_counts[k] += 1    

    # Convert to feature matrix and dump into a classifier
    feature_to_index = {}
    for d in X_dicts:
        for k in d.keys():
            if k not in feature_to_index and feature_counts[k] >= MIN_FREQ:
                feature_to_index[k] = len(feature_to_index)

    num_feats = len(feature_to_index)                
    print(('Using %d features (out of %d)' % (num_feats, len(feature_counts))))
    
    
    with open(outfile + '.features.tsv', 'w') as outf:
        for k,v in feature_to_index.items():
            outf.write('%s\t%d\n' % (k, v))


    # TODO: Use median feature value for unknowns somehow
    print("creating X")
    X = []            
    for j, d in enumerate(X_dicts):
        social = '\t'.join(y[j])
        
        vec = num_feats * [ float('nan') ]
        for k, v in d.items():
            # Skip rare features
            if k not in feature_to_index:
                continue
            i = feature_to_index[k]
            vec[i] = v
        X.append(np.array(vec))

            
    X = np.array(X)
    print(len(y), len(X))
    
    with open(outfile + '.zero.tsv', 'w') as outf:        
        for j, row in enumerate(np.nan_to_num(X)):
            social = '\t'.join(y[j])
            outf.write("%s\t%s\n" % (social, " ".join([str(x) for x in row])))

def normalize(text):
    text = text.strip()
    if len(text) > 1:
        text = text[0].upper() + text[1:]
    
    text = re.sub(r'\bi\b', 'I', text)
    text = re.sub(r'\b([yY])oure\b', r"\1ou're", text)
    text = re.sub(r'\b([yY])ouve\b', r"\1ou've", text)
    text = re.sub(r'\b([wWdD])ont\b', r"\1on't", text)
    text = re.sub(r'\b([tTwW])hats\b', r"\1hat's", text)
    text = re.sub(r'\b[iI]m\b', "I'm", text)
    text = re.sub(r'\bu\b', 'you', text)

    return text
    
        
if __name__ == '__main__':
    main()
