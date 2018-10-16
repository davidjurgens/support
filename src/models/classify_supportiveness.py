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

    base_dir = '../..//working-dir/feature-vectors/'
    clf_file = base_dir + 'social-features.min-5.tsv.support.rf-128.ml-5.feat-0.25.classifier.pkl'
    feat_file = base_dir +  'social-features.min-5.tsv.features.tsv'

    max_lines = -1
    
    print(('Classifying %d instances from %s using the classifier at %s' \
          % (max_lines, input_file, clf_file)))
        
    time_sum = 0

    feature_to_index = {}
    with open(feat_file, 'r') as f:
        for line in f:
            cols = line[:-1].split('\t')
            feature_to_index[cols[0]] = int(cols[1])
    num_feats = len(feature_to_index)
            
    print('loading classifier')
    with open(clf_file, 'rb') as f:
        clf = pickle.load(f)
        clf.n_jobs = 1
        clf.verbose = 0

    header = [ 'instance_id',  'location', 'self-attribute', 'other-attribute', ]
    print(list(clf.classes_))
    # return
    header.append('support')
        
    with open(input_file) as f, open(outfile, 'w') as outf:

        outf.write('\t'.join(header) + "\n")
        
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            try:
                cols = line[:-1].split('\t')
            
                post = normalize(cols[0])
                post_id = cols[1]
                poster_id = cols[2]
                reply = normalize(cols[3])
                reply_id = cols[4]
                replier_id = cols[5]
                location = cols[6]
                size = cols[7]
                instance_id = cols[8]
                if len(cols) > 10:
                    identity = cols[9]
                    other_identity = cols[10]
                elif len(cols) > 9:
                    identity = cols[9]
                    other_identity = 'unknown'
                else:
                    identity = 'unknown'
                    other_identity = 'unknown'
                    
                start = time.time()
                feats = get_reply_only_social_features(post, reply)
                if len(feats) == 0:
                    continue

                vec = num_feats * [ 0 ] 
                for k, v in feats.items():
                    # Skip  features not seen in training
                    if k not in feature_to_index:
                        continue
                    vec[feature_to_index[k]] = v

                vec = np.array(vec).reshape((1, -1))

                output_row = [ instance_id, location, identity, other_identity ]
                        
                preds = clf.predict(vec)
                pred = str(preds[0])
                output_row.append(pred)
            
                end = time.time()
                time_sum += end-start
            
            except BaseException as e:
                print((repr(e)))
                continue

            try:
                outf.write('\t'.join(output_row) + '\n')

                if line_no % 500 == 0:
                    print(('Saw %d instances, average time: %f (%s)' \
                          % (line_no, time_sum / float(line_no), input_file)))
                    outf.flush()
                    # break
                    # outf.flush()
                
            except BaseException as e:
                print((repr(e)))
                continue

           

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
