import sys
from post_gender_age_features import get_text_features
import time
from collections import *
import re
import numpy as np
import random

def main():

    input_file = sys.argv[1]
    outfile = sys.argv[2]
    attr = 'gender'
    
    MIN_FREQ = 5
    UNIGRAM_MIN_FREQ = 5
    BIGRAM_MIN_FREQ = 50
    TRIGRAM_MIN_FREQ = 100
    MAX_NGRAM_FEATS = 20000
    
    max_lines = 1000000
    if len(sys.argv) > 4:
        max_lines = int(sys.argv[4])
        
    y = []
    X_dicts = []

    time_sum = 0

    # Get a balanced set of instances
    class_to_insts = defaultdict(set)
    with open(input_file) as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue

            if line_no % 500000 == 0:
                print('Loaded %d class counts' % (line_no))
            
            cols = line[:-1].split('\t')

            inst_id = cols[-3]
            clazz = cols[-2]
            clazz = [x for x in clazz.split(',') if attr in x][0]
            
            class_to_insts[clazz].add(inst_id)

    # If we have enough instances, it's the maximum number of instances, equally
    # divided
    min_class_count = int(max_lines / len(class_to_insts))
    for z, c in class_to_insts.items():
        if len(c) < min_class_count:
            min_class_count = len(c)

            
    print("choosing %d instances for %d classes (max: %d)"  \
          % (min_class_count, len(class_to_insts), (max_lines / len(class_to_insts))))
    insts_to_use = set()
    for clazz, insts in class_to_insts.items():
        tmp = list(insts)
        random.shuffle(tmp)
        for i in tmp[:min_class_count]:
            insts_to_use.add(i)

    print("converting instances to feature vectors")
    with open(input_file) as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            
            cols = line[:-1].split('\t')
            # All four social scores
            
            inst_id = cols[-3]
            if inst_id not in insts_to_use:
                continue

            post = normalize(cols[0])            
            clazz = cols[-2]
            
            if len(post) == 0:
                continue
            
            start = time.time()
            feats = get_text_features(post)
            end = time.time()
            time_sum += end-start

            if len(feats) == 0:
                continue
            
            y.append(cols[-2:])
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

    # Take a maximum number of uni/bi/tri features
    print("Truncating n-gram features")
    uni_fc = Counter({f: c for f, c in feature_counts.items() if f.startswith('unigram:')})
    bi_fc = Counter({f: c for f, c in feature_counts.items() if f.startswith('bigram:')})
    tri_fc = Counter({f: c for f, c in feature_counts.items() if f.startswith('trigram:')})

    uni_feats = set([f for f, c in uni_fc.most_common(MAX_NGRAM_FEATS)])
    bi_feats = set([f for f, c in bi_fc.most_common(MAX_NGRAM_FEATS)])
    tri_feats = set([f for f, c in tri_fc.most_common(MAX_NGRAM_FEATS)])
            
    # Convert to feature matrix and dump into a classifier
    print("filtering other features")
    feature_to_index = {}
    for d in X_dicts:
        for k in d.keys():
            if k not in feature_to_index:
                if k.startswith('unigram:'):
                    if k in uni_feats:
                        feature_to_index[k] = len(feature_to_index)
                elif k.startswith('bigram:'):
                    if k in bi_feats:
                        feature_to_index[k] = len(feature_to_index)
                elif k.startswith('trigram:'):
                    if k in tri_feats:
                        feature_to_index[k] = len(feature_to_index)
                else:
                    if feature_counts[k] >= MIN_FREQ:
                        feature_to_index[k] = len(feature_to_index)

    num_feats = len(feature_to_index)                
    print(('Using %d features (out of %d)' % (num_feats, len(feature_counts))))
    
    
    with open(outfile + '.features.tsv', 'w') as outf:
        for k,v in feature_to_index.items():
            outf.write('%s\t%d\n' % (k, v))

    print('Saving sparse X')
    with open(outfile + '.sparse.tsv', 'w') as outf:        
        for j, d in enumerate(X_dicts):
            row = []
            row.extend(y[j])
            row.append(num_feats)
            #social = '\t'.join(y[j])
            
            for k, v in d.items():
                # Skip rare features
                if k not in feature_to_index:
                    continue
                i = feature_to_index[k]
                #vec[i] = v
                row.append(i)
                row.append(v)
            outf.write('\t'.join([str(x) for x in row]) + '\n')
            

    # TODO: Use median feature value for unknowns somehow
    if False:
        print("creating X")
        X = []            
        for j, d in enumerate(X_dicts):
            social = '\t'.join(y[j])
            
            vec = num_feats * [ 0 ]
            for k, v in d.items():
                # Skip rare features
                if k not in feature_to_index:
                    continue
                i = feature_to_index[k]
                vec[i] = v
                X.append(np.array(vec))

                
        X = np.array(X)
        print(len(y), len(X))
    
        print('Saving zero X')
        with open(outfile + '.zero.tsv', 'w') as outf:        
            for j, row in enumerate(X):
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
