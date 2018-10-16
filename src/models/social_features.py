from nltk.parse.corenlp import CoreNLPDependencyParser
from textblob import TextBlob
import re
import string

import nltk
from nltk.corpus import stopwords
from textstat.textstat import textstat
from collections import Counter
from collections import defaultdict

import datrie

import gensim
from gensim import utils
import numpy as np

import trie_search
from unidecode import unidecode

from empath import Empath

from politeness import get_politeness_indicators
from support import get_support_indicators

from nltk.tokenize.stanford import StanfordTokenizer

import random
import gzip

import sys

import spacy


import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
empath = Empath()


en_nlp = spacy.load('en')

EN_STOPWORDS = set(stopwords.words('english'))

HEDGES = set([
    'seem', 'tend', 'look like', 'appear to be',
    'think', 'believe', 'doubt', 'be sure', 'indicate', 'suggest'
    'believe', 'assume', 'suggest', 'would',
    'may', 'might', 'could', 'often', 'sometimes', 'usually',
    'probably', 'possibly', 'perhaps', 'conceivably',
    'probable', 'possible', 'assumption', 'possibility', 'probability',

    'practically', 'apparently', 'virtually', 'basically', 'approximately',
    'roughly', 'somewhat', 'somehow', 'partially', 'assume',
    'assumes', 'assumed', 'appear', 'appears', 'appeared', 'seem', 'seems'
    'seemed', 'suppose', 'supposes', 'supposed', 'guess', 'guesses', 
    'guessed', 'estimate', 'estimates', 'estimated', 'speculate', 'speculates'
    'speculated', 'suggest', 'suggests', 'suggested', 'likely', 'maybe'
    'perhaps', 'unsure', 'probable', 'unlikely', 'possibly', 'possible',
    'evidently', 'fairly', 'hopefully', 'mainly', 'mostly', 'overall',
    'presumably', 'conceivably',    
])

RE_LEXICONS = {}
LEXICONS = {
    "ARGUMENTATION_NOUN": set(["assumption", "belief", "hypothesis", "hypotheses", "claim", "conclusion", "confirmation", "opinion", "recommendation", "stipulation", "view"]),

    "PROBLEM_NOUN": set(["Achilles heel", "caveat", "challenge", "complication", "contradiction", "damage", "danger", "deadlock", "defect", "detriment", "difficulty", "dilemma", "disadvantage", "disregard", "doubt", "downside", "drawback", "error", "failure", "fault", "foil", "flaw", "handicap", "hindrance", "hurdle", "ill", "inflexibility", "impediment", "imperfection", "intractability", "inefficiency", "inadequacy", "inability", "lapse", "limitation", "malheur", "mishap", "mischance", "mistake", "obstacle", "oversight", "pitfall", "problem", "shortcoming", "threat", "trouble", "vulnerability", "absence", "dearth", "deprivation", "lack", "loss", "fraught", "proliferation", "spate"]),

    "QUESTION_NOUN": set(["question", "conundrum", "enigma", "paradox", "phenomena", "phenomenon", "puzzle", "riddle"]),
    
    "SOLUTION_NOUN": set(["answer", "accomplishment", "achievement", "advantage", "benefit", "breakthrough", "contribution", "explanation", "idea", "improvement", "innovation", "insight", "justification", "proposal", "proof", "remedy", "solution", "success", "triumph", "verification", "victory"]),

    "TRADITION_NOUN": set(["acceptance", "community", "convention", "disciples", "disciplines", "folklore", "literature", "mainstream", "school", "tradition", "textbook"]),

    "BEFORE_ADJ": set(["earlier", "initial", "past", "previous", "prior"]),
    
    "CONTRAST_ADJ": set(["different", "distinguishing", "contrary", "competing", "rival"]),

    "CONTRAST_ADV": set(["differently", "distinguishingly", "contrarily", "otherwise", "other than", "contrastingly", "imcompatibly", "on the other hand", ]),

    "TRADITION_ADJ": set(["better known", "better-known", "cited", "classic", "common", "conventional", "current", "customary", "established", "existing", "extant", "available", "favourite", "fashionable", "general", "obvious", "long-standing", "mainstream", "modern", "naive", "orthodox", "popular", "prevailing", "prevalent", "published", "quoted", "seminal", "standard", "textbook", "traditional", "trivial", "typical", "well-established", "well-known", "widelyassumed", "unanimous", "usual"]),

    "HELP_NOUN": set(['help', 'aid', 'assistance', 'support' ]),

    'DOWNTONERS': set([ 'almost', 'barely', 'hardly', 'merely', 'mildly', 'nearly', 'only', 'partially', 'partly', 'practically', 'scarcely', 'slightly', 'somewhat', ]),

    'AMPLIFIERS': set([ 'absolutely', 'altogether', 'completely', 'enormously', 'entirely', 'extremely', 'fully', 'greatly', 'highly', 'intensely', 'strongly', 'thoroughly', 'totally', 'utterly', 'very', ]),

    
    'PUBLIC_VERBS': set(['acknowledge', 'admit', 'agree', 'assert', 'claim', 'complain', 'declare', 'deny', 'explain', 'hint', 'insist', 'mention', 'proclaim', 'promise', 'protest', 'remark', 'reply', 'report', 'say', 'suggest', 'swear', 'write', ]),
    
    'PRIVATE_VERBS': set([ 'anticipate', 'assume', 'believe', 'conclude', 'decide', 'demonstrate', 'determine', 'discover', 'doubt', 'estimate', 'fear', 'feel', 'find', 'forget', 'guess', 'hear', 'hope', 'imagine', 'imply', 'indicate', 'infer', 'know', 'learn', 'mean', 'notice', 'prove', 'realize', 'recognize', 'remember', 'reveal', 'see', 'show', 'suppose', 'think', 'understand', ]),
    
    'SUASIVE_VERBS': set([ 'agree', 'arrange', 'ask', 'beg', 'command', 'decide', 'demand', 'grant', 'insist', 'instruct', 'ordain', 'pledge', 'pronounce', 'propose', 'recommend', 'request', 'stipulate', 'suggest', 'urge', ]),

    'EXISTENTIAL_INDEFINITES': set(['whatever','whoever','whichever','whenever','someone','somebody','something','anything','somehow','anyhow','anything','anyone','anybody','however','sometime','someday']),

    'NEGATION': { 'not', 'never', 'nor', "didn't", "couldn't", "won't", "shouldn't", "doesn't",
                 "isn't", "wasn't", "wouldn't", "couldn't",  "don't", "no", "nobody", "nothing",
                 "neither", "nowhere", "hardly", "barely", "scarcely", "n't" }
    
}


              
FIRST_PERSON_PRONOUN = set(["we", "i", "ours", "mine"])
THIRD_PERSON_PRONOUN = set(["they", "he", "she", "theirs", "hers", "his"])

WORD_TO_LOG_FREQS = Counter()
MAX_WORD_FREQS=500000
# Use a flag here for fast loading
if True:
    print('Loading log-transformed Google NGram counts')
    ngram_count_file='../../resources/google-ngram-freqs.no-pos.sorted.tsv.gz'
    with gzip.open(ngram_count_file, mode='rt') as f:
        for line_no, line in enumerate(f):
            cols = line[:-1].split('\t')
            word = cols[0]
            if '_' in word:
                continue
            WORD_TO_LOG_FREQS[word] = float(cols[1])
            if len(WORD_TO_LOG_FREQS) >= MAX_WORD_FREQS:
                break

WORD2VEC = None
# If debugging, save a few minutes by avoiding the word2vec parts of the
# pipeline by switching this conditional to 'False' (everything else should
# still just work)
if False:
    print('Loading word2vec data')
    word2vec_file = '../../resources/GoogleNews-vectors-negative300.bin.gz'
    WORD2VEC = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    print('...done loading word2vec data')

PHRASE_FORMALITIES = {}
PHRASE_FORMALITIES_TRIE = None
if True:
    print('Loading phrase formalities')
    # This is naacl-2015-style-scores/formality/automatic/phrase-scores
    phrase_formalities_file = '../../resources/phrase-formality-scores.tsv'
    with open(phrase_formalities_file) as f:
        for line in f:
            cols = line[:-1].split('\t')
            formality = float(cols[0])
            phrase = ' ' + cols[1] + ' '
            PHRASE_FORMALITIES[phrase] = formality

        PHRASE_FORMALITIES_TRIE = trie_search.TrieSearch(list(PHRASE_FORMALITIES.keys()))


PHRASE_CONCRETENESS = {}
PHRASE_CONCRETENESS_TRIE = None
if True:
    print('Loading phrase concreteness')
    phrase_concreteness_file = '../../resources/AC_ratings_google3m_koeper_SiW.csv.gz'
    with gzip.open(phrase_concreteness_file) as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                continue
            # print line
            cols = line[:-1].split('\t')
            concreteness = float(cols[1])
            phrase = ' ' + cols[0].replace('_', ' ') + ' '
            PHRASE_CONCRETENESS[phrase] = concreteness

        PHRASE_CONCRETENESS_TRIE = trie_search.TrieSearch(list(PHRASE_CONCRETENESS.keys()))

        
CURSE_WORDS = set()
LIWC_RE_LEXICONS = {}
LIWC_LEXICONS = {}
if True:
    print('Loading lexicons')
    curse_word_file = '../../resources/curse-words.txt'
    with open(curse_word_file) as f:
        for line in f:
            CURSE_WORDS.add(line[:-1])
    LEXICONS['CURSE_WORDS'] = CURSE_WORDS

    base_lex_dir = '../../resources/lexicons/'
    
    for fname in ['abstract.txt', 'concrete.txt', 'negative.txt', 'positive.txt', 'swearWords.txt']:
        lex_name = fname.split('.')[0]
        words = set()
        with open(base_lex_dir + fname, encoding = "ISO-8859-1") as f:
            for line in f:
                words.add(line.strip().lower())
        LEXICONS['Lexicon:' + lex_name] = words

    with open(base_lex_dir + 'nrc-emotion.txt') as f:
        emo2terms = defaultdict(set)
        for line in f:
            cols = line.strip().split('\t')
            if int(cols[2]) > 0:
                emo2terms['NRC_' + cols[1]].add(cols[0])
        LEXICONS.update(emo2terms)

    # LIWC can't be distributed so we skip it if it's not there
    if os.path.isfile(base_lex_dir + 'en_liwc.txt'):
        with open(base_lex_dir + 'en_liwc.txt') as f:
            for l in f:
                category = "LIWC_" + l.split(':')[0].strip()
                cat_list = l.strip().split()[1:]
                re_list = []
                word_list = set()
                for x in cat_list:
                    if '*' in x:                    
                        re_list.append(x.lower()[:len(x)-1])
                    else:
                        word_list.add(x.lower())

                trie = datrie.Trie(string.ascii_lowercase)
                for s in re_list:
                    trie[s] = 1
                RE_LEXICONS[category] = trie # re_list
                LEXICONS[category] = word_list
            
                LIWC_RE_LEXICONS[category] = trie # re_list
                LIWC_LEXICONS[category] = set(word_list)
    else:
        # Make sure the code works by matching each category with an empty trie
        print('LIWC not seen so skipping those features; if classifying new ' + \
              'instances,\nplease be sure your classifier was trained *without* LIWC')
        empty_trie = datrie.Trie(string.ascii_lowercase)
        LIWC_RE_LEXICONS = defaultdict(lambda: return empty_trie)
        # Empty set
        LIWC_LEXICONS = defaultdict(set)
                
                    
            


def get_tokens(text):
    sentences = en_nlp(text).sents
    all_tokens = []
    for s in sentences:
        all_tokens.extend(to_token_representation(s))
    return all_tokens
            
def get_reply_only_social_features(post, reply):

    reply_feats, reply_tokens = get_social_features_internal(reply)
    # Need post tokens still
    post_tokens = get_tokens(post)

    combined_feats = Counter()
    for f, v in reply_feats.items():
        combined_feats['reply:' + f] = v
        
    # Number of shared non-stop words between post-and-reply
    s1 = set([t for t in reply_tokens if t not in EN_STOPWORDS])
    s2 = set([t for t in post_tokens if t not in EN_STOPWORDS])
    combined_feats['content_overlap'] = len(s1.intersection(s2)) / float(max(1, len(s1.union(s2))))
    combined_feats['content_matching'] = len(s1.intersection(s2)) / float(max(1, len(s2)))

    # Stopword matching (just common words)
    s1 = set([t for t in reply_tokens if t in EN_STOPWORDS])
    s2 = set([t for t in post_tokens if t in EN_STOPWORDS])
    combined_feats['stopword_overlap'] = len(s1.intersection(s2)) / float(max(1, len(s1.union(s2))))
    combined_feats['stopword_matching'] = len(s1.intersection(s2)) / float(max(1, len(s2)))

    # Function word matching
    s1 = set()
    s2 = set()

    funct_prefix_trie = LIWC_RE_LEXICONS['LIWC_Funct']
    for t in post_tokens:
        try:
            funct_prefix_trie.longest_prefix(t)
            s1.add(t)
        except KeyError as ke:
            pass
    for t in reply_tokens:
        try:
            funct_prefix_trie.longest_prefix(t)
            s2.add(t)
        except KeyError as ke:
            pass
    

    s1 = s1.union(LIWC_LEXICONS['LIWC_Funct'].intersection(set(post_tokens)))
    s2 = s2.union(LIWC_LEXICONS['LIWC_Funct'].intersection(set(reply_tokens)))

    combined_feats['funcword_overlap'] = len(s1.intersection(s2)) / float(max(1, len(s1.union(s2))))
    combined_feats['funcword_matching'] = len(s1.intersection(s2)) / float(max(1, len(s2)))
        
    return combined_feats


def get_social_features(post, reply):
    reply_feats, reply_tokens = get_social_features_internal(reply)
    post_feats, post_tokens = get_social_features_internal(post)

    combined_feats = Counter()
    for f, v in reply_feats.items():
        combined_feats['reply:' + f] = v
    for f, v in post_feats.items():
        combined_feats['post:' + f] = v
        
    # Number of shared non-stop words between post-and-reply
    s1 = set([t for t in reply_tokens if t not in EN_STOPWORDS])
    s2 = set([t for t in post_tokens if t not in EN_STOPWORDS])
    combined_feats['content_overlap'] = len(s1.intersection(s2)) / float(max(1, len(s1.union(s2))))
    combined_feats['content_matching'] = len(s1.intersection(s2)) / float(max(1, len(s2)))

    # Stopword matching (just common words)
    s1 = set([t for t in reply_tokens if t in EN_STOPWORDS])
    s2 = set([t for t in post_tokens if t in EN_STOPWORDS])
    combined_feats['stopword_overlap'] = len(s1.intersection(s2)) / float(max(1, len(s1.union(s2))))
    combined_feats['stopword_matching'] = len(s1.intersection(s2)) / float(max(1, len(s2)))

    # Function word matching
    s1 = set()
    s2 = set()

    funct_prefix_trie = LIWC_RE_LEXICONS['LIWC_Funct']
    for t in post_tokens:
        try:
            funct_prefix_trie.longest_prefix(t)
            s1.add(t)
        except KeyError as ke:
            pass
    for t in reply_tokens:
        try:
            funct_prefix_trie.longest_prefix(t)
            s2.add(t)
        except KeyError as ke:
            pass
    

    s1 = s1.union(LIWC_LEXICONS['LIWC_Funct'].intersection(set(post_tokens)))
    s2 = s2.union(LIWC_LEXICONS['LIWC_Funct'].intersection(set(reply_tokens)))

    combined_feats['funcword_overlap'] = len(s1.intersection(s2)) / float(max(1, len(s1.union(s2))))
    combined_feats['funcword_matching'] = len(s1.intersection(s2)) / float(max(1, len(s2)))
        
    return combined_feats



def to_parsed_representations(spacy_sent):

    def get_triples(n, triples):
        n_tag = n.tag_
        if len(n_tag) == 0:
            n_tag = ' '
        elif n_tag == 'BES':
            n_tag = 'VBE'
        elif n_tag == 'HVS':
            n_tag == 'VHV'
        elif n_tag == 'NFP':
            n_tag == 'Punct'
            
        for c in n.children:
            c_tag = c.tag_
            if len(c_tag) == 0:
                c_tag = ' '
            elif c_tag == 'BES':
                c_tag = 'VBE'
            elif c_tag == 'HVS':
                c_tag == 'VHV'
            elif c_tag == 'NFP':
                c_tag == 'Punct'
                
            t = ((n.orth_, n_tag), c.dep_, (c.orth_, c_tag))
            if len(c.dep_) > 0:
                triples.append(t)
            get_triples(c, triples)
         
    triples = []
    get_triples(spacy_sent.root, triples)
    
    tokens = []
    pos_tags = []
    for t in spacy_sent:
        if t.orth_ == ' ':
            continue
        tokens.append(t.orth_)
        t_tag = t.tag_
        if len(t_tag) == 0:
            t_tag = ' '
        elif t_tag == 'BES':
            t_tag = 'VBE'
        elif t_tag == 'HVS':
            t_tag == 'VHV'
        elif t_tag == 'NFP':
            t_tag == 'Punct'
        
        pos_tags.append(t_tag)
    return tokens, pos_tags, triples

def to_token_representation(spacy_sent):
    
    tokens = []
    for t in spacy_sent:
        if t.orth_ == ' ':
            continue
        tokens.append(t.orth_)
    return tokens

    

def get_social_features_internal(text):

    sentences = en_nlp(text).sents

    features = Counter()

    total_num_tokens = 0

    all_tokens = []
    all_pos_tags = []
    all_spaced_text = ''
    
    parse_feat_names = set()
    whole_text_feat_names = set()

    list_of_sentence_tokens = []
    
    all_parse_feats = Counter()
    
    # Manually count this due to the fact that spaCy returns a generator and not
    # a list, which breaks calling len() since we're bastardizing the object
    # type of sentences
    num_sents = 0

    
    for sentence in sentences:       
        num_sents += 1
        
        tokens, pos_tags, dep_parse_triples = to_parsed_representations(sentence)

        all_tokens.extend(tokens)
        all_pos_tags.extend(pos_tags)
        list_of_sentence_tokens.append(tokens)

        spaced_text = ' ' + ' '.join(tokens) + ' '
        all_spaced_text += spaced_text
        
        parse_feats = get_sentence_features(spaced_text, tokens, pos_tags, dep_parse_triples)
        for k, v in parse_feats.items():
            all_parse_feats[k] += v
        parse_feat_names |= set(parse_feats.keys())

    whole_text_feats, word_features = get_whole_text_features(all_spaced_text, all_tokens,
                                                              list_of_sentence_tokens, all_pos_tags)
    whole_text_feat_names |= set(whole_text_feats.keys())
    

    num_sents = float(num_sents)
    num_tokens = float(len(all_tokens))

    features.update(all_parse_feats)
    features.update(whole_text_feats)
    features.update(word_features)

    # Normalize by # of sentences
    for feat in parse_feat_names:
        features[feat] /= num_sents

    # Normalize by # of words
    if num_tokens > 0:
        for feat in whole_text_feat_names:
            features[feat] /= num_tokens

    # Track min/max sentence lengths
    max_sent_length = min([len(x) for x in list_of_sentence_tokens])
    min_sent_length = max([len(x) for x in list_of_sentence_tokens])

    return features, all_tokens
        

def get_sentence_features(spaced_text, tokens, pos_tags, dep_parse_triples):

    features = Counter()
    
    # Binary indicator for whether the sentence is all lower case
    if spaced_text == spaced_text.lower():
        features['num_lower_case_sents'] += 1
        
    # Indicator for whether the first word is capitalized
    if tokens[0][0].isupper():
        features['num_sents_with_first_word_capitalized'] += 1


    for governor, dep, dependent in dep_parse_triples:
        # (gov, typ, dep) tuples with gov and dep backed off to their POS tags
        features['dependency:' + governor[1] + ':' + dep + ':' + dependent[1]] = 1
        
        #  (gov, typ) tuples with gov and dep backed off to their POS tags
        features['dependency:' + governor[1] + ':' + dep + ':'] = 1
        
        # (typ, dep) tuples with gov and dep backed off to their POS tags
        features['dependency:' + ':' + dep + ':' + dependent[1]] = 1
                 
        # (gov dep) tuples with gov and dep backed off to their POS tags
        features['dependency:' + governor[1] + ':' + ':' + dependent[1]] = 1
        

    features['mean_sentence_length_in_words'] += len(tokens)
    features['mean_sentence_length_in_chars'] += len(spaced_text) - 2


    politeness_features = get_politeness_indicators(spaced_text, tokens, dep_parse_triples)
    features.update(politeness_features)

    support_features = get_support_indicators(spaced_text, tokens, dep_parse_triples)
    features.update(support_features)    

    if pos_tags[0][0] == 'V':
        features['sentence_starts_with_verb'] += 1
        
    return features
                       


        
def get_whole_text_features(spaced_text, tokens, list_of_sentence_tokens, pos_tags):    

    
   
    features = Counter()
    word_features = Counter()

    # word_counts = Counter()
    
    num_passive = 0
    for i, pos in enumerate(pos_tags):
        if pos == 'VBZ' and i + 1 < len(pos_tags) and pos_tags[i+1] == 'VBN':
            num_passive += 1
    features['percent_passive'] = num_passive
    
    num_hedges = 0
    num_fpp = 0
    num_tpp = 0

    # % words that are ALL CAPS words
    num_all_caps = 0
    cur_all_caps_seq_len = 0
    # length of longest ALL CAPS span
    longest_all_caps_seq_len = 0
    
    #for i, t in enumerate(tokens):
    for sent_tokens in list_of_sentence_tokens:
        for i, t in enumerate(sent_tokens):


            if t == t.upper():
                num_all_caps += 1
                cur_all_caps_seq_len += 1
                if cur_all_caps_seq_len > longest_all_caps_seq_len:
                    longest_all_caps_seq_len = cur_all_caps_seq_len
            else:
                cur_all_caps_seq_len = 0
            
            t = t.lower()
            
            # Number of hedge words, normalized by the length of the
            # sentence. Based on a list of hedge words taken from a combination
            # of online sources.
            if t in HEDGES:
                num_hedges += 1
            # Number of 1st person pronouns, normalized by the length of the
            # sentence
            elif t in FIRST_PERSON_PRONOUN:
                num_fpp += 1
            # Number of 3rd person pronouns, normalized by the length of the
            # sentence
            elif t in THIRD_PERSON_PRONOUN:
                num_tpp += 1
    
            # Unigrams, as 1-hot features
            features['unigram:' + t] = 1
            # word_counts[t] += 1

            # Bigrams, as 1-hot features
            if i < len(sent_tokens) - 1:
                features['bigram:' + t + "_" + sent_tokens[i+1]] = 1            

            # Trigrams, as 1-hot features
            if i < len(sent_tokens) - 2:
                features['trigram:' + t + "_" + sent_tokens[i+1] + '_' + sent_tokens[i+2]] = 1

                
            # While we're iterating over the tokens, check each against the regular
            # expressions used in the lexicons (this is typically LIWC)
            for lex, prefix_trie in RE_LEXICONS.items():
                try:
                    # This is just for LIWC, which only has suffix wildcards so
                    # we can do a more efficient substring search                    
                    prefix_trie.longest_prefix(t)
                    features[lex] += 1
                except KeyError as ke:
                    pass
            

            for lex, vocab in LEXICONS.items():
                if t in vocab:
                    features[lex] += 1


        # Finally, add extra bigrams and trigam feats for the beginning and end
        # of sentences
        features['bigram:<bos>_' + sent_tokens[0]] = 1
        features['bigram:' + sent_tokens[-1] + '_<eos>'] = 1            

        # Trigrams, as 1-hot features
        if i < len(sent_tokens) > 1:
            features['trigram:<bos>_' + sent_tokens[0] + '_' + sent_tokens[1]] = 1
            features['trigram:' + sent_tokens[-2] + '_' + sent_tokens[-1] + "_<eos>"] = 1

                    

    features['num_all_caps'] = num_all_caps 
    word_features['longest_all_caps_seq_len'] = longest_all_caps_seq_len
           
            
    if num_hedges > 0:
        features['num_hedges'] = num_hedges
    if num_fpp > 0:
        features['num_first_party_pronouns'] = num_fpp
    if num_tpp > 0:
        features['num_third_party_pronouns'] = num_tpp


    # Subjectivity score of the sentence, according the the TextBlob sentiment module
    tb_text = TextBlob(spaced_text)
    subjectivity = tb_text.sentiment.subjectivity
    features['subjectivity'] = subjectivity

    # Binary indicator for whether the sentiment is positive or negative,
    # according the the TextBlob sentiment module
    if tb_text.sentiment.polarity > 0:
        features['is_positive_sentiment'] = 1

    # Number of capitalized words, not including 'I'
    wl_sum = 0
    for t in tokens:
        if t[0].isupper() and t != 'I':
            features['num_capitalized'] += 1
            
        # Average word length, in characters
        wl_sum += len(t)
    features['mean_word_length'] += wl_sum

    n_matches = 0
    formality_sum = 0
    for pattern, start_idx in PHRASE_FORMALITIES_TRIE.search_all_patterns(spaced_text):
        #print pattern
        formality_sum += PHRASE_FORMALITIES[pattern]
        n_matches += 1
        
    if n_matches > 0:
        word_features['average_phrase_formality'] = formality_sum / n_matches       


    n_matches = 0
    ac_sum = 0
    for pattern, start_idx in PHRASE_CONCRETENESS_TRIE.search_all_patterns(spaced_text):
        #print pattern
        ac_sum += PHRASE_CONCRETENESS[pattern]
        n_matches += 1
        
    if n_matches > 0:
        word_features['average_phrase_concreteness'] = ac_sum / n_matches       

        
        
    for c in spaced_text:
        if c == '!':
            word_features['num_exclamations'] += 1
        elif c == '?':
            word_features['num_question_marks'] += 1
        elif c == '?':
            word_features['num_question_marks'] += 1

                
    # Number of '...' in the sentence
    word_features['num_ellipses'] += len(spaced_text.split('...')) - 1  
    

    # Average word log frequency according the Google Ngram corpus, not including stop words
    num_non_stop = 0
    avg_log_freq = 0
    for t in tokens:
        if t in EN_STOPWORDS:
            continue
        avg_log_freq += WORD_TO_LOG_FREQS[t]
        num_non_stop += 1
    if num_non_stop > 0:
        word_features['average_word_log_freq'] = avg_log_freq / float(num_non_stop)
                             
    
    # Length of the sentence, in characters
    word_features['num_characters'] = len(spaced_text) - 2
    
    
    # Flesch-Kincaid Grade Level
    # Check this: https://pypi.python.org/pypi/textstat
    try:
        fk_grade_level = textstat.flesch_kincaid_grade(spaced_text)
    except:
        print(("weird error for FK Grade on '%s'" % (' '.join(tokens))))
        fk_grade_level = 0
    features['fk_grade_level'] = fk_grade_level    
    
 
    # The number of occurrences of each POS tag in the sentence, normalized by
    # the length of the sentence
    pos_tag_counts = Counter(pos_tags)
    for tag, count in pos_tag_counts.items():
        # print(tag)
        # tag = tag[1]
        features['POS:' + tag] += count
    
    empath_feats = empath.analyze(spaced_text)
    for k, v in empath_feats.items():
        if v > 0:
            word_features[k] = v        


                
            
    # We compute the sentence vector to be the average of the precomputed w2v word
    # vectors in the sentence. Words for which there is not pre-computed vector
    # are skipped.
    #
    # Also calculate the averge word vector separately for nouns, verbs, adjectives, adverbs
    #
    if WORD2VEC is not None:
        
        all_vecs = []
        pos_vecs = { 'N': [], 'V': [], 'J': [], 'R': [] }
        
        for i, t in enumerate(tokens):
            if t not in WORD2VEC.vocab:
                continue

            vec = WORD2VEC[t]
            all_vecs.append(vec)

            pos = pos_tags[i][0]
            if pos in pos_vecs:
                pos_vecs[pos].append(vec)

        if len(all_vecs) > 0:
            if len(all_vecs) == 1:
                mean_vec = all_vecs[0]
            else:   
                mean_vec = np.mean(all_vecs, axis=0)

            for i, x in enumerate(mean_vec):
                word_features['word2vec_%d' % i] = x

            for pos, vecs in pos_vecs.items():
                if len(vecs) == 0:
                    continue
                if len(vecs) == 1:
                    mean_vec = vecs[0]
                else:
                    mean_vec = np.mean(vecs, axis=0)

                for i, x in enumerate(mean_vec):
                    word_features['%s_word2vec_%d' % (pos, i)] = x        

    
    features['laughs'] = get_laugh_freq(spaced_text)
    features['disfluencies'] = get_disfluencies_freq(spaced_text)
                
    return features, word_features


def get_laugh_freq(text):
    return len(re.findall(r'\brofl\b|\bl(ol)+\b|\b(h+e+h+e+)+\b|\b(h+a+)+\b', text))

def get_disfluencies_freq(text):
    return len(re.findall(r'\bum+\b|\buh+\b|\bhuh+\b|\bhmm+\b|\boh+\b', text))


if __name__ == '__main__':
    main()

    
