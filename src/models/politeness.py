from collections import Counter

P_HEDGES = set([
    "think", "thought", "thinking", "almost",
    "apparent", "apparently", "appear", "appeared", "appears", "approximately", "around",
    "assume", "assumed", "certain amount", "certain extent", "certain level", "claim",
    "claimed", "doubt", "doubtful", "essentially", "estimate",
    "estimated", "feel", "felt", "frequently", "from our perspective", "generally", "guess",
    "in general", "in most cases", "in most instances", "in our view", "indicate", "indicated",
    "largely", "likely", "mainly", "may", "maybe", "might", "mostly", "often", "on the whole",
    "ought", "perhaps", "plausible", "plausibly", "possible", "possibly", "postulate",
    "postulated", "presumable", "probable", "probably", "relatively", "roughly", "seems",
    "should", "sometimes", "somewhat", "suggest", "suggested", "suppose", "suspect", "tend to",
    "tends to", "typical", "typically", "uncertain", "uncertainly", "unclear", "unclearly",
    "unlikely", "usually", "broadly", "tended to", "presumably", "suggests",
    "from this perspective", "from my perspective", "in my view", "in this view", "in our opinion",
    "in my opinion", "to my knowledge", "fairly", "quite", "rather", "argue", "argues", "argued",
    "claims", "feels", "indicates", "supposed", "supposes", "suspects", "postulates"
])


DEFERENCE = set(["great","good","nice","good","interesting","cool","excellent","awesome"])
SORRY = set(["sorry","woops","oops","whoops"])
GROUP = set(["we", "our", "us", "ourselves"])
FIRST_PERSON = set(["i", "my", "mine", "myself"])
SECOND_PERSON = set(["you","your","yours","yourself"])
HELLO = set(["hi","hello","hey"])
FACTUAL = set(["really", "actually", "honestly", "surely"])
QUESTIONS = set(["what","why","who","how"])
STARTS = set(["so","then","and","but","or"])

POLAR = set([
    "is", "are", "was", "were", "am", "have", 
    "has", "had", "can", "could", "shall", 
    "should", "will", "would", "may", "might", 
    "must", "do", "does", "did", "ought", "need", 
    "dare", "if", "when", "which", "who", "whom", "how"
])

pos_filename = "../../resources/liu-positive-words.txt"
neg_filename = "../../resources/liu-negative-words.txt"

POSITIVE_WORDS = set([x.strip() for x in open(pos_filename, encoding = "ISO-8859-1").read().splitlines()])
NEGATIVE_WORDS = set([x.strip() for x in open(neg_filename, encoding = "ISO-8859-1").read().splitlines()])

def get_politeness_indicators(spaced_text, tokens, dep_triples):

    if len(tokens) == 0:
        return {}
    
    features = Counter()

    uniq_terms = set([x.lower() for x in tokens])

    if len(tokens) > 1:
        uniq_terms_from_second = set([x.lower() for x in tokens[1:]])
    else:
        uniq_terms_from_second = set()
        
    if 'please' in uniq_terms_from_second:
        features['politeness:please'] = 1

    if tokens[0].lower() == 'please':
        features['politeness:please_start'] = 1

    for d in dep_triples:
        if d[2][0].lower() in P_HEDGES:
            features['politeness:hedges'] = 1
            break
    
    if tokens[0].lower() in DEFERENCE:
        features['politeness:deference'] = 1
        
    if len(uniq_terms.intersection(SORRY)) > 0:
        features['politeness:apologize'] = 1

    if features['politeness:apologize'] == 0:
        for d in dep_triples:
            if (d[0][0].lower() == 'excuse' and d[2][0] == 'me' and d[1] == 'dobj') or \
               (d[0][0].lower() == 'apologize' and d[2][0] == 'i' and d[1] == 'nsubj')  or \
               (d[0][0].lower() == 'forive' and d[2][0] == 'me' and d[1] == 'dobj'):
                features['politeness:apologize'] = 1
                break


    if len(uniq_terms.intersection(GROUP)) > 0:
        features['politeness:1st_person_plural'] = 1


    if len(uniq_terms_from_second.intersection(FIRST_PERSON)) > 0:
        features['politeness:1st_person'] = 1

    if tokens[0].lower() in SECOND_PERSON:
        features['politeness:2nd_person_start'] = 1

    if tokens[0].lower() in FIRST_PERSON:
        features['politeness:1st_person_start'] = 1

    if tokens[0].lower() in HELLO:
        features['politeness:Indirect_greeting'] = 1
        
    for d in dep_triples:
        if (d[0][0].lower() == 'fact' and d[2][0] == 'in') or \
           (d[0][0].lower() == 'point' and d[2][0] == 'the') or \
           (d[0][0].lower() == 'reality' and d[2][0] == 'in') or \
           (d[0][0].lower() == 'truth' and d[2][0] == 'in'):
           
            features['politeness:factuality'] = 1
            break

    if features['politeness:factuality'] == 0:
        if len(uniq_terms.intersection(FACTUAL)) > 0:
            features['politeness:factuality'] = 1

    if (tokens[0].lower() in QUESTIONS) or (len(tokens) > 1 and tokens[1].lower() in QUESTIONS):
        features['politeness:direct_question'] = 1

    if tokens[0].lower() in STARTS:
        features['politeness:direct_start'] = 1

    if len(tokens) > 2 and tokens[0].lower() == 'by' \
       and tokens[1].lower() == 'the' and tokens[2].lower() == 'way':
        features['politeness:indirect_btw'] = 1

    if len(uniq_terms_from_second.intersection(SECOND_PERSON)) > 0:
        features['politeness:2nd_person'] = 1


    if tokens[0].lower() in POLAR:
        features['politeness:initial_polar'] = 1

    for d in dep_triples:
        if d[1] == 'aux' and (d[0][0].lower() in POLAR or d[2][0].lower() in POLAR):
            features['politeness:aux_polar'] = 1
            break


    if ' could you ' in spaced_text or ' would you ' in spaced_text:
        features['politeness:subjunctive'] = 1        

    if ' can you ' in spaced_text or ' will you ' in spaced_text:
        features['politeness:indicative'] = 1        

    if len(uniq_terms.intersection(P_HEDGES)) > 0:
        features['politeness:has_hedge'] = 1
        features['politeness:num_hedge'] = len(uniq_terms.intersection(P_HEDGES))

    if len(uniq_terms.intersection(NEGATIVE_WORDS)) > 0:
        features['politeness:has_negative'] = 1
        features['politeness:num_negative'] = len(uniq_terms.intersection(NEGATIVE_WORDS))
        
    if len(uniq_terms.intersection(POSITIVE_WORDS)) > 0:
        features['politeness:has_positive'] = 1
        features['politeness:num_positive'] = len(uniq_terms.intersection(POSITIVE_WORDS))
      
        
    return features
