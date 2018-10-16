from collections import Counter
from collections import defaultdict
import re
import spacy

en_nlp = spacy.load('en')

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


DEFERENCE = set(["great","good","nice","good","interesting","cool",
                 "excellent","awesome", "fantastic", "wonderful"])
SORRY = set(["sorry","woops","oops","whoops"])
GROUP = set(["we", "our", "us", "ourselves"])
FIRST_PERSON = set(["i", "my", "mine", "myself"])
SECOND_PERSON = set(["you","your","yours","yourself"])
HELLO = set(["hi","hello","hey"])
FACTUAL = set(["really", "actually", "honestly", "surely"])
QUESTIONS = set(["what","why","who","how"])
STARTS = set(["so","then","and","but","or"])

W_WORDS = { 'who', 'which', 'whose', 'what' } # Nouns

INTERROGATIVES = { 'who', 'which', 'whose', 'what', 'why', 'how', 'when', 'whom', 'where', 'did', 'do',  }

POLAR = set([
    "is", "are", "was", "were", "am", "have", 
    "has", "had", "can", "could", "shall", 
    "should", "will", "would", "may", "might", 
    "must", "do", "does", "did", "ought", "need", 
    "dare", "if", "when", "which", "who", "whom", "how"
])

ADVICE_STARTS = [
    ' you need ', ' you should ',  ' you must ', ' you might '
    ]

SUGGESTION = {
    'recommend', 'suggest', 'propose', 'advise', 'advocate', 'encourage', 
}

SUGGESTION_NOUN = {
    'recommendation', 'suggestion', 'proposal', 'idea', 'advocate', 'encourage', 'insight', 'answer',
    'observation', 'point',
}


RELATED = {
    'similar', 'related', 'analogous', 'comparable', 
    }

REFERRAL_PHRASES = {
    ' please see ', ' check out ', ' look at ' ,
}

SITUATION = {
    'situation', 'circumstances', 'circumstance', 'issue', 'problem', 'predicament',
    'dilemma', 'crisis', 'position', 'quandry', 'status', 'plight', 'incident',

    'situations', 'circumstances',  'issues', 'problems', 'predicaments',
    'dilemmas', 'crises', 'positions', 'quandries', 'statuses', 'plights', 'incidents', 
    }

EXPLANATION = {
    'reason', 'explanation',  'evidence', 'justification',
    'reasons', 'explanations',  'evidences', 'justifications', 
    }

HELP = {
    'help', 'arrange', 'assist', 'aid', 
    }

VALIDATION_REFERENTS = { 'you', 'that', }
VALIDATION = { 'right', 'correct', 'true', }
VALIDATION_VERBS = {'work', 'works', 'fixes' }
INITIAL_VALIDATION = { 'right', 'correct', 'sure', 'yeah', 'yup', 'yes',  }
ONE_WORD_VALIDATION = { 'right', 'correct', 'sure', 'yeah', 'yup', 'yes', 'totally', 'absolutely'  }
I_VALIDATION = { 'agree', 'acknowledge', 'admit', 'recognize', 'concede', 'concur', 'affirm', 'jibe',
                 'accept', 'recognize', 'endorse', 'see' }


COMPLIMENT = { 'like', 'appreciate', 'enjoy', 'love', 'adore', 'approve', 'dig', }

FIRST_PERSON_HELP_MODALS = { 'might', 'would', }

pos_filename = "../../resources/liu-positive-words.txt"
neg_filename = "../../resources/liu-negative-words.txt"

RELIEF_PHRASES = [
    ' not your fault ', ' not ur fault ', ' no worries ', ' do n\'t worry ',
    ' dont worry ', ' its going to be ok', 'its okay to ', 
    ' it\'s going to be ok ',  ' it\'s going to be okay ', ' it\'s okay to ', 
    ]

PRESENCE_PHRASES = [
    ' here for you ', ' not ur fault ', ' no worries ', ' do n\'t worry ',
    ' it\'s going to be ok ',  ' it\'s going to be okay ', ' it\'s okay to ', 
    ]

TEACHING_PHRASES = [
    ' because you ', " the reason ", " a reason ", ' it means that ',
    ' it means you' # intentional space for you|your
    ]

COMPANIONS = { 'buddy', 'sons', 'mothers', 'neighbour', 'relationship', 'grandchild', 'grandchildren', 'family', 'momma', 'nephew', 'uncle', 'grandkid', 'companionship', 'ma', 'mummy',
               'stepfather', 'sis', 'uncles', 'neighbor', 'neighbors', 'stepkid', 'roommate', 'kids', 'mum', 'roomate', 'girlfriend', 'boyfriend', 'grandma', 'grandmother', 'families', 'bros',
               'baby', 'marriage', 'matess', 'friend', 'friends', 'boyfriend', 'boyfriends', 'relatives', 'team', 'cousin', 'partner', 'grandpa', 'stepchild',
               'grandson', 'sister', 'sister', 'husband', 'wife', 'dad', 'daddy', 'brother', 'brothers', 'companion', 'kid', 'daughter', 'daughters', 'coworker', 'coworkers',  'mother'
               'hubby', 'granddaughter', 'wives', 'wife', 'buddies', 'fiance', 'fiancee', 'parent', 'parents', 'companions', 'soulmate', 'mommy', 'father',
               'stepmother', 'son', 'niece', 'nephew', 'uncle', 'granny', 'children', }

POSITIVE_WORDS = set([x.strip() for x in open(pos_filename, encoding = "ISO-8859-1").read().splitlines()])
# get rid of some false positives
NEGATIVE_WORDS = set([x.strip() for x in open(neg_filename, encoding = "ISO-8859-1").read().splitlines()])

NEGATION_WORDS = { 'no', 'not', 'never', }

SYMPATHY_TERMS = { 'empathy', 'empathize', 'console', 'consolation', 'sympathy', 'sympathize', 'sympathies', 
                  'condole', 'condelence', 'condolences', }

UNDERSTANDING_VERBS = { 'understand', 'empathize', 'relate', 'know' }

I_VALIDATION = { 'agree', 'acknowledge', 'admit', 'recognize', 'concede', 'concur', 'affirm', 'jibe',
                 'accept', 'recognize', 'endorse', 'see' }

PRAYER_PHRASES = { ' pray for you ', ' pray for your ' }

AFFECTION_PHRASES = { ' give you a hug ', ' give you a big hug ', }

NETWORK_VERBS = { 'email', 'contact', 'call', 'reach' }

ENCOURAGEMENT_PHRASES = {

    ' give it a try ',
    ' go for it ',
    ' it \'s worth a shot ',
    ' what are you waiting for ',
    ' what do you have to lose ',
    ' just do it ',

    ' there you go ',
    ' keep up the good work ',
    ' keep it up ',
    ' good job ',
    ' proud of you ',

    ' hang in there ',
    ' don n\'t give up ',
    ' keep pushing ',
    ' keep fighting ',
    ' keep going ',

    ' have fun ',
    ' good luck ',
    
    ' stay strong ',
    ' never give up ',
    ' Never say die ',
    ' you can do it ',

    ' follow your dreams ',
    ' reach for the stars ',
    ' do the impossible ',
    ' believe in yourself ',
    ' the sky is the limit ',
    }

def debug(s, t):
    # print( s, t)
    pass

def get_support_indicators(spaced_text, tokens, dep_triples):

    if len(tokens) == 0:
        return {}


    lc_spaced_text = spaced_text.lower()
    # print lc_spaced_text
    
    features = Counter()

    uniq_terms = set([x.lower() for x in tokens if x.isalpha()])

    if len(tokens) > 1:
        uniq_terms_from_second = set([x.lower() for x in tokens[1:]])
    else:
        uniq_terms_from_second = set()

    ## Informational Support

    # Suggestion advice
    dep_to_arcs = defaultdict(list)
    
    verbs_to_rels = defaultdict(set)
    for d in dep_triples:
        # print(d)
        if d[0][1][0] == 'V':
            verbs_to_rels[d[0]].add(d[2])
        
        dep_to_arcs[d[0]].append((d[1], d[2]))
        dep_to_arcs[d[2]].append((d[1], d[0]))


        
    # Skip advice phrase as a question
    if not '?' in lc_spaced_text:
        #for verb, rels in verbs_to_rels.iteritems():
        for d, arcs in dep_to_arcs.items():
            if not d[1][0] == 'V':
                continue

            
            has_you = False
            has_modal = False
            has_fp = False
            has_fp_modal = False
            seems_like_question = False
            has_neg = False
            
            for a in arcs:

                if a[1][0].lower() == 'i':
                    has_fp = True
                if a[1][0].lower() == 'you':
                    has_you = True
                # Avoid the future tense, which will get tagged as modals
                if a[1][1] == 'MD' and a[1][0] != 'will':
                    has_modal = True
                    if a[1][0] in FIRST_PERSON_HELP_MODALS:
                        has_fp_modal = True
                if a[1][0].lower() in W_WORDS:
                    seems_like_question = True
                    break
                if a[0].lower() == 'neg':
                    has_neg = True
            
                    
            if not seems_like_question and has_you and has_modal:
                debug('ADVICE1?', spaced_text)
                features['support:advice'] = 1
            if not has_neg and not seems_like_question and has_fp and has_fp_modal:
                debug('ADVICE2?', spaced_text)
                features['support:advice'] = 1

            
        for d in dep_triples:
            if d[0][0] in SUGGESTION:
                debug('ADVICE3?', spaced_text)
                features['support:advice'] = 1
                break

        if ' if I were ' in lc_spaced_text:
            debug('ADVICE24?', spaced_text)
            features['support:advice'] = 1

            
        for ap in ADVICE_STARTS:
            if lc_spaced_text.startswith(ap):
                debug('ADVICE4?', spaced_text)
                features['support:advice'] = 1
                break
    # Referral
    if 'aDDRESS' in tokens or  ' <link> ' in lc_spaced_text:
        debug('REFERRAL?', spaced_text)
        features['support:referral'] = 1


    for rp in REFERRAL_PHRASES:
        if rp in lc_spaced_text:
            debug('REFERRAL?', spaced_text)
            features['support:referral'] = 1
            break
        
    # Situational appraisal
    for d in dep_triples:
        if d[0][0].lower() in SITUATION and d[2][0].lower() in { 'your', 'that' }:
            features['support:situational'] = 1
            debug('SITUATION1?', spaced_text)
            break

    for d in dep_triples:
        if is_word_set_match(d, SITUATION) and is_word_set_match(d, RELATED):
            features['support:situational'] = 1
            debug('SITUATION2?', spaced_text)
            break

    for d in dep_triples:
        if (d[0][0].lower() == 'fact' and d[2][0] == 'in') or \
           (d[0][0].lower() == 'point' and d[2][0] == 'the') or \
           (d[0][0].lower() == 'reality' and d[2][0] == 'in') or \
           (d[0][0].lower() == 'truth' and d[2][0] == 'in'):

            features['support:situational'] = 1
            debug('SITUATION3?', spaced_text)
            break
        
       
    # Teaching -- Can't be a question
    if '?' not in lc_spaced_text[-10:-1]:
        for tp in TEACHING_PHRASES:
            if tp in lc_spaced_text:
                features['support:teaching'] = 1
                debug('TEACHING?', spaced_text)
                break

        for d, arcs in dep_to_arcs.items():
            if d[0].lower() in EXPLANATION:
                has_neg = False
                has_your = False
                for a in arcs:
                    if a[0] == 'neg':
                        has_neg = True
                        break
                    if a[1][0].lower() == 'your':
                        has_your = True

                # NB: check the "your" which seems a bit too noisy
                if not has_neg and not has_your:
                    features['support:teaching'] = 1
                    debug('TEACHING2?', spaced_text)
                    break                
                

        # Exta bonus
        if "afaik" in uniq_terms:
            features['support:teaching'] = 1
            debug('TEACHING4?', spaced_text)
            # break                
                
    ## Tangible Assitance 
    # Loan (not seen with any systematic language in our analysis)

    # Direct task (offer to do something) 
    # -> Modal + I in Q
    if '?' in lc_spaced_text[-10:-1]:
        for d in dep_triples:
            tmp = get_alternate_if_is_word_match(d, "i")
            # Look for verbs
            if tmp is not None and tmp[1][0] == 'V':
                has_modal = False
                # print tmp, dep_to_arcs[tmp]
                for a in dep_to_arcs[tmp]:
                    # print a
                    if a[1][1] == 'MD':
                        has_modal = True
                        break
                if has_modal:
                    features['support:direct_ask'] = 1
                    debug('DIRECT_ASK1?', spaced_text)
                    break
    
    
    # Indirect task -- Tough to find (could be a TODO)
    
    # Active participation -- Tough to find (could be a TODO)
    

    # Willingness
    if not uniq_terms.isdisjoint(HELP) and 'i' in uniq_terms:
        features['support:willingness'] = 1
        debug('WILLINGNESS?', spaced_text)

    for d in dep_triples:
        if is_word_match(d, 'you') and is_word_match(d, 'welcome'):
            features['support:willingness'] = 1
            debug('WILLINGNESS2?', spaced_text)
            break
        
    ## Esteen Suppport

    # Compliment

    has_i_like = False
    like_verb = None
    has_negation = False

    for d in dep_triples:
        if is_word_set_match(d, COMPLIMENT):
            verb = d[0] if d[0][1][0] == 'V' else d[2]

            has_neg = False
            has_i = False
            dobj = None
            for a in dep_to_arcs[verb]:
                if a[0] == 'neg':
                    has_neg = True
                    break
                elif a[1][0] == 'I':
                    has_i = True
                elif a[0] == 'dobj':
                    dobj = a[1]

            if has_neg or not has_i or dobj is None:
                continue

            if dobj[0].lower() == 'you':
                features['support:compliment'] = 1
                debug('COMPLIMENT9?', spaced_text)
                break

            found = False
            for a in dep_to_arcs[dobj]:
                if a[1][0].lower() == 'your':
                    features['support:compliment'] = 1
                    debug('COMPLIMENT8?', spaced_text)
                    found = True
                    break
            if found:
                break


    m = re.search(' (?:thanks|thank you) ', lc_spaced_text)
    if m:
        if not ' nothing ' in lc_spaced_text[m.end():m.end()+15]:
            features['support:compliment'] = 1
            debug('COMPLIMENT11?', spaced_text)
            

    # Too Noisy
    if False:
        for verb, rels in verbs_to_rels.items():
            # print verb[0]
            if verb[0].lower() in COMPLIMENT:
                # debug('SAW VERB COMPLIMENT')
                for rel in rels:
                    if rel[0].lower() == 'i':
                    # debug('SAW I')
                        has_i_like = True
                        like_verb = verb[0].lower()
                    
        if has_i_like and ('you' in uniq_terms or 'your' in uniq_terms):
            debug('COMPLIMENT3?', spaced_text)
            features['support:compliment'] = 1
        
                    
    if False:
        if has_i_like:
            obj = None
            for d in dep_triples:
                tmp = get_alternate_if_is_word_match(d, like_verb)
                if tmp is not None:
                    obj = tmp

            if obj is not None:
                if obj[0].lower() == 'you':
                    features['support:compliment'] = 1
                    debug('COMPLIMENT2?', spaced_text)
                else:
                    for d in dep_triples:
                        if d[0] == obj and d[2][0].lower() == 'your':
                            features['support:compliment'] = 1
                            debug('COMPLIMENT3?', spaced_text)
                            break
                
                
    # Validation       
    for verb, rels in verbs_to_rels.items():
        
        has_you = False
        has_correct = False
        for rel in rels:
            if rel[0].lower() in { 'that',  'you' } :
               has_you = True
               # debug('HAS YOU' 
            elif rel[0].lower() in VALIDATION:
                # debug('HAS VALIDATION'
                has_correct = True
                    
        if has_you and has_correct:
            debug('VALIDATION1?', spaced_text)
            features['support:validation'] = 1
            
    correct_thing = None
    correct_phrase = None
    for d in dep_triples:

        if d[1] == 'det':
            continue
        
        # print d[0][0], d[2][0]
        if d[0][0].lower() in VALIDATION:
            correct_thing = d[2]
            correct_phrase = d[0]
        if d[2][0].lower() in VALIDATION:
            correct_thing = d[0]
            correct_phrase = d[2]            

        if is_word_match(d, 'that') and is_word_set_match(d, VALIDATION_VERBS) and \
           uniq_terms.isdisjoint(NEGATION_WORDS):
            debug('VALIDATION4?', spaced_text)
            features['support:validation'] = 1

    
        if is_word_set_match(d, VALIDATION_REFERENTS) and is_word_set_match(d, VALIDATION):
            arcs = dep_to_arcs[d]

            # print d

            has_neg = False
            for a in arcs:
                if a[0] == 'neg':
                    has_neg = True
                    break
                
            if not has_neg:
                debug('VALIDATION5?', spaced_text)
                features['support:validation'] = 1
            
            
    # Check for negation
    if correct_thing is not None:
        is_negated = False
        saw_your = False
        for d in dep_triples:
            if d[1] == 'neg' and (d[0] == correct_thing or d[2] == correct_thing \
                                  or d[0] == correct_phrase or d[2] == correct_phrase):
                is_negated = True
                break
            if (d[0] == correct_thing and d[2][0].lower() in { 'you', 'your'}) \
               or (d[2] == correct_thing and d[0][0].lower() in { 'you', 'your'}):
                saw_your = True
        if not is_negated and (saw_your or correct_thing[0].lower() == 'you'):
            debug('VALIDATION12?', spaced_text)
            features['support:validation'] = 1

    if " you 're right " in lc_spaced_text or " you 're correct " in lc_spaced_text \
       or lc_spaced_text == ' for sure ':
        debug('VALIDATION2?', spaced_text)
        features['support:validation'] = 1
                
    # -> "totally!"
    if len(uniq_terms) == 1 and next(iter(uniq_terms)) in ONE_WORD_VALIDATION:
        debug('VALIDATION6?', spaced_text)
        features['support:validation'] = 1

    if tokens[0].lower() in INITIAL_VALIDATION:
        debug('VALIDATION23?', spaced_text)
        features['support:validation'] = 1

    # <positive> to me
    for d in dep_triples:
        if d[1] == 'nmod' and is_word_match(d, 'me') and is_word_set_match(d, POSITIVE_WORDS):
            debug('VALIDATION7?', spaced_text)
            features['support:validation'] = 1
            break


    # <positive> to me
    for d in dep_triples:
        if is_word_set_match(d, SUGGESTION_NOUN) and is_word_set_match(d, POSITIVE_WORDS):
            # print d
            # Check for negation and speculation
            has_neg = False
            for a in dep_to_arcs[d[0]]:
                if a[0] == 'neg' or a[1][0] == 'if':
                    has_neg = True
            if not has_neg:
                for a in dep_to_arcs[d[2]]:
                    if a[0] == 'neg' or a[1][0] == 'if':
                        has_neg = True
                
            if not has_neg:
                debug('VALIDATION11?', spaced_text)
                features['support:validation'] = 1
                break

        
    for d in dep_triples:
        # TODO: See if we can refine the deprel a bit more based on positive words
        if d[1] == 'dobj':
            continue
        alt = get_alternate_if_is_word_set_match(d, POSITIVE_WORDS)
        if alt is not None:

            # Check to avoid validation statements
            m = get_word_set_match(d, POSITIVE_WORDS)

            if alt[0].lower() == 'you' and not m[0] in VALIDATION:
                debug('COMPLIMENT6?', spaced_text)
                features['support:compliment'] = 1
            else:
                for a in dep_to_arcs[alt]:
                    if a[1][0].lower() == 'your':
                        debug('VALIDATION9?', spaced_text)
                        features['support:validation'] = 1
                        break
        # TODO: improve this filter to support noun compounds

    for d in dep_triples:
        if is_word_match(d, "i"):
            m = get_word_set_match(d, I_VALIDATION)
            if m is None:
                continue
            # It's something like "I agree" so check for negation
            is_neg = False
            for a in dep_to_arcs[m]:
                if a[0] == 'neg':
                    is_neg = True
                    break
            if not is_neg:
                debug('VALIDATION22?', spaced_text)
                features['support:validation'] = 1
                break
            

    if False:
        has_i_validation = False
        is_neg = False
        for verb, rels in verbs_to_rels.items():
            if verb[0].lower() in I_VALIDATION:
                debug('HAS I VALIDATION')
                for rel in rels:
                    if rel[0].lower() == 'i':
                        has_i_validation = True
                    if rel[1] == 'neg':
                        is_neg = True
                        break
        if has_i_validation and not is_neg: # and (('you' in uniq_terms or 'your' in uniq_terms) \
           #    or len(uniq_terms) == 2):
            debug('VALIDATION24?', spaced_text)
            features['support:validation'] = 1

    # Relief of blame
    for rp in RELIEF_PHRASES:
        if rp in lc_spaced_text:
            debug('BLAME_RELIEF?', spaced_text)
            features['support:blame_relief'] = 1

    ## Network support 
    
    # Presence (of replier) -- TBD
    
    # Companions (reminds of people)
    if not COMPANIONS.isdisjoint(uniq_terms):
        for d in dep_triples:
            if (d[0][0].lower() in COMPANIONS and d[2][0].lower() == 'your') \
               or (d[2][0].lower() in COMPANIONS and d[0][0].lower() == 'your'):
                debug('COMPANIONS?', spaced_text)
                features['support:companions'] = 1

    ## Emotional support

    # Relationship (stresses importance of closeness and love) -- TBD
    
    
    # Physical affection
    for ap in AFFECTION_PHRASES:
        if ap in lc_spaced_text:
            debug('AFFECTION?', spaced_text)
            features['support:affection'] = 1

    
    # Confidentiality 

    # Sympathy (Sorrow or Regret)
    sympathy_terms = SYMPATHY_TERMS.intersection(uniq_terms)
    if len(sympathy_terms) > 0:
        # Check for negation
        has_neg = False
        for dep, arcs in dep_to_arcs.items():
            if dep[0][0].lower() in sympathy_terms:
                for a in arcs:
                    if a[0] == 'neg':
                        has_neg = True
                        break
            if has_neg:
                break
        if not has_neg:
            debug('SYMPATHY1?', spaced_text)
            features['support:sympathy'] = 1
    

    # Listening (attentiveness)
    # -> follow up questions

    # Listening can either be a question or an unpunctuated statement but can't
    # end with non-question punctuation
    if tokens[0].lower() in INTERROGATIVES and tokens[-1] != '.' and tokens[-1] !='!': # and '?' in lc_spaced_text:
        debug('LISTENING1?', spaced_text)
        features['support:listening'] = 1
        


    # Understanding/empathy (expresses understanding or discloses personal situation)
    # -> I understand [ you or your ] ; I know ; I feel
    for d in dep_triples:
        #print d

        if is_word_set_match(d, UNDERSTANDING_VERBS) and is_word_match(d, 'i'):
            
            verb = d[0] if d[0][1][0] == 'V' else d[2]
            # debug('->thing->', verb)

            has_neg = False
            ccomp_d = None
            dobj = None
            for a in dep_to_arcs[verb]:
                if a[0] == 'neg':
                    has_neg = True
                    break
                elif a[0] == 'ccomp' or a[0] == 'advcl':
                    ccomp_d = a[1]


            #debug('->ccomp_d->', ccomp_d)
                    
            if has_neg or ccomp_d is None:
                continue
            
            # See if we can find something about what they're experiencing
            found_you = False
            for a in dep_to_arcs[ccomp_d]:
                if a[1][0].lower() == 'you':
                    found_you = True
                    break
            if found_you:
                features['support:understanding'] = 1
                debug('UNDERSTANDING1?', spaced_text)
                break

            # Check for your
            for a in dep_to_arcs[ccomp_d]:
                if not (a[0] == 'nsubj' or a[0] == 'dobj'):
                    continue
                # See if the subject or object of the complemented verb has a
                # possessive modifier
                for a2 in dep_to_arcs[a[1]]:
                    if a[1][0] == 'your':
                        found_you = True
                        break
                if found_you:
                    break
                    
            if found_you:
                features['support:understanding'] = 1
                debug('UNDERSTANDING2?', spaced_text)
                break

    # Encouragement
    for pp in ENCOURAGEMENT_PHRASES:
        if pp in lc_spaced_text:
            debug('ENCOURAGEMENT?', spaced_text)
            features['support:encouragement'] = 1
        
    # Prayer
    for pp in PRAYER_PHRASES:
        if pp in lc_spaced_text:
            debug('PRAYER?', spaced_text)
            features['support:prayer'] = 1
            
    return features        

def is_word_match(dep_triple, lemma):
    return dep_triple[0][0].lower() == lemma \
        or dep_triple[2][0].lower() == lemma

def is_word_set_match(dep_triple, words):
    return dep_triple[0][0].lower() in words \
        or dep_triple[2][0].lower() in words

def get_alternate_if_is_word_match(dep_triple, lemma):
    if dep_triple[0][0].lower() == lemma:
        return dep_triple[2]
    elif dep_triple[2][0].lower() == lemma:
        return dep_triple[0]
    else:
        return None

def get_alternate_if_is_word_set_match(dep_triple, words):
    if dep_triple[0][0].lower() in words:
        return dep_triple[2]
    elif dep_triple[2][0].lower() in words:
        return dep_triple[0]
    else:
        return None

def get_word_set_match(dep_triple, words):
    if dep_triple[0][0].lower() in words:
        return dep_triple[0]
    elif dep_triple[2][0].lower() in words:
        return dep_triple[2]
    else:
        return None
    
       
    
if __name__ == '__main__':

    import sys
    from social_features import to_parsed_representations
    from random import shuffle
    
    input_file = sys.argv[1]    

    with open(input_file) as f:
        lines = [line.split('\t')[5] for line in f.readlines()]
    # Strip header
    lines = lines[1:]
        
    for line_no, line in enumerate(lines):
            
        reply = line

        # Some text normalization:
        reply = reply.strip()
        reply = reply[0].upper() + reply[1:]           
        reply = re.sub(r'\bi\b', 'I', reply)
        reply = re.sub(r'\b([yY])oure\b', r"\1ou're", reply)
        reply = re.sub(r'\b([yY])ouve\b', r"\1ou've", reply)
        reply = re.sub(r'\b([wWdD])ont\b', r"\1on't", reply)
        reply = re.sub(r'\b([tTwW])hats\b', r"\1hat's", reply)
        reply = re.sub(r'\b[iI]m\b', "I'm", reply)
        reply = re.sub(r'\bu\b', 'you', reply)

        sentences = en_nlp(reply).sents
        tokens, pos_tags, dep_triples = to_parsed_representations(next(sentences))        
        spaced_text = ' ' + ' '.join(tokens) + ' '
        
        feats = get_support_indicators(spaced_text, tokens, dep_triples)
        print('')


