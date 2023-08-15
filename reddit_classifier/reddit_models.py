import pickle
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime
from datetime import timedelta
import sys
import csv
import nltk
nltk.download('wordnet')
import LIWCMeta
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
# dictionary for showing which model performs the best
subreddit_model_dict = {'psychosis':'linear_svm', 'stress':'linear_svm', 'selfharm':'linear_svm',
 'schizophrenia':'linear_svm', 'SuicideWatch':'linear_svm', 'depression':'linear_svm', 'anxiety':'linear_svm'}

# Load all the models
mh_model_list = [joblib.load('Models/'+x+'_liwc_'+y+'.joblib') for x,y in subreddit_model_dict.items()]

mh_model_dict = {x:joblib.load('Models/'+x+'_liwc_'+y+'.joblib') for x,y in subreddit_model_dict.items()}


# Dictionary for the indices of the LIWC categories
liwc_dict = {'present_tense':1,'exclusive':2,'family':3,'inclusive':4,'feel':5,
'money':6,'causation':7,'insight':8,'humans':9,'relative':10,'preposition':11,
'see':12,'adverbs':13,'article':14,'anger':15,'home':16,'sexual':17,
'future_tense':18,'death':19,'third_person':20,'negation':21,'discrepancies':22,
'religion':23,'percept':24,'verbs':25,'health':26,'past_tense':27,
'first_person_plural':28,'bio':29,'tentativeness':30,'first_person_singular':31,
'body':32,'inhibition':33,'hear':34,'cognitive_mech':35,'second_person':36,
'quantifier':37,'conjunction':38,'friends':39,'achievement':40,
'negative_affect':41,'auxiliary_verbs':42,'anxiety':43,'certainty':44,'work':45,
'indefinite_pronoun':46,'sadness':47,'swear':48,'positive_affect':49,'social':50}
liwc_category_count = 50
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



def get_liwc_embedding(post):
    """Returns a liwc vector for the given mood
    Input: post - a single data unit
    Output: A 50-dim vector that contains the normalized count of the LIWC
    """
    # initialize the vector
    post = ' '.join([lemmatizer.lemmatize(w) for w in post.split()])
    result = []
    for i in range(0, liwc_category_count):
        result.append(0)
    liwc_lexicon = LIWCMeta.extract_liwc_features('all')
    post_liwc = LIWCMeta.getLex(post, liwc_lexicon)
    for key, value in post_liwc.items():
        idx = liwc_dict[key]-1
        result[idx] = value/len(post.split(' '))
    return result


def logistic(x):
    # generic logistic regression for understanding probabilities of topics
    ex = np.exp(-x)
    return 1/(1+ex)

def get_proba(posts):
    # score=0: not the subreddit, score=0: the subreddit
    for post in posts:
        # dictionary for each subreddit's probabilities; need to account for probabilties associated with score=0
        
        for subreddit, model in mh_model_dict.items():
            score = model.decision_function([get_liwc_embedding(text)])
            print(f'{subreddit}: Score: {logistic(score)}')
