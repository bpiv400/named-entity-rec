from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from nltk.stem import SnowballStemmer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD

import numpy as np
import unidecode
import get_lemmas
import pickle

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

## Added indicator for whether all letters are capital (offset-capital)
## Added index in sentence (sent-pos) 
## EXPERIMENT RESULTS: 
# Perceptron: 
## o-sentence-index, o-all-caps, o-word : 17.14
## o-all-caps, o-word: 51.41
## o-all-caps, o-word, o-sentence-start: 47.45
# All Features (2 history entity tags; lemma, word, and pos in 2 word symmetric
# window; whether the current word is all caps)
#   Default Tolerance: 73.10
# All Features except History: 64.39
## ADA BOOST: (Intuition: Each feature is weakly useful, combining them iteratively could be amazing)
# All Features (2 history entity tags; lemma, word, and pos in 2 word symmetric
# window; whether the current word is all caps)
# has first letter capped, prefix, and suffix)
# 600 Trees of depth 2: 21.48
# 100 trees depth of 1: 23.41
# WHY ARE THESE FAILING

## double pass -- 2 perceptrons
## All features with 2 predicted history chunks in the final model: 42.85

def write_results(name, preds): 
    # format is: word gold pred
    j = 0
    with open("results" + name + ".txt", "w") as out:
        for sent in curr_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = preds[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

def getfeats(word, pos, ent, o, pred, i):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o_str = str(o)
    isUpper, first_cap, is_first = False, False, False
    prefix, suffix = '', ''
    if (o == 0):
        isUpper = (word.upper() == word)
        first_cap = word[0].isupper()
        is_first = (i == 0)
        stem = stemmer.stem(word)
        comp_word = unidecode.unidecode(word.lower())
        comp_stem = unidecode.unidecode(stem.lower())
        while comp_stem not in comp_word:
            comp_stem_opt1 = comp_stem[:len(comp_stem) - 1]
            comp_stem_opt2 = comp_stem[1:]
            comp_stem_opt3 = comp_stem[1:len(comp_stem)-1]
            if comp_stem_opt1 in comp_word:
                comp_stem = comp_stem_opt1
            elif comp_stem_opt2 in comp_word:
                comp_stem = comp_stem_opt2
            else:
                comp_stem = comp_stem_opt3
        affixes = comp_word.split(comp_stem, 1)
        prefix = word[:len(affixes[0])]
        suffix = word[(len(word)-len(affixes[1])):]
    
    # Determine lemmas
    lemma = ''
    if word in word2lemma:
        lemma = word2lemma[word.lower()]
    else:
        lemma = word
    features = []
    features.append((o_str+'word', word))
    features.append((o_str+'lemma', lemma))
    features.append((o_str+'pos', pos))

    ## Maybe include depending on performance

    if o == 0:
        features.append(('all-cap', isUpper))
        features.append(('first-cap', first_cap))
        features.append(('prefix', prefix))
        features.append(('suffix', suffix))
        # features.append(('index', i))
        features.append(('sen-start', is_first))
    if pred is not None:
        if o == -1 or o == -2:
            if is_training:
                features.append((o_str+'pred', ent))
            else: 
                features.append((o_str+'pred', ent))
    return features
    

def word2features(sent, i, prev_preds = None):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-3, -2, -1, 0, 1, 2, 3]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            pos = sent[i+o][1]
            ent = sent[i+o][2]
            pred = None
            if prev_preds is not None:
                pred = prev_preds[i+o]
            featlist = getfeats(word, pos, ent, o, pred, i)
            features.extend(featlist)
    
    return dict(features)

if __name__ == "__main__":
    # Load the training data
    path_to_lemmas = './lemmatization-es.txt'
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    # Construct lemmas
    word2lemma = get_lemmas.extract_lemmas(path_to_lemmas)

    # get stemmer
    stemmer = SnowballStemmer("spanish") 

    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i, None)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer1 = DictVectorizer()
    vectorizer2 = DictVectorizer()

    X_train = vectorizer1.fit_transform(train_feats)

    # TODO: play with other models
    # model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    # n_estimators=100,
    # learning_rate=1)

    print("Starting TruncatedSVD") 
    tsvd = TruncatedSVD(n_components = 2000)
    tsvd.fit(X_train)
    print("TruncatedSVD complete")
    mlp1a3 = MLPClassifier(alpha = 1, hidden_layer_sizes = (25))
    m2p1a3 = MLPClassifier(alpha = 1, hidden_layer_sizes = (50))
    m3p1a3 = MLPClassifier(alpha = 1, hidden_layer_sizes = (100))
    m4p1a3 = MLPClassifier(alpha = 1, hidden_layer_sizes = (200))

    X_train = tsvd.transform(X_train)
    print("Training New Model")
    mlp1a3.fit(X_train, train_labels)  
    print("Training New Model")
    mlp2a3.fit(X_train, train_labels)
    print("Training New Model")
    mlp3a3.fit(X_train, train_labels)
    print("Training New Model")
    mlp4a3.fit(X_train, train_labels)
    
    # train_pred = model1.predict(X_train)
    
    train_feats = []
    train_labels = []

    # Preparing Second Pass Model
    #instance_counter = 0
    #for sent in train_sents:
    #    curr_preds = [0]*len(sent)
    #    for j in range(len(sent)):
    #        curr_preds[j] = train_pred[instance_counter]
    #        instance_counter += 1
    #    for i in range(len(sent)):
    #        feats = word2features(sent, i, curr_preds)
    #        train_feats.append(feats)
    #        train_labels.append(sent[i][-1])

    #X_train = vectorizer2.fit_transform(train_feats)

    # print("Training Second Pass Model")
    # print("Second model training x_size: " +str(X_train.shape))
    # model2.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    #First pass over test_sents
    curr_sents = test_sents

    # switch to test_sents for your final results
    for sent in curr_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i, None)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer1.transform(test_feats)
    X_test = tsvd.transform(X_test)

    mlp1a3_pred = mlp1a3.predict(X_test)  
    mlp2a3_pred = mlp2a3.predict(X_test)
    mlp3a3_pred = mlp3a3.predict(X_test)
    mlp4a3_pred = mlp4a3.predict(X_test)
    
    #instance_counter = 0
    #for sent in curr_sents:
    #    curr_preds = [0]*len(sent)
    #    for j in range(len(sent)):
    #        curr_preds[j] = y_pred_prime[instance_counter]
    #        instance_counter += 1
    #    for i in range(len(sent)):
    #        feats = word2features(sent, i, curr_preds)
    #        test_feats.append(feats)
    #        test_labels.append(sent[i][-1])
    #
    #X_test = vectorizer2.transform(test_feats)
    #
    #print("Second model test x_size: " +str(X_test.shape))
    #y_pred = model2.predict(X_test)
    print("Writing Results")

    write_results("mlp1a3_short", mlp1a3_pred)
    write_results("mlp2a3_short", mlp2a3_pred)
    write_results("mlp3a3_short", mlp3a3_pred)
    write_results("mlp4a3_short", mlp4a3_pred)
    
    print("Now run: python conlleval.py results.txt")






