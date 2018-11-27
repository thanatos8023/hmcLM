# -*- coding:utf-8 -*-

import os
import numpy as np
import random
import sys
import argparse
import pickle
import nltk
from nltk import bigrams, trigrams
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from eunjeon import Mecab
from sklearn import metrics
#import MeCab


def make_vocab(corpus_path, save_path):
    mecab = Mecab()
    fl = glob.glob(corpus_path+"/*.txt")

    all_morphs = []
    all_tri = []
    all_bi = []
    for fn in fl:
        with open(fn, 'r', encoding='utf-8') as f:
            raw = f.readlines()
        for s in raw:
            morphs = mecab.morphs(s)
            for m in morphs:
                all_morphs.append(m)

            for w1, w2, w3 in trigrams(s, pad_left=True, pad_right=True):
                all_tri.append((w1, w2, w3))

            for w1, w2 in bigrams(s, pad_left=True, pad_right=True):
                all_bi.append((w1, w2))

    #all_morphs = nltk.FreqDist(w for w in all_morphs)
    #uni = list(all_morphs)[:200]
    #all_tri = nltk.FreqDist(tri for tri in all_tri)
    #tri = list(all_tri)[:200]
    #all_bi = nltk.FreqDist(bi for bi in all_bi)
    #bi = list(all_bi)[:200]

    #vocab = uni + tri + bi

    vocab = list(set(all_morphs))
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("--- Vacabulary saved in", save_path)
    print('%% The size of vocabulary:', len(vocab))


def make_data(path, testprob):
    print('--- Making data')

    # Get corpus file list
    corpuslist_abs = os.listdir(path)

    template = []

    X = []
    y = []

    # make model corpus by corpus
    for cabs in corpuslist_abs:
        # make corpusname
        filename = os.path.basename(cabs)
        corpusname = os.path.splitext(filename)[0]

        # Get corpus
        ########## corpus frame ############
        # sentence1
        # sentence2
        # ...
        ####################################
        with open(path + '/' + cabs, 'r', encoding='utf-8') as f:
            raw = f.readlines()

        for sent in raw:
            template.append((sent, corpusname))

    random.shuffle(template)

    for sent in template:
        X.append(sent[0])
        y.append(sent[1])

    #for i in range(10):
    #    print('{}\t{}'.format(X[i], y[i]))

    idx = int(len(X) - (len(X)*testprob))
    train_X, train_y, test_X, test_y = X[:idx], y[:idx], X[idx:], y[idx:]

    print("--- Making data Done")
    print('--- Data information')
    print('%% The number of sentences of train:', len(train_X))
    print('%% The number of intentions:', len(list(set(train_y))))

    return train_X, train_y, test_X, test_y


def train():

    # tokenizer
    mecab = Mecab()

    train_X, train_y, test_X, test_y = make_data('corpus', testprob=0.1)
    #print(len(train_X), len(train_y))

    print('--- Get vocabulary')
    with open('vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    print('--- Load vocabulary successfully')
    print('%% Vacabulary size:', len(vocab))

    count_vect = CountVectorizer(
        tokenizer=mecab.morphs,
        ngram_range=(1, 3),
        max_features=10000,
        vocabulary=vocab
    )

    X_train_counts = count_vect.transform(train_X)
    print("The number of features: {}".format(X_train_counts.shape[1]))

    tfidf_transformer = TfidfTransformer(
        use_idf=False,
        smooth_idf=False,
        norm='l2'
    )
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    #print(X_train_tfidf.shape)

    # Naive Beyesian
    # clf = MultinomialNB().fit(X_train_tfidf, train_y)

    # SVM
    clf_svm = SGDClassifier().fit(X_train_tfidf, train_y)

    # Evaluation
    X_test_counts = count_vect.transform(test_X)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print()
    predicted = clf_svm.predict(X_test_tfidf)
    print("SVM: ", np.mean(predicted == test_y))

    print()
    print("Examples: ")
    print("Input\t   Predicted\t  Correct")
    for i in range(3):
        print("%s\t=> %s\t: %s" % (test_X[i], predicted[i], test_y[i]))

    # model save
    # first, delete old model
    os.remove('model/hmc.model')
    print()
    with open('model/hmc.model', 'wb') as f:
        pickle.dump(clf_svm, f)

    #pickle.dump(count_vect, open('model/count.pickle', 'wb'))
    #pickle.dump(X_train_tfidf, open('model/train_feature.pickle', 'wb'))
    #pickle.dump(X_test_tfidf, open('model/test_feature.pickle', 'wb'))
    print('SVM classifier model saved at "model/hmc.model"')
    print('If you want to load the model, use "pickle.load" in python.')


def decode_in_server(sentence):
    # define tokenizer
    def tokenizer(sent):
        mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
        result = []

        tags = mecab.parseToNode(sent)
        while tags:
            #output = '%s/%s' % (tags.surface, tags.feature.split(',')[0])
            result.append(tags.surface)
            tags = tags.next

        return result

    print('--- Get vocabulary')
    try:
        with open('vocab.pickle', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print('Loading vocabulary ERROR. There is no vocabulary.')
        return None

    print('--- Load vocavulary successfully')
    print('%% Vocabulary size:', len(vocab))

    try:
        with open('model/hmc.model', 'rb') as f:
            model = pickle.load(f)
            print("--- Loading model Successfully")
    except FileNotFoundError:
        print("Loading model Failed. There is no model.")
        return None

    count_vect = CountVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 3),
        max_features=10000,
        vocabulary=vocab)

    tfidf_vect = TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 3),
        max_features=10000,
        vocabulary=vocab
    )

    # vectorize
    sent_counts = count_vect.transform([sentence])
    # print(sent_counts.shape)

    tfidf_transformer = TfidfTransformer(
        use_idf=False,
        smooth_idf=False,
        norm='l2'
    )
    sent_tfidf = tfidf_transformer.transform(sent_counts)

    pred = model.predict(sent_tfidf)

    print('Input:', sentence)
    print('Prediction:', pred)

    return pred


def decode(sentence):
    # tokenizer
    mecab = Mecab()

    print('--- Get vocabulary')
    try:
        with open('vocab.pickle', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print('Loading vocabulary ERROR. There is no vocabulary.')
        return None

    print('--- Load vocabulary successfully')
    print('%% Vocabulary size:', len(vocab))

    try:
        with open('model/hmc.model', 'rb') as f:
            model = pickle.load(f)
            print("--- Loading model Successfully")
    except FileNotFoundError:
        print("Loading model Failed. There is no model.")
        return None

    #count_vect = pickle.load(open('model/count.pickle', 'rb'))

    count_vect = CountVectorizer(
        tokenizer=mecab.morphs,
        ngram_range=(1, 3),
        max_features=10000,
        vocabulary=vocab)

    tfidf_vect = TfidfVectorizer(
        tokenizer=mecab.morphs,
        ngram_range=(1, 3),
        max_features=10000,
        vocabulary=vocab
    )

    # vectorize
    sent_counts = count_vect.transform([sentence])
    #print(sent_counts.shape)

    tfidf_transformer = TfidfTransformer(
        use_idf=False,
        smooth_idf=False,
        norm='l2'
    )
    sent_tfidf = tfidf_transformer.transform(sent_counts)

    pred = model.predict(sent_tfidf)

    print('Input:', sentence)
    print('Prediction:', pred)

    return pred


if __name__ == '__main__':
    #make_vocab('corpus', 'vocab.pickle')
    #train()

    # on local
    decode("시동켜주세요")

    # on server
    #decode_in_server("시동켜주세요")