# -*- coding:utf-8 -*-

import os
import numpy as np
import random
import sys
import argparse
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


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
        # This corpusname will be reference of model in defaultdict
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

    idx = int(len(X) - (len(X)*testprob))
    train_X, train_y, test_X, test_y = X[:idx], y[:idx], X[idx:], y[idx:]

    return train_X, train_y, test_X, test_y


def train():
    train_X, train_y, test_X, test_y = make_data('corpus', testprob=0.1)
    print(len(train_X), len(train_y))

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_X)
    print("The number of features: {}".format(X_train_counts.shape[1]))

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)

    # Naive Beyesian
    clf = MultinomialNB().fit(X_train_tfidf, train_y)

    # SVM
    clf_svm = SGDClassifier().fit(X_train_tfidf, train_y)

    # Evaluation
    X_test_counts = count_vect.transform(test_X)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    print("Naive Bayesian: ", np.mean(predicted == test_y))

    predicted = clf_svm.predict(X_test_tfidf)
    print("SVM: ", np.mean(predicted == test_y))

    # model save
    with open('model/hmc.model', 'wb') as f:
        pickle.dump(clf_svm, f)
    print('SVM classifier model saved at "model/hmc.model"')
    print('If you want to load the model, use "pickle.load" in python.')

if __name__ == '__main__':
    train();