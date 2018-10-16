# -*- coding:utf-8 -*-

import os
import pickle
from corpus2raw import pos2word
import glob
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from eunjeon import Mecab


def train(corpus, *args):
    # corpus: directory path of corpus
    # *args
    # ngrams = 3
    # katz = true

    # Get corpus file list
    corpuslist_abs = os.listdir(corpus)

    # We need Morpheme analyzer
    # We will use mecab
    mecab = Mecab()

    # Models list
    models = defaultdict(lambda: defaultdict(lambda: defaultdict))

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
        with open(cabs, 'r', encoding='utf-8') as f:
            corpus = f.readlines()

        # tokenize with tagged
        word_tokens = list()
        for sentence in corpus:
            word_tokens.append(pos2word(mecab.pos(sentence)))

        # make model
        # Laplace smoothing
        model_tri = defaultdict(lambda: defaultdict(lambda: 0))
        for tokens in word_tokens:
            for w1, w2, w3 in trigrams(tokens, pad_left=True, pad_right=True):
                model_tri[(w1, w2)][w3] += 1
        for w1_w2 in model_tri:
            total_count = float(sum(model_tri[w1_w2].values()))
            for w3 in model_tri[w1_w2]:
                model_tri[w1_w2][w3] /= total_count

        model_bi = defaultdict(lambda: defaultdict(lambda: 0))
        for tokens in word_tokens:
            for w1, w2 in bigrams(tokens, pad_left=True, pad_right=True):
                model_bi[w1][w2] += 1
        for w1 in model_bi:
            total_count = float(sum(model_bi[w1].values()))
            for w2 in model_bi[w1]:
                model_bi[w1][w2] /= total_count

        models[corpusname] = [model_tri, model_bi]

    # save model
    with open('model/hmc.model', 'wb') as f:
        pickle.dump(models, f)
        print('Successfully Model saved!')


if __name__ == '__main__':
    train('corpus')
