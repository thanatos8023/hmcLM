# -*- coding:utf-8 -*-

import dill
import os
from corpus2raw import pos2word
from eunjeon import Mecab
import nltk
from nltk import trigrams, bigrams
from collections import Counter, defaultdict


def tri(tokens, model):
    pass


def bi(tokens, model):
    pass


def decode(sentence, model):
    # sentence: Input sentence
    # model: model path

    # We need mecab
    mecab = Mecab()

    # Convert sentence to token
    word_token = pos2word(mecab.pos(sentence))

    # Load model
    with open(model, 'rb') as f:
        models = dill.load(f)
        print("model loaded")


if __name__ == '__main__':
    decode('내차 시동 켜볼래', 'model/hmc.model')