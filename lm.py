import nltk
from nltk import bigrams, trigrams
import os
from collections import Counter, defaultdict
from eunjeon import Mecab
import random


mecab = Mecab()

# get whole data for making vocaburary(feature)
all_morphs = []
all_tri = []
all_bi = []

database = []
file_list = os.listdir('corpus')
for filename in file_list:
    with open('corpus/'+filename, 'r', encoding='utf-8') as f:
        raw = f.readlines()

    corpusname = filename.replace('.txt', '')
    for line in raw:
        database.append([(line, corpusname)])

        morphs = mecab.morphs(line)
        for m in morphs:
            all_morphs.append(m)

        for w1, w2, w3 in trigrams(morphs, pad_left=True, pad_right=True):
            all_tri.append((w1, w2, w3))

        for w1, w2 in bigrams(morphs, pad_left=True, pad_right=True):
            all_bi.append((w1, w2))

random.shuffle(database)

idx = int(len(database)*0.9)

all_morphs = nltk.FreqDist(w for w in all_morphs)
uni = list(all_morphs)[:500]
all_tri = nltk.FreqDist(tri for tri in all_tri)
tri = list(all_tri)[:500]
all_bi = nltk.FreqDist(bi for bi in all_bi)
bi = list(all_bi)[:500]

