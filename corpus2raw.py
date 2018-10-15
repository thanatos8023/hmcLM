# -*- coding:utf-8 -*-

from eunjeon import Mecab
import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def pos2word(pos_list):
    #########################################################
    # Mecab에서 분석된  문장의 형식
    # [(morph1, tag1), (morph2, tag2), (morph3, tag3), ...]
    #
    # 이 형식을
    # ['morph1/tag1', 'morph2/tag2', 'morph3/tag3', ...]
    # 로 변환하는 함수
    #########################################################
    result = list()
    for pos_tuple in pos_list:
        # pos_tuple[0]: morph
        # pos_tuple[1]: tag
        result.append('{}/{}'.format(pos_tuple[0], pos_tuple[1]))
    return result


def main():
    # Mecab 불러오기
    mecab = Mecab()

    # Corpus 불러오기: 
    word_token = pos2word(mecab.pos(u'자연주의 쇼핑몰은 어떤 곳인가?'))

    #print(word_token)



if __name__ == '__main__':
    main()