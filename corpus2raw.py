# -*- coding:utf-8 -*-

from eunjeon import Mecab
from nltk import bigrams, trigrams
from collections import Counter, defaultdict


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
    with open('corpus/Control_Engine_Start_Temp.txt', 'r', encoding='utf-8') as f:
        raws = f.read().split('\n')

    first_sentence = pos2word(mecab.pos(raws[0]))
    print('Mecab Test: ', first_sentence)

    # Get the bigrams
    print('bigrams: ', list(bigrams(first_sentence)))

    # Get the padded bigrams
    print('bigrams (padded): ', list(bigrams(first_sentence, pad_left=True, pad_right=True)))

    # Get the trigrams
    print('trigrams: ', list(trigrams(first_sentence)))

    # Get the padded trigrams
    print('trigrams (padded): ', list(trigrams(first_sentence, pad_left=True, pad_right=True)))

    model = defaultdict(lambda: defaultdict(lambda: 0))

    for raw in raws:
        sentence = pos2word(mecab.pos(raw))
        for w1, w2, w3 in trigrams(sentence, pad_left=True, pad_right=True):
            model[(w1, w2)][w3] += 1

    test_sentence = "내 차 공조 23도로 시동 켜줄래"
    test_tokens = pos2word(mecab.pos(test_sentence))
    print('test sentence: ', test_tokens)

    print(model[(test_tokens[0], test_tokens[1])][test_tokens[2]])
    print(model[(test_tokens[0], test_tokens[1])][test_tokens[4]])
    print(model[(None, None)][test_tokens[0]])

    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count

    print(model[(test_tokens[0], test_tokens[1])][test_tokens[2]])
    print(model[(test_tokens[0], test_tokens[1])][test_tokens[4]])
    print(model[(None, None)][test_tokens[0]])


if __name__ == '__main__':
    main()