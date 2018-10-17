# -*- coding:utf-8 -*-

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer


def decode(sentence, model_path):
    # sentence: Input sentence
    # model: model path

    # sentence transform
    hash_vect = HashingVectorizer(n_features=(697))
    count_vect = CountVectorizer()
    sent_counts = hash_vect.fit_transform([sentence])
    #sent_counts = count_vect.fit_transform([sentence])
    print(sent_counts.shape)

    tfidf_transformer = TfidfTransformer()
    sent_tfidf = tfidf_transformer.fit_transform(sent_counts)

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        print("model loaded")

    # get result
    return model.predict(sent_tfidf)


if __name__ == '__main__':
    intention = decode('내차 시동 켜볼래', 'model/hmc.model')
    print(intention)
