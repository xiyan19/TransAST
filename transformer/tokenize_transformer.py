# -*- coding: utf-8 -*-

"""
    This script reads the corpus and constructs the lexicon.
    —————————————————
    usage: python3 tokenize_transformer.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/05/28
    $Annotation: Create.
    $Author: xiyan19
"""


import gensim
import nltk


if __name__ == "__main__":
    corpus_file = '.............../raw_corpus'
    save_model = '/................../ast1.dict'

    with open(corpus_file, 'r') as f: # 1
        dataset = []
        count = 0

        for text in f.readlines():
            print(count)
            tokens = nltk.word_tokenize(str(text))
            dataset.append(tokens)
            count += 1

        dictionary = gensim.corpora.Dictionary(dataset)
        dictionary.save(save_model)
