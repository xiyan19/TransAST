# -*- coding: utf-8 -*-

"""
    This script converts the corpus into character format.
    —————————————————
    usage: python3 trans_corpus.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/06
    $Annotation: Create.
    $Author: xiyan19
"""


import gensim

if __name__ == "__main__":
    corpus_file = '................./raw_corpus'
    trans_corpus_file = '......................./char_corpus'

    dict = gensim.corpora.dictionary.Dictionary.load('....................../ast.dict')

    trans_list = []

    with open(corpus_file, 'r') as cf:
        context = cf.readlines()

        for line in context:
            # line = line.replace('\n', '')
            sp = line.split(' ')
            sp = sp[1:-1]
            trans = ''

            for s in sp:
                if s != '':
                    ascii = dict.token2id[s] + 65
                    if ascii > 90:
                        ascii += 6
                    trans += chr(ascii)

            trans_list.append(trans + '\n')

    with open(trans_corpus_file, 'w') as tf:
        tf.writelines(trans_list)

    print('Done!')