# -*- coding: utf-8 -*-

"""
    This script converts the character format into a corpus.
    —————————————————
    usage: python3 trans_corpus2.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/16
    $Annotation: Create.
    $Author: xiyan19
"""


import gensim


if __name__ == "__main__":
    corpus_file = '/home/guest/JSObuAST/test_jo_high/TRANSSEQ/raw_corpus'
    trans_corpus_file = '/home/guest/JSObuAST/test_jo_high/TRANSSEQ/char_corpus'

    dict = gensim.corpora.dictionary.Dictionary.load('/home/guest/JSObuAST/model/ast.dict')

    trans_list = []

    A = list(dict.token2id.keys())
    B = list(dict.token2id.values())

    with open(trans_corpus_file, 'r') as cf:
        context = cf.readlines()

        for line in context:
            sp = list(line)

            trans = ''

            for s in sp[:-1]:
                if s != ' ' and s != '⁇':
                        idx = ord(s)
                        if idx > 96:
                            idx -= 6
                        idx -= 65
                        trans += A[B.index(idx)] + ' '

            trans_list.append('Program ' + trans + 'End\n')

    with open(corpus_file, 'w') as tf:
        tf.writelines(trans_list)

    print('Done!')
