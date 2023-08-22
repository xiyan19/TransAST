# -*- coding: utf-8 -*-

"""
    This script calculates the similarity between two strings.
    —————————————————
    usage: python3 distance.py [-origin file] [-obfs file] [-w2v model_path] [-max-length value]

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/16
    $Annotation: create.
    $Author: xiyan19
"""


import argparse
import gensim
from nltk.metrics import jaccard_distance
from nltk.cluster import cosine_distance
import torch
import numpy as np
import sacrebleu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distance')
    parser.add_argument('-origin', type=str, default='/home/czx/pycharm/JSob/JSob_seq/origin/test/raw_corpus', help='path of the origin dataset')
    parser.add_argument('-obfs', type=str, default='/home/czx/pycharm/JSob/JSob_seq/medium/test/raw_corpus', help='path of the obfuscation dataset')
    # parser.add_argument('-w2v', type=str, default='model/new/w2v.model', help='path of the dictionary')
    parser.add_argument('-max-length', type=int, default=10000, help='max length of seq [default: 10000]')
    args = parser.parse_args()


    jaccard = []
    # cosine = []
    bleu = []



    # loading data
    with open(args.origin, 'r') as f:
        origin = f.readlines()
    with open(args.obfs, 'r') as f:
        obfs = f.readlines()

    for origin_seq, obfs_seq in zip(origin, obfs):

        origin_seq = origin_seq.replace('\n', '')
        origin_seq_temp = origin_seq.split(' ')
        origin_seq_set = set()

        obfs_seq = obfs_seq.replace('\n', '')
        obfs_seq_temp = obfs_seq.split(' ')
        obfs_seq_set = set()

        origin_seq_vector = torch.zeros(args.max_length)
        for i in range(args.max_length):
            if i < len(origin_seq_temp):
                # origin_seq_vector[i] = torch.tensor(dictionary.wv.key_to_index[origin_seq_temp[i].lower()])
                origin_seq_set.add(origin_seq_temp[i].lower())
            # else:
                # origin_seq_vector[i] = torch.tensor(dictionary.wv.key_to_index['End'.lower()])

        obfs_seq_vector = torch.zeros(args.max_length)
        for i in range(args.max_length):
            if i < len(obfs_seq_temp):
                # obfs_seq_vector[i] = torch.tensor(dictionary.wv.key_to_index[obfs_seq_temp[i].lower()])
                obfs_seq_set.add(obfs_seq_temp[i].lower())
            # else:
                # obfs_seq_vector[i] = torch.tensor(dictionary.wv.key_to_index['End'.lower()])

        # jaccard distance 
        jaccard_distance_ = jaccard_distance(origin_seq_set, obfs_seq_set)
        jaccard.append(jaccard_distance_)

        # cosine_distance_ = cosine_distance(origin_seq_vector, obfs_seq_vector)
        # cosine.append(cosine_distance_)

        # BLEU score
        obfs_seq_maxlength = ' '.join(obfs_seq_temp[:args.max_length])
        origin_seq_maxlength = ' '.join(origin_seq_temp[:args.max_length])
        bleu_ = sacrebleu.sentence_bleu(obfs_seq_maxlength, [origin_seq_maxlength])
        bleu.append(float(bleu_.score))

    jaccard = np.array(jaccard)
    # cosine = np.array(cosine)
    bleu = np.array(bleu)

    print('Jaccard distance mean: ' + str(jaccard.mean()))
    print('Jaccard distance median: ' + str(np.median(jaccard)))
    # print('Cosine distance mean: ' + str(cosine.mean()))
    # print('Cosine distance median: ' + str(np.median(cosine)))
    print('Bleu mean: ' + str(bleu.mean()))
    print('Bleu median: ' + str(np.median(bleu)))
