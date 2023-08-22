# -*- coding: utf-8 -*-

"""
    This script builds a corpus.
    —————————————————
    usage: python3 corpus.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/05/28
    $Annotation: Create.
    $Author: xiyan19
"""


import search

if __name__ == "__main__":
    path = '.........' # train SEQ folders, including Benign and Malicious
    fileList = []
    lines = []

    search.search_dir(path, fileList, 'seq')

    for file in fileList:
        with open(file, 'r') as f: # 1
            corpus = f.read()
            lines.append(corpus + '\n') # 1

    # with open(path+'/corpus', "w") as fc:
    with open(path + '/raw_corpus', "w") as fc: # 1
        fc.writelines(lines)

    print("lines of corpus: ", len(lines))
    print("-------- Get Corpus ! --------")
