# -*- coding: utf-8 -*-

"""
    This script extracts abstract syntax trees and sequences from the JS file.
    —————————————————
    usage: python3 JS2ASTSEQ.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/02/28
    $Annotation: Create.
    $Author: xiyan19
"""


from pyjsparser import parse
import json
# import search
import os
import sys
sys.setrecursionlimit(1000000)

focus = ['type']
str_focus = ['name', 'value', 'type']


def json_extract(obj, keys):
    # Recursively fetch values from nested JSON.
    arr = []

    def extract(obj, arr, keys):
        # Recursively search for values of keys in JSON tree.
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, keys)
                elif k in keys:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, keys)
        return arr

    values = extract(obj, arr, keys)
    return values


def js2astseq(js_file_list, ast_output_dir, seq_output_dir, str_seq_output_dir):
    i = 1
    j = 10

    for js in js_file_list:
        try:
            with open(js, 'r') as fp:
                temp1 = js.split('/')[-1]
                temp2 = temp1.split('.')[0]
                # ast_output = ast_output_dir + '/' + temp2 + '.json'
                seq_output = seq_output_dir + '/' + temp2 + '.seq'
                # str_seq_output = str_seq_output_dir + '/' + temp2 + '.seq'

                i += 1

                js_code = fp.read()
                ast = parse(js_code)
                seq = json_extract(ast, focus)
                # str_seq = json_extract(ast, str_focus)

                # with open(ast_output, 'w') as f:
                #     json.dump(ast, f)

                with open(seq_output, 'w') as f:
                    for line in seq:
                        f.write(str(line) + ' ')
                    f.write('End')

                # with open(str_seq_output, 'w') as f:
                #     for line in str_seq:
                #         f.write(str(line) + ' ')
                #     f.write('End')

        except Exception as e:
            print(js + ' ' + str(e))


        # if i % int(len(fileList) * 0.1) == 0:
        #     print('----------' + str(j) + '%----------')
        #     j += 10


if __name__ == '__main__':
    js_dir = '.........'
    ast_output_dir = '^^^^^^^^'  # ignore
    seq_output_dir = '.........'
    str_seq_output_dir = '^^^^^^^' # ignore


    fileList = os.listdir(js_dir)
    # search.search_dir(js_dir, fileList, 'js')
    fileList = [js_dir+"/"+i for i in fileList]
    print(len(fileList))


    # ast、seq、strseq
    js2astseq(fileList, ast_output_dir, seq_output_dir, str_seq_output_dir)

    print('Done.')

# /home/czx/data2/test_data/SIMPLE/JS/Benign/1222.js Line 54: Unexpected token =
# SIMPLETest  Benign:987/1308  Malicious:1340/1352
# SIMPLETrain Benign:4298/5584 Malicious:/6257/6259
