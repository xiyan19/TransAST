# -*- coding: utf-8 -*-

"""
    This script performs translation tasks using the Transformer model.
    https://github.com/hemingkx/ChineseNMT
    —————————————————
    usage: python3 translate.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/10
    $Annotation: Create.
    $Author: xiyan19
"""


import argparse
import gensim
import data_translate
from torch.utils.data import DataLoader
import xprint
from model_transformer import make_model
import torch
# from tqdm import tqdm
import warnings
import sentencepiece as spm
from beam_decoder import beam_search
from model_transformer import batch_greedy_decode
import search


def translate(data, model, args):
    model.eval()

    with torch.no_grad():
        i = 1

        if args.con:
            c_filelist = []
            search.search_dir(args.output, c_filelist, 'seq')

        # for batch in tqdm(data):
        for batch in data:
            print(i)
            i += 1
            if i < 1421 : continue
            if args.con:
                f = batch[2][0].replace(args.input.replace('/char_corpus', ''), args.output)
                if f in c_filelist:
                    continue

            src = batch[1]
            src = src.to(args.device)
            src_mask = (src != 0).unsqueeze(-2)
            if args.use_beam:
                decode_result, _ = beam_search(model, src, src_mask, args.max_length,
                                           args.padding_idx, args.bos_idx, args.eos_idx,
                                           args.beam_size, args.device)
                decode_result = [h[0] for h in decode_result]
            else:
                decode_result = batch_greedy_decode(model, src, src_mask, max_len=args.max_length)
            translation = [args.spm.decode_ids(_s) for _s in decode_result]

            temp = batch[2][0].split('/')
            output_file = args.output + '/' + temp[-2] + '/' + temp[-1]

            with open(output_file, "w") as fp:
                fp.write(translation[0])
            # print(i)
            # i += 1


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Transformer model for translate task')

    # data
    parser.add_argument('-input', type=str, help='where to load the data')
    parser.add_argument('-output', type=str, help='where to save the result')

    # model
    parser.add_argument('-model', type=str, help='transformer model')
    parser.add_argument('-dict', type=str, help='dictionary of source corpus')
    parser.add_argument('-spm', type=str, help='bpe model')

    #method
    parser.add_argument('-use-beam', action='store_true', default=False, help='beam search or greedy search')

    # device
    parser.add_argument('-device', type=str, default='-1',
                        help='use for model, -1 mean cpu [default: -1]')

    parser.add_argument('-con', action='store_true', default=False, help='continue the work')
    parser.add_argument('-part-size', type=int, default=None, help='the size of split')
    parser.add_argument('-part-num', type=int, default=None, help='the num of split')
    args = parser.parse_args()

    # GPU
    if args.device != '-1':
        args.device = torch.device(f"cuda:{args.device}")
    else:
        args.device = torch.device('cpu')


    xprint.sprint("-------- Loading Dictionary --------")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm)
    args.spm = sp
    args.vocab_size = sp.vocab_size()
    args.padding_idx, args.bos_idx, args.eos_idx = 0, 2, 3 # 1 0, 2, 3
    args.dict = gensim.corpora.dictionary.Dictionary.load(args.dict)
    args.max_length = 1000
    args.beam_size = 3

    xprint.sprint("-------- Dataset Build --------")
    dataset = data_translate.SeqDataset(path=args.input, dict=args.dict, model=args.spm, maxlength=args.max_length, args=args)
    dataloader = DataLoader(dataset, batch_size=1)

    # for data in dataloader:
    #     print('train')

    xprint.sprint("-------- Get Dataloader --------")

    xprint.sprint('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # loading model
    model = make_model(src_vocab=args.vocab_size, tgt_vocab=args.vocab_size, d_model=128, d_ff=512, h=2,
                       maxlength=args.max_length, device=args.device)
    model.load_state_dict(torch.load(args.model))


    translate(dataloader, model, args)
