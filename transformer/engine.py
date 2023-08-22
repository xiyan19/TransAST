# -*- coding: utf-8 -*-

"""
    This script is a Transformer model implemented using pytorch.
    https://github.com/hemingkx/ChineseNMT
    —————————————————
    usage: python3 engine.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/05/24
    $Annotation: Create.
    $Author: xiyan19
"""


import argparse
import gensim
import data_utils
from torch.utils.data import DataLoader
import xprint
from model_transformer import make_model, LabelSmoothing
import torch
# from train import train, test
from train import train
import warnings
import sentencepiece as spm


class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model, args):
    """for batch_size 32, 400 steps for one epoch, 2 epoch for warm-up"""
    # return NoamOpt(model.src_embed[0].d_model, 1, 800,
                   # torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-7))
    return NoamOpt(model.src_embed[0].d_model, 1, 800,
                   torch.optim.AdamW(model.parameters(), lr=0, eps=1e-7))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Transformer model')

    # learning
    parser.add_argument('-lr', type=float, default=3e-4, help='initial learning rate [default: 0.0003]')
    parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
    parser.add_argument('-max-length', type=int, default=1000, help='number of embedding length [default: 1000]')
    parser.add_argument('-use-smoothing', action='store_true', default=False, help='use label smoothing')
    parser.add_argument('-use-noamopt', action='store_true', default=False, help='use warmup')

    # data
    parser.add_argument('-l1-path', type=str, help='where to load the language 1 data')
    parser.add_argument('-l2-path', type=str, help='where to load the language 2 data')
    # parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    parser.add_argument('-test-l1-path', type=str, help='where to load test language 1 data')
    parser.add_argument('-test-l2-path', type=str, help='where to load test language 2 data')

    # Vob
    # parser.add_argument('-w2v', type=str, help='pretrained model of word to vector')
    parser.add_argument('-dict', type=str, help='dictionary of source corpus')
    parser.add_argument('-spm', type=str, help='bpe model')

    # device
    parser.add_argument('-device', type=str, default='-1',
                        help='use for model, -1 mean cpu [default: -1]')
    parser.add_argument('-device-id', type=str, default='-1',
                        help='use for iterate data, -1 mean cpu [default: -1]')

    # model
    parser.add_argument('-save-dir', type=str, help='where to save the snapshot')
    parser.add_argument('-test-interval', type=int, default=200,
                        help='how many steps to wait before testing [default: 200]')
    parser.add_argument('-save-best', type=bool, default=True,
                        help='whether to save when get best performance [default: True]')
    parser.add_argument('-least-step', type=int, default=20000,
                        help='least iteration numbers to stop without performance increasing [default:20000]')
    parser.add_argument('-early-stop', type=int, default=2000,
                        help='iteration numbers to stop without performance increasing [default: 2000]')
    parser.add_argument('-save-interval', type=int, default=200,
                        help='how many steps to wait before saving [default:200]')

    args = parser.parse_args()

    # GPU
    if args.device != '-1':
        args.device = torch.device(f"cuda:{args.device}")
    else:
        args.device = torch.device('cpu')
    args.device_id = [int(x) for x in args.device_id.split(',')]

    xprint.sprint("-------- Loading Dictionary --------")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm)
    args.spm = sp
    vocab_size = sp.vocab_size()
    args.padding_idx, args.bos_idx, args.eos_idx = 0, 2, 3 # 1
    args.dict = gensim.corpora.dictionary.Dictionary.load(args.dict)
    # vocab_size = len(args.dict.token2id) + 1
    # args.padding_idx = len(args.dict.token2id)
    # args.bos_idx, args.eos_idx = args.dict.token2id['Program'], args.dict.token2id['End']
    # args.beam_size = 3

    xprint.sprint("-------- Dataset Build --------")
    train_dataset = data_utils.SeqDataset(l1_path=args.l1_path, l2_path=args.l2_path, dict=args.dict, model=args.spm, maxlength=args.max_length, device=args.device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    dataset = data_utils.SeqDataset(l1_path=args.test_l1_path, l2_path=args.test_l2_path, dict=args.dict, model=args.spm, train=False, maxlength=args.max_length, device=args.device)
    dev_dataset, test_dataset = dataset.split(2000)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size*2, collate_fn=dev_dataset.collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=len(args.device_id)*2, collate_fn=test_dataset.collate_fn)

    # for tdata in train_dataloader:
    #     print('train')
    # for ddata in dev_dataloader:
    #     print('dev')
    # for ttdata in test_dataloader:
    #     print('test')

    xprint.sprint("-------- Get Dataloader --------")

    model = make_model(src_vocab=vocab_size, tgt_vocab=vocab_size, d_model=128, d_ff=512, h=2,
                       maxlength=args.max_length, device=args.device)
    model_par = torch.nn.DataParallel(model, device_ids=args.device_id)

    xprint.sprint('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.use_smoothing:
        criterion = LabelSmoothing(size=vocab_size, padding_idx=args.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    if args.use_noamopt:
        optimizer = get_std_opt(model, args)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-7)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer, args) # epoch_num = 100
    # test(test_dataloader, model, criterion, args)
