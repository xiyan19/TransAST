# -*- coding: utf-8 -*-

"""
    This script is used to build data set objects.
    —————————————————
    usage: Usage is not independent

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/05/24
    $Annotation: Create.
    $Author: xiyan19
"""


from torch.utils.data import Dataset
import search
import pandas as pd
import numpy as np
from sklearn import model_selection
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable


def get_label(x):
    if 'Benign' in x or 'benign' in x:
        return 0
    elif 'Malicious' in x or 'malicious' in x:
        return 1
    else:
        print('Label Error! Info: ' + str(x))
        exit(-1)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=None, device=None):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(device)
        self.src = src

        self.src_mask = (src != pad).unsqueeze(-2)
 
        if trg is not None:
            trg = trg.to(device)
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()


    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class SeqDataset(Dataset):

    def __init__(self, l1_path=None, l2_path=None, dataframe=None, dict=None, model=None, maxlength=None, train=True, device=None):

        self.model = model
        self.dict = dict
        self.train = train
        self.device = device
        self.maxlength = maxlength
        self.PAD = 0 # 1
        # self.PAD = len(model.token2id)

        if dataframe is None:
            # l1_filelist = []
            # search.search_dir(l1_path, l1_filelist, 'seq')
            #
            # data = []
            # for l1_file in l1_filelist:
            #     l2_file = l1_file.replace(l1_path, l2_path)
            #     with open(l1_file, 'r') as f1:
            #         row1 = f1.read()
            #         l1_length = len(row1.split())
            #     # with open(l2_file, 'r') as f2:
            #     #     row2 = f2.read()
            #     data.append([l1_file, l2_file, l1_length])
            data = []
            with open(l1_path, 'r') as f:
                context1 = f.readlines()
            with open(l2_path, 'r') as f:
                context2 = f.readlines()
            for l1, l2 in zip(context1, context2):
                l2_length = len(l2)
                data.append([l1, l2, l2_length])

            self.df = pd.DataFrame(data, columns=['origin', 'obfuscation', 'length'])
            self.df['length'] = self.df['length'].astype(int)
            if self.train:
                self.df.sort_values(by='length', inplace=True, ascending=True)
                self.df = self.df.reset_index()
        else:
            self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        origin_seq_ = self.df.origin[idx]
        origin_seq_ = [2] + self.model.EncodeAsIds(origin_seq_) + [3] # 1
        origin_seq_ = origin_seq_[:self.maxlength]

        obfs_seq_ = self.df.obfuscation[idx]
        obfs_seq_ = [2] + self.model.EncodeAsIds(obfs_seq_) + [3] # 1
        obfs_seq_ = obfs_seq_[:self.maxlength]

        origin_seq_index, obfs_seq_index = np.array(origin_seq_).astype(int), np.array(obfs_seq_).astype(int)
        origin_seq_index, obfs_seq_index = torch.from_numpy(origin_seq_index), torch.from_numpy(obfs_seq_index)

        origin_seq_text = self.model.decode_ids(origin_seq_)
        obfs_seq_text = self.model.decode_ids(obfs_seq_)

        return [obfs_seq_text, origin_seq_text, obfs_seq_index, origin_seq_index]
        # with open(self.df.origin[idx], 'r') as f:
        #     origin_seq = f.read()
        #     origin_seq = origin_seq.split(' ')
        #     origin_seq = origin_seq[:self.maxlength]
        #     origin_seq_ = []
        #     for word in origin_seq:
        #         origin_seq_.append(self.model.token2id[word])
        #
        # with open(self.df.obfuscation[idx], 'r') as f:
        #     obfs_seq = f.read()
        #     obfs_seq = obfs_seq.split(' ')
        #     obfs_seq = obfs_seq[:self.maxlength]
        #     obfs_seq_ = []
        #     for word in obfs_seq:
        #         obfs_seq_.append(self.model.token2id[word])
        #
        # origin_seq_index, obfs_seq_index = np.array(origin_seq_).astype(int), np.array(obfs_seq_).astype(int)
        # origin_seq_index, obfs_seq_index = torch.from_numpy(origin_seq_index), torch.from_numpy(obfs_seq_index)
        #
        # origin_seq_text = ' '.join(origin_seq).replace('Program ', '').replace(' End', '')
        # obfs_seq_text = ' '.join(obfs_seq).replace('Program ', '').replace(' End', '')
        #
        # return [obfs_seq_text, origin_seq_text, obfs_seq_index, origin_seq_index]

    def split(self, test=0.1):
        skf = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=69)
        filelist = []
        # search.search_dir('/home/qy/JSObuAST/test/SEQ', filelist, 'seq')
        search.search_dir('/home/czx/pycharm/JSob/JSob_seq/origin/test', filelist, 'seq')
        self.df['filelist'] = filelist
        self.df['label'] = self.df['filelist'].apply(lambda x: get_label(x))
        for dev_index, test_index in skf.split(self.df, self.df['label']):
            dev_data, test_data = self.df.iloc[dev_index], self.df.iloc[test_index]

            dev_data.sort_values(by='length', inplace=True, ascending=True)
            test_data.sort_values(by='length', inplace=True, ascending=True)

            dev_data = dev_data.reset_index()
            test_data = test_data.reset_index()

            return SeqDataset(dataframe=dev_data, dict=self.dict, model=self.model, maxlength=self.maxlength, device=self.device), \
                   SeqDataset(dataframe=test_data, dict=self.dict, model=self.model, maxlength=self.maxlength, device=self.device)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]
        src_tokens = [x[2] for x in batch]
        tgt_tokens = [x[3] for x in batch]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD, self.device)
