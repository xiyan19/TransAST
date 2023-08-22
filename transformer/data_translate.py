# -*- coding: utf-8 -*-

"""
    This script is used to build data set objects for translation tasks.
    —————————————————
    usage: Usage is not independent

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/10
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


class SeqDataset(Dataset):

    def __init__(self, path=None, dict=None, model=None, maxlength=None, args=None):

        self.model = model
        self.dict = dict
        self.maxlength = maxlength
        self.pad = 0 # 1

        data = []
        with open(path, 'r') as f:
            context = f.readlines()

        filelist = []
        search.search_dir(path.replace('/char_corpus1', ''), filelist, 'seq') # 1

        for text, file in zip(context, filelist):
            data.append([text, file])

        # split data
        if args.part_size is not None and args.part_num is not None:
            start = args.part_size * args.part_num
            end = args.part_size * (args.part_num + 1)
            if end > len(filelist):
                end = len(filelist)
            data = data[start:end]

        self.df = pd.DataFrame(data, columns=['text', 'file'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq_ = self.df.text[idx]
        seq_ = [2] + self.model.EncodeAsIds(seq_) + [3] # 1
        if len(seq_) < self.maxlength:
            seq_.extend([self.pad] * (self.maxlength - len(seq_)))
        else:
            seq_ = seq_[:self.maxlength]

        seq_index = np.array(seq_).astype(int)
        seq_index = torch.from_numpy(seq_index)

        seq_text = self.model.decode_ids(seq_)

        return seq_text, seq_index, self.df.file[idx]
