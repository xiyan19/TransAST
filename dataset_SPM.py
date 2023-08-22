# -*- coding: utf-8 -*-

"""
    This script is used to build data set objects with sequence compression.
    —————————————————
    usage: Usage is not independent

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/09
    $Annotation: Create.
    $Author: xiyan19
"""


from torch.utils.data import Dataset
import search
import pandas as pd
import numpy as np
from sklearn import model_selection
import torch


def get_label(x):
    if 'Benign' in x or 'benign' in x:
        return 0
    elif 'Malicious' in x or 'malicious' in x:
        return 1
    else:
        print('Label Error! Info: ' + str(x))
        exit(-1)


class SeqDataset(Dataset):

    def __init__(self, file_path=None, dir_path=None, dataframe=None, model=None, maxlength=None):

        self.model = model
        self.maxlength = maxlength

        if dataframe is None:
            data = []
            with open(file_path, 'r') as f:
                context = f.readlines()
            filelist = []
            search.search_dir(dir_path, filelist, 'seq')

            for l, file in zip(context, filelist):
                data.append([l, file])


            self.df = pd.DataFrame(data, columns=['seq', 'file'])
            self.df['label'] = self.df['file'].apply(lambda x: get_label(x))
        else:
            self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq_ = self.df.seq[idx]
        seq_ = [2] + self.model.EncodeAsIds(seq_) + [3] # 1
        if len(seq_) < self.maxlength:
            seq_.extend([0] * (self.maxlength - len(seq_)))
        else:
            seq_ = seq_[:self.maxlength]

        seq_index = np.array(seq_).astype(int)
        seq_index = torch.from_numpy(seq_index)

        return seq_index, self.df.label[idx]

    def split(self, test=0.1):
        skf = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=69)

        for dev_index, test_index in skf.split(self.df, self.df['label']):
            dev_data, test_data = self.df.iloc[dev_index], self.df.iloc[test_index]

            dev_data = dev_data.reset_index()
            test_data = test_data.reset_index()

            return SeqDataset(dataframe=dev_data, model=self.model, maxlength=self.maxlength), \
                   SeqDataset(dataframe=test_data, model=self.model, maxlength=self.maxlength)
