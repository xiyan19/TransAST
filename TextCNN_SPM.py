# -*- coding: utf-8 -*-

"""
    This script uses TextCNN model to test the detection effect of the compressed sequence.
    —————————————————
    usage: python3 TextCNN_SPM.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/09
    $Annotation: Create.
    $Author: xiyan19
"""


import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import dataset_SPM
from torch.utils.data import DataLoader
import datetime
import sys
import xprint
import sentencepiece as spm


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        in_channels = 1
        out_channels = args.kernel_num
        kernel_size = args.kernel_sizes
        vocab_num = args.spm.vocab_size()
        dim = args.dim

        self.embed = nn.Embedding(vocab_num, dim)
        if self.args.static:
            self.embed.weight.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (k, dim)) for k in kernel_size])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_size) * out_channels, class_num)

    def forward(self, x):
        x = self.embed(x)  # (batch_size, seq_len, dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, out_channels, seq_len-kernel_size+1), ...] * len(kernel_size)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, out_channels), ...] * len(kernel_size)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (batch_size, len(kernel_size) * out_channels)
        out = self.fc(x)  # (batch_size, class_num)

        return out


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    # best_acc = 0
    best_loss = 1000
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            feature, target = batch[0], batch[1]
            # feature.t_()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / len(batch[1])
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             len(batch[1])))

            if steps % args.test_interval == 0:
                dev_loss = eval(dev_iter, model, args)
                # if dev_acc > best_acc:
                #     best_acc = dev_acc
                #     last_step = steps
                #     if args.save_best:
                #         save(model, args.save_dir, 'best', steps)
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop and steps >= args.least_step:
                        xprint.sprint('early stop by {} steps.'.format(args.early_stop))
                        exit(0)
            if steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    TP, TN, FN, FP = 0, 0, 0, 0
    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch[0], batch[1]
            # feature.t_()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = model(feature)
            loss = F.cross_entropy(logit, target, reduction='sum')

            avg_loss += loss.item()
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            # for pre, trg in zip(torch.max(logit, 1)[1].view(target.size()).data, target):
            #     if trg == 0:
            #         if pre == 0:
            #             tn += 1
            #         else:
            #             fp += 1
            #     else:
            #         if pre == 0:
            #             fn += 1
            #         else:
            #             tp += 1
            # TP    predict 和 label 同时为1
            TP += ((torch.max(logit, 1)[1].view(target.size()).data == 1) & (target.data == 1)).cpu().sum()
            # TN    predict 和 label 同时为0
            TN += ((torch.max(logit, 1)[1].view(target.size()).data == 0) & (target.data == 0)).cpu().sum()
            # FN    predict 0 label 1
            FN += ((torch.max(logit, 1)[1].view(target.size()).data == 0) & (target.data == 1)).cpu().sum()
            # FP    predict 1 label 0
            FP += ((torch.max(logit, 1)[1].view(target.size()).data == 1) & (target.data == 0)).cpu().sum()


    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    precision = 100.0 * TP / (TP + FP)
    recall = 100.0 * TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    xprint.sprint('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})  pre: {:.4f}%  recall: {:.4f}%  f1: {:.4f}%  tp: {}  fp: {}  tn: {} fn: {}'
                  .format(avg_loss,
                          accuracy,
                          corrects,
                          size,
                          precision,
                          recall,
                          f1,
                          TP,
                          FP,
                          TN,
                          FN))
    return avg_loss


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TextCNN classifier')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=1000,
                        help='how many steps to wait before saving [default:1000]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-least-step', type=int, default=10000,
                        help='least iteration numbers to stop without performance increasing [default:10000]')
    parser.add_argument('-early-stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # data
    parser.add_argument('-file-path', type=str, help='the corpus file of the train data')
    parser.add_argument('-dir-path', type=str, help='the dir of the train data')
    parser.add_argument('-test-file-path', type=str, help='the corpus file of the test data')
    parser.add_argument('-test-dir-path', type=str, help='the dir of the test data')
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    # parser.add_argument('-dict', type=str, help='dictionary of source corpus')
    parser.add_argument('-sp', type=str, help='bpe model')
    parser.add_argument('-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-embed-len', type=int, default=1000, help='number of embedding length [default: 1000]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()


    xprint.sprint('Loading dict...')
    sp = spm.SentencePieceProcessor()
    sp.Load(args.sp)
    args.spm = sp


    xprint.sprint('Loading data...')
    train_dataset = dataset_SPM.SeqDataset(file_path=args.file_path, dir_path=args.dir_path, model=args.spm, maxlength=args.embed_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    dataset = dataset_SPM.SeqDataset(file_path=args.test_file_path, dir_path=args.test_dir_path, model=args.spm, maxlength=args.embed_len)
    dev_dataset, test_dataset = dataset.split(2000)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size*2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size*2)

    args.class_num = 2
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    xprint.sprint('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # model
    cnn = TextCNN(args)
    if args.snapshot is not None:
        xprint.sprint('Loading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))

    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()

    # train
    if args.test:
        try:
            eval(test_dataloader, cnn, args)
        except Exception as e:
            xprint.fprint(e)
    else:
        try:
            train(train_dataloader, dev_dataloader, cnn, args)
        except KeyboardInterrupt:
            xprint.sprint('-' * 69)
        xprint.sprint('Exiting from training early')
