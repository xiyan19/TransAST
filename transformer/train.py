# -*- coding: utf-8 -*-

"""
    This script is used for the Transformer model training process.
    —————————————————
    usage: Usage is not independent

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/05/28
    $Annotation: Create.
    $Author: xiyan19
"""


from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import sacrebleu
import gensim
from beam_decoder import beam_search
# from model_transformer import batch_greedy_decode
# from torch.cuda.amp import autocast, GradScaler
import os
import xprint
import sys


# def eval_epoch(data, model, loss_compute):
#     total_tokens = 0.
#     total_loss = 0.
#
#     for batch in tqdm(data):
#         out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
#         loss = loss_compute(out, batch.trg_y, batch.ntokens)
#
#         total_loss += loss
#         total_tokens += batch.ntokens
#     return total_loss / total_tokens


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def train(train_data, dev_data, model, model_par, criterion, optimizer, args):

    # best_bleu_score = 0.0
    best_dev_loss = 100000
    epoch_num = 100
    steps = 0
    last_step = 0

    # args.scaler = GradScaler()

    for epoch in range(1, epoch_num + 1):
        for batch in train_data:
            model.train()
            torch.cuda.empty_cache()

            # try:
            # with autocast():
            out = model_par(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss_func = MultiGPULossCompute(model.generator, criterion, args, optimizer)
            train_loss = loss_func(out, batch.trg_y, batch.ntokens)
            # except RuntimeError as e:
            #     if 'CUDA out of memory' in str(e):
            #         xprint.fprint('WARNING: CUDA out of memory, skipping this batch.')
            #         continue
            #     else:
            #         raise e

            steps += 1
            sys.stdout.write('\rBatch[{}] - loss: {:.6f}'.format(steps, train_loss))

            if steps % args.test_interval == 0:
                model.eval()
                torch.cuda.empty_cache()

                with torch.no_grad():
                    total_tokens = 0.
                    total_loss = 0.
                    # bleu_ = 0.
                    # size_ = 0.
                    for batch in tqdm(dev_data):
                        # with autocast():
                        out = model_par(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
                        loss_func = MultiGPULossCompute(model.generator, criterion, args, None)
                        dev_loss_b = loss_func(out, batch.trg_y, batch.ntokens)
                        total_loss += dev_loss_b
                        total_tokens += batch.ntokens

                        # trg_sent = batch.trg_text
                        # src = batch.src
                        # src_mask = (src != 0).unsqueeze(-2)
                        # decode_result, _ = beam_search(model, src, src_mask, args.max_length,
                        #                                args.padding_idx, args.bos_idx, args.eos_idx,
                        #                                args.beam_size, args.device)
                        # # decode_result = batch_greedy_decode(model, src, src_mask,max_len=args.max_length,
                        # #                                     start_symbol=args.bos_idx, end_symbol=args.eos_idx)
                        # decode_result = [h[0] for h in decode_result]
                        # translation = [ids2word(args.dict, _s, args.padding_idx) for _s in decode_result]
                        # for i in range(len(translation)):
                        #     translation_str = ' '.join(translation[i])
                        #     bleu = sacrebleu.sentence_bleu(translation_str, [trg_sent[i]])
                        #     bleu_ += float(bleu.score)
                        #     size_ += 1

                    dev_loss = float(total_loss / total_tokens)
                    xprint.sprint('Batch: [{}] - Dev loss: {:.6f}'.format(steps, dev_loss))
                    # bleu_score = float(bleu_ / size_)
                    # xprint.sprint('Batch: [{}] - Dev loss: {:.6f}, Bleu Score: {}'.format(steps, dev_loss, bleu_score))

                    # if bleu_score > best_bleu_score:
                    #     best_bleu_score = bleu_score
                    #     last_step = steps
                    #     if args.save_best:
                    #         save(model, args.save_dir, 'best', steps)
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        last_step = steps
                        if args.save_best:
                            save(model, args.save_dir, 'best', steps)
                    else:
                        if steps - last_step >= args.early_stop and steps >= args.least_step:
                            xprint.sprint('early stop by {} steps.'.format(args.early_stop))
                            exit(0)
            if steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""

    def __init__(self, generator, criterion, args, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.devices = args.device_id
        self.criterion = nn.parallel.replicate(criterion, devices=self.devices)
        self.opt = opt
        self.args = args

        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # with autocast():
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                # self.args.scaler.scale(l_).backward()
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            # self.args.scaler.scale(o1).backward(gradient=o2)

            self.opt.step()
            if self.args.use_noamopt:
                # self.opt.step()
                # self.args.scaler.step(self.opt.optimizer)
                # self.args.scaler.update()
                self.opt.optimizer.zero_grad()
            else:
                # self.args.scaler.step(self.opt)
                # self.args.scaler.update()
                self.opt.zero_grad()

        return total * normalize
