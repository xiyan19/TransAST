# -*- coding: utf-8 -*-

"""
    This script does BPE segmentation.
    —————————————————
    usage: python3 spm.py

    Last commit info:
    ~~~~~~~~~~~~~~~~~
    $LastChangedDate: 2022/06/06
    $Annotation: Create.
    $Author: xiyan19
"""


import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    # input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
    #                  '--pad_id=40 --unk_id=41 --bos_id=24 --eos_id=11 --max_sentence_length=30000'
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                     '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    spm.SentencePieceTrainer.Train(cmd)


def test(model):
    sp = spm.SentencePieceProcessor()
    text = 'IXELMMMMMMMMDIVBMELMMMDJVBMCMOBMOCOMYMIBMCCCBMEQMMCMOOOMUMOEBMLMDIVBMEQMMMBMQEQMMMOBQMMCOMEQQMMMMMCCCCMFCOEQMMOOOOEQRMMOEMCMOMMOOG'

    sp.Load(model)
    print(sp.EncodeAsPieces(text))
    a = sp.EncodeAsIds(text)
    print(a)
    print(sp.decode_ids(a))
    print(str(len(text)) + ' -- ' + str(len(a)))


if __name__ == "__main__":
    # input_file, vocab_size, model_name, model_type, character_coverage, max_sentence_length
    train('............../char_corpus', 10000, 'JSob_spm', 'bpe', 1)
    print('done')
