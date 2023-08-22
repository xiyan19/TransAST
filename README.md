# TransAST: A Machine Translation-Based Approach for Obfuscated Malicious JavaScript Detection

![](https://github.com/xiyan19/TransAST/blob/main/docs/overview_page-0001.jpg)

Yan Qin, Weiping Wang†, Zixian Chen, Hong Song, Shigeng Zhang(†corresponding author)[Paper](https://ieeexplore.ieee.org/document/10202623) 

## Requirements

The codebase is tested on a server with Intel Xeon Silver 4114 2.20GHz

-   Ubuntu 16.04
-   Python 3.10
-   PyTorch 1.13.0
-   Gensim  3.8.1
-   Sentencepiece  0.1.95
-   2 NVIDIA TITAN V 12G with CUDA version 11.7, and 256G of memory

To run the code, please install  the relevant python packages.

## [](https://github.com/qiuyu96/codef#data)Data

### [](https://github.com/qiuyu96/codef#provided-data)Provided data

We have provided some JS file and corresponding obfuscated JS file **(To be uploaded)** for quick test. Please download and unzip the data.

### [](https://github.com/qiuyu96/codef#customize-your-own-data)Customize your own data

_Stay tuned for data preparation scripts._

Please organize your own data as follows:

```
TransAST
│
└─── Origin JS file (or corresponding AST sequences file)
    │
    └─ train
	     └─ Benign JS or AST sequences
	     └─ Malicious JS or AST sequences
    └─ test
	     └─ Benign JS or AST sequences
	     └─ Malicious JS or AST sequences
│
└─── Corresponding Ofuscated JS file (or corresponding AST sequences file)
    │
    └─ train
	     └─ Benign JS or AST sequences
	     └─ Malicious JS or AST sequences
    └─ test
	     └─ Benign JS or AST sequences
	     └─ Malicious JS or AST sequences	     
│
└─── ...

```
The corresponding obfuscation code file can be obtained  the origin code processed by [JavaScript Obfuscator Tool.](https://obfuscator.io/)  

## Compressed sequence

**./js2astseq.py**: The JS code is processed into an AST sequence.

**./transformer/courpus.py**: Create a corpus and generate the raw_corpus.

**./transformer/tokenize_transformer.py**: Create a dictionary based on raw_corpus and generate a dict.

**./transformer/trans_corpus.py**: Compress the corpus based on the generated dict dictionary, and change the raw_corpus into char_corpus

**./transformer/spm.py**: Train the compressed data from char_corpus to generate an spm model (.model) for subsequent feature processing.

where
-   `'.........'`: Be replaced with the specific location of the corresponding file.

Please check configuration files in  `configs/`, and you can always add your own model config.


## Translation task

**./transformer/engine.py** : Train translation model.

where

-   `-l1-path`: Put the origin char_corpus of train set;
-   `-l2-path`: Put the the corresponding obfuscated char_corpus of train set;
-   `-test-l1-path`: Origin char_corpus file path of test set;
-   `-test-l2-path`: The corresponding obfuscated char_corpus file path of test set;
-   `-dict`: File path of dict;
-   `-spm`: File path of spm model (.model);
-   `-save-dir`: The path where the model is saved;


**./transformer/translate.py** : Use the trained model to translate, that is, to de-obfuscate the obfuscated JS code into origin JS code.

where

-   `-input`: Obfuscated char_corpus file path of test set;
-   `-output`: File path of  Origin char_corpus file from the translation model processes the obfuscated char_corpus file;
-   `-model`: File path of the translation model;
-   `-dict`: File path of dict;
-   `-spm`: File path of spm model (.model);

Please carefully check configuration parameter, and you can always add your own model config.


## Detection task


**./TextCNN_SPM.py** : Use the textCNN model to test the performance of the translation model translation (deobfuscation)


where

-   `-file-path`:  Origin char_corpus file path of train set;
-   `-dir-path`: Origin char_corpus file path of train set(remove the char_corpus file name);
-   `-test-file-path`:  Origin char_corpus file path of test set;
-   `-test-dir-path`: Origin char_corpus file path of test set(remove the char_corpus file name);
-   `-shuffle`: Default;
-   `-sp`: File path of spm model (.model);
-   `-embed-len`: Default is 1000;
-   `-device`: Default is 0;
-   `-test`: The second run is used;
-   `-snapshot`: File path of the model trained by textCNN during the first run;

Please carefully check configuration parameter, and you can always add your own model config.

_Note_: You should run it twice, the first time to see how the model performs on the origin JS code detection, and the second time to see how the model performs when the training set is the origin JS code and the test set is the de-obfuscated JS code (i.e. the translation performance).`-test`and `-snapshot`only are used during the second run.


### BibTeX

@article{10202623,
  author={Qin, Yan and Wang, Weiping and Chen, Zixian and Song, Hong and Zhang, Shigeng},
  booktitle={2023 53rd Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)}, 
  title={TransAST: A Machine Translation-Based Approach for Obfuscated Malicious JavaScript Detection}, 
  year={2023},
  volume={},
  number={},
  pages={327-338},
  doi={10.1109/DSN58367.2023.00040}}
