# Rebuilt Sentence-State LSTM

This is our project report for the course "Natural Network and Deep Learning, Autumn 2018". 
We reproduce the ACL 2018 paper [Sentence-State LSTM for Text Representation](https://arxiv.org/abs/1805.02474) based on the structure of [FastNLP](https://github.com/fastnlp/fastNLP).
We test our rebuilt model on classification and sequence labeling (POS and NER) tasks.
Besides, we also apply the famous modular BERT on this task to explore if the pretrained language model can bring about additional promotions.
Please refer to the [Report.pdf](./Report.pdf) for detailed description.

## File Organization

```
SLSTM
├── cache               # cache for data in pkl format
├── data                # data for three tasks
│   ├── cls     
│   ├── ner    
│   └── pos
├── model               # model 
│   ├── __init__.py         # init file
│   ├── BertModel.py        # Bert classification model
│   └── SModel.py           # SLSTM model
├── save                # save models
│   ├── cls
│   ├── ner
│   └── pos      
├── args.py             # cmdline args
├── dataset.py          # dataset for Conll format
├── preprocess.py       # load data
├── slstm_cls.py        # train-val-test for classification
├── slstm_sl.py         # train-val-test for sequence labeling
├── Report.pdf          # description for this project 
└── README.md           # Readme 
```



## Environment

- System: Ubuntu 16.04
- Development Environment:
  - Python 3.6
- Requirements
  - Pytorch 1.0.0
  - FastNLP 0.3.0
  - pytorch-pretrained-BERT
  



## Install

``` shell
# for pytorch and fastnlp
# refer to https://github.com/fastnlp/fastNLP for more information
$ pip install fastNLP

# for bert
# refer to https://github.com/huggingface/pytorch-pretrained-BERT for more information
$ pip install pytorch-pretrained-bert
$ git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
$ cd pytorch-pretrained-BERT
$ pip install .
```



## Run the code

### Classification
``` shell
$ CUDA_VISIBLE_DEVICES=0 python slstm_cls.py --dataset mr --data_dir data/cls --glove_path /path/to/glove.42B.300d --batch_size 16
```

### Sequence labeling
``` shell
$ CUDA_VISIBLE_DEVICES=0 python slstm_sl.py --dataset pos --data_dir data/pos --glove_path /path/to/glove.42B.300d --batch_size 16
```
