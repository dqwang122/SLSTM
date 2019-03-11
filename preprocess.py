import os

import torch

from fastNLP import DataSet, Vocabulary, Trainer, Tester
from fastNLP.io.embed_loader import EmbedLoader

def load_dataset(
        data_dir='/remote-home/ygxu/workspace/Product_all',
        data_path='mr.task.train',
        # bert_dir='/home/ygxu/BERT/BERT_English_uncased_L-12_H-768_A_12',
        bert_dir='/remote-home/ygxu/workspace/BERT/BERT_English_uncased_L-24_H-1024_A_16',
):

    path = os.path.join(data_dir, data_path)

    ds = DataSet.read_csv(path, headers=('label', 'raw_sentence'), sep='\t')

    ds.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')

    ds.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)

    def transfer_bert_to_fastnlp(ins):
        result = "[CLS] "
        bert_text = ins['bert_tokenize_list']
        for text in bert_text:
            result += text + " "
        return result.strip()

    with open(os.path.join(bert_dir, 'vocab.txt')) as f:
        lines = f.readlines()
    vocabs = []
    for line in lines:
        vocabs.append(line[: -1])

    vocab_bert = Vocabulary(unknown=None, padding=None)
    vocab_bert.add_word_lst(vocabs)
    vocab_bert.build_vocab()
    vocab_bert.unknown = '[UNK]'
    vocab_bert.padding = '[PAD]'

    from pytorch_pretrained_bert import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_dir, 'vocab.txt'))
    ds.apply(lambda x: tokenizer.tokenize(x['raw_sentence']), new_field_name='bert_tokenize_list')
    ds.apply(transfer_bert_to_fastnlp, new_field_name='bert_tokenize')
    ds.apply(lambda x: [vocab_bert.to_index(word) for word in x['bert_tokenize_list']], new_field_name='index_words',
             is_input=True)

    ds.rename_field('index_words', 'tokens')
    ds.apply(lambda x: [1.] * len(x['tokens']), new_field_name='masks', is_input=True)

    return ds


def combine_data_set(ds_a, ds_b):
    ds = DataSet()
    for ins in ds_a:
        ds.append(ins)
    for ins in ds_b:
        ds.append(ins)
    for k in ds_a.field_arrays.keys():
        ds.set_input(k, flag=ds_a.field_arrays[k].is_input)
        ds.set_target(k, flag=ds_a.field_arrays[k].is_target)
    return ds


def load_dataset_with_glove(
        data_dir,
        data_path='mr.task.train',
        glove_path="",
        load_glove=True,
        vocabs=None
):
    path = os.path.join(data_dir, data_path)
    print(f"start load dataset from {path}.")

    ds = DataSet.read_csv(path, headers=('label', 'sentence'), sep='\t')

    ds.apply(lambda x: x['sentence'].lower(), new_field_name='sentence')
    ds.apply(lambda x: x['sentence'].strip().split(), new_field_name='sentence')
    ds.apply(lambda x: len(x['sentence']) * [1.], new_field_name='mask', is_input=True)
    ds.apply(lambda x: int(x['label']), new_field_name='label', is_target=True)

    if vocabs is None:
        vocab = Vocabulary(max_size=30000, min_freq=2, unknown='<unk>', padding='<pad>')
        ds.apply(lambda x: [vocab.add(word) for word in x['sentence']])
        vocab.build_vocab()
    else:
        vocab = vocabs

    ds.apply(lambda x: [vocab.to_index(w) for w in x['sentence']], new_field_name='data', is_input=True)

    if not load_glove:
        print(f"successful load dataset from {path}")
        return ds

    embedding, _ = EmbedLoader().load_embedding(300, glove_path, 'glove', vocab)

    print(f"successful load dataset and embedding from {path}")

    return ds, embedding, vocab

def load_conll_with_glove(
        data_dir,
        data_path='train.pos',
        glove_path="",
        # glove_path='/remote-home/ygxu/dataset/glove.empty.txt',
        load_glove=True,
        vocabs=None
):
    path = os.path.join(data_dir, data_path)
    print(f"start load dataset from {path}.")

    from dataset import MyConllLoader
    ds = MyConllLoader().load(path)
    print(ds)
    ds.rename_field('word_seq', 'sentence')
    ds.rename_field('label_seq', 'label')
    #ds = DataSet.read_pos(path, headers=('sentence', 'label'), sep='\t')

    #ds.apply(lambda x: x['sentence'].lower(), new_field_name='sentence')
    #ds.apply(lambda x: x['sentence'].strip().split(), new_field_name='sentence')
    ds.apply(lambda x: len(x['sentence']) * [1.], new_field_name='word_seq_origin_len', is_input=True)

    if vocabs is None:
        vocab = Vocabulary(max_size=30000, min_freq=2, unknown='<unk>', padding='<pad>')
        ds.apply(lambda x: [vocab.add(word) for word in x['sentence']])
        vocab.build_vocab()
        vocab_label = Vocabulary(max_size=200, unknown=None, padding='<pad>')
        ds.apply(lambda x: [vocab_label.add(label) for label in x['label']])
        vocab_label.build_vocab()
    else:
        vocab, vocab_label = vocabs

    ds.apply(lambda x: [vocab.to_index(w) for w in x['sentence']], new_field_name='word_seq', is_input=True)
    ds.apply(lambda x: [vocab_label.to_index(w) for w in x['label']], new_field_name='truth', is_input=True, is_target=True)

    if not load_glove:
        print(f"successful load dataset from {path}")
        return ds

    embedding, _ = EmbedLoader().load_embedding(300, glove_path, 'glove', vocab)

    print(f"successful load dataset and embedding from {path}")

    return ds, embedding, (vocab, vocab_label)




