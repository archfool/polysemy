import os
import sys
import collections
import torch
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import math

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

from init_path_config import *
from util_tools import print_fun_time, check_gpu

if root_path.startswith("/media"):
    batch_size = 32
else:
    batch_size = 8


@print_fun_time
def load_model_torch(model_path):
    # Reload a pretrained model
    if torch.cuda.is_available():
        print("torch.cuda.is_available={}".format(torch.cuda.is_available()))
        reloaded = torch.load(model_path)
    else:
        print("torch.cuda.is_available={}".format(torch.cuda.is_available()))
        reloaded = torch.load(model_path, map_location=torch.device('cpu'))

    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    if torch.cuda.is_available():
        print("construct model in GPU")
        model = model.cuda()

    model.eval()
    model.load_state_dict(reloaded['model'])

    return model, params, dico


# @print_fun_time
def generate_model_input(sentences, key_word_idxs, params, dico):
    # check how many tokens are OOV
    # n_w = len([w for w in ' '.join(sentences).split()])
    # n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])
    # print('Number of out-of-vocab words: %s/%s' % (n_oov, n_w))

    # todo 关键词位置大于128的，截断

    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]

    # Create batch
    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])
    # print("max seq len: {}".format(slen))

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])

    # NOTE: No more language id (removed it in a later version)
    # langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs) if params.n_langs > 1 else None
    langs = None

    return word_ids, lengths, langs


# @print_fun_time
def infer_one_batch(tf_file_writer, one_batch_input, para_input, model_xml):
    corpus_df = one_batch_input[0]
    # sentences, key_word_idxs = one_batch_input
    params, dico = para_input

    sent1_bpe = corpus_df['sent1_bpe'].to_list()
    sent2_bpe = corpus_df['sent2_bpe'].to_list()
    sentences = []
    for s_1, s_2 in zip(sent1_bpe, sent2_bpe):
        sentences.append(s_1)
        sentences.append(s_2)

    key_word_idxs_1 = corpus_df['sent1_bpe_keyword_idx'].apply(lambda x: x.split(' ')).to_list()
    key_word_idxs_2 = corpus_df['sent2_bpe_keyword_idx'].apply(lambda x: x.split(' ')).to_list()
    key_word_idxs = []
    for idx_1, idx_2 in zip(key_word_idxs_1, key_word_idxs_2):
        key_word_idxs.append([int(idx) for idx in idx_1])
        key_word_idxs.append([int(idx) for idx in idx_2])

    word_ids, lengths, langs = generate_model_input(sentences, key_word_idxs, params, dico)

    # Forward
    tensor = model_xml('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
    # print(tensor.device)
    # print(torch.cuda.device_count())

    feature_tensor_batch = torch.tensor([])
    feature_tensor_batch = check_gpu(feature_tensor_batch)
    for i, key_word in enumerate(key_word_idxs):
        tensor_single = tensor[:, i, :]
        index = torch.tensor(key_word).unsqueeze(1).expand([-1, tensor_single.size()[1]])
        index = check_gpu(index)
        if torch.cuda.is_available():
            feature_tensor_batch = feature_tensor_batch.cuda()
        key_word_tensor = torch.gather(tensor_single, dim=0,
                                       index=index)
        key_word_tensor_max_pooling = torch.max(key_word_tensor, dim=0).values
        key_word_tensor_avg_pooling = torch.mean(key_word_tensor, dim=0)
        sent_tensor = tensor_single[0, :]
        feature_tensor_single = torch.cat((key_word_tensor_max_pooling,
                                           key_word_tensor_avg_pooling,
                                           sent_tensor),
                                          dim=0)
        if i % 2 == 0:
            sent1_feature_tensor_single = feature_tensor_single
        else:
            sent_couple_feature_tensor_single = torch.cat(
                (sent1_feature_tensor_single, feature_tensor_single),
                dim=0).reshape([1, -1])
            # a = sent_couple_feature_tensor_single.detach().numpy()
            # features = collections.OrderedDict()
            # features["input_ids"] = create_float_feature(feature.input_ids)
            # features["input_ids"] = create_float_feature(feature.input_ids)
            # features["input_ids"] = create_float_feature(feature.input_ids)
            # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            # tf_file_writer.write(tf_example.SerializeToString())
            feature_tensor_batch = torch.cat((feature_tensor_batch, sent_couple_feature_tensor_single), dim=0)
        # np.savetxt(os.path.join(data_path, 'test.txt'), feature_tensor_single.detach().numpy())
        # np.loadtxt(os.path.join(data_path, 'test.txt'))

    return feature_tensor_batch


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


@print_fun_time
def infer_all(batch_input, para_input, model_xml, batch_size=10):
    n_batch = len(batch_input[0])
    print("data num : {}".format(n_batch))
    tensor = torch.tensor([])
    tensor = check_gpu(tensor)

    tf_file_writer = tf.python_io.TFRecordWriter(os.path.join(data_path, '123'))

    for i in range(math.ceil(n_batch / batch_size)):
        count_batch = i + 1
        one_batch_input = [x[:batch_size] for x in batch_input]
        batch_input = [x[batch_size:] for x in batch_input]
        tensor_one_batch = infer_one_batch(tf_file_writer, one_batch_input, para_input, model_xml)
        # tensor = tensor_one_batch
        # tensor = torch.cat((tensor, tensor_one_batch), dim=0)
        if count_batch % 100 == 0:
            print("data count: {}: {}".format(batch_size * count_batch,
                                              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    tf_file_writer.close()
    return tensor_one_batch


# 加载模型并预测
@print_fun_time
def infer(model_path, corpus_df):
    model, params, dico = load_model_torch(model_path)

    # batch_input = [sentences, key_word_idxs]
    batch_input = [corpus_df]
    para_input = [params, dico]

    tensor = infer_all(batch_input, para_input, model, batch_size=batch_size)

    print(tensor.size())
    return tensor


if __name__ == '__main__':
    # corpus_bpe_path = os.path.join(data_path, 'SemEval2021_Task2_corpus.txt')
    # with open(corpus_bpe_path, 'r', encoding='utf-8') as f:
    #     sentences = f.readlines()
    corpus_df = pd.read_csv(os.path.join(data_path, 'SemEval2021_Task2_corpus.csv'), sep='\001', encoding='utf-8')

    # key_word_idxs_1 = corpus_df['sent1_bpe_keyword_idx'].apply(lambda x: x.split(' ')).to_list()
    # key_word_idxs_2 = corpus_df['sent2_bpe_keyword_idx'].apply(lambda x: x.split(' ')).to_list()
    # key_word_idxs = []
    # for idx_1, idx_2 in zip(key_word_idxs_1, key_word_idxs_2):
    #     key_word_idxs.append([int(idx) for idx in idx_1])
    #     key_word_idxs.append([int(idx) for idx in idx_2])
    model_path = os.path.join(model_xml_path, u'mlm_17_1280.pth')
    # sentences = sentences[::-1]
    # key_word_idxs = key_word_idxs[::-1]
    tensor = infer(model_path, corpus_df)
    # tensor = infer(model_path, sentences, key_word_idxs)
    print(tensor.size())

    print('END')
