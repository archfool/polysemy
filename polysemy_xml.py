# -*- coding: utf-8 -*-
"""
@author: archfool
Created on 2021/1/24 下午4:47
"""
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import time
import torch

from src_xml.utils import AttrDict
from src_xml.data import dictionary
from src_xml.model.transformer import TransformerModel

from init_config import *
from util_tools import print_fun_time, logger

import framework

if root_path.startswith("/media"):
    batch_size = 16
else:
    batch_size = 8


def reload_model(model_path):
    # Reload a pretrained model
    reloaded = torch.load(model_path, map_location=torch.device('cpu'))
    # model.load_state_dict(torch.load(
    #     params["reload_model"],
    #     map_location=lambda storage, loc: storage),
    #     False)
    return reloaded


def load_model_xml(reloaded):
    # extract para
    params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    # build dictionary
    dico = dictionary.Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params['dico'] = dico

    # update parameters
    params.n_words = len(dico)
    params.bos_index = dico.index(dictionary.BOS_WORD)
    params.eos_index = dico.index(dictionary.EOS_WORD)
    params.pad_index = dico.index(dictionary.PAD_WORD)
    params.unk_index = dico.index(dictionary.UNK_WORD)
    params.mask_index = dico.index(dictionary.MASK_WORD)

    # build model / reload weights
    model = TransformerModel(params, params['dico'], True, True)
    model = framework.check_gpu(model)

    framework.load_model_params(model=model, model_params_from_file=reloaded['model'], frozen=True)
    # model.load_state_dict(torch.load(
    #     params["reload_model"],
    #     map_location=lambda storage, loc: storage),
    #     False)

    return model, params


class data_stream():
    def __init__(self, params, data_set):
        self.corpus_df = pd.read_csv(os.path.join(data_path, 'SemEval2021_Task2_corpus.csv'), sep='\001',
                                     encoding='utf-8')
        self.data_set = data_set
        if self.data_set == 'train':
            self.corpus_df = self.corpus_df[
                (self.corpus_df['data_set'] == 'training') | (self.corpus_df['data_set'] == 'dev')]
        elif self.data_set == 'eval':
            self.corpus_df = self.corpus_df[self.corpus_df['data_set'] == 'trial']
        elif self.data_set == 'predict':
            self.corpus_df = self.corpus_df[self.corpus_df['data_set'] == 'test']
        else:
            logger.info('Invalid data_set: {}'.format(data_set))

        # do shuffle
        if self.data_set == 'train':
            self.corpus_df = self.corpus_df.sample(frac=1).reset_index(drop=True)

        self.params = params
        self.batch_size = batch_size

        self.n_corpus = len(self.corpus_df)
        logger.info("data num : {}".format(self.n_corpus))

    def __call__(self):
        for i in range(math.ceil(self.n_corpus / self.batch_size)):
            c_batch = i + 1
            corpus_df_batch = self.corpus_df[:self.batch_size]
            self.corpus_df = self.corpus_df[self.batch_size:]

            label = corpus_df_batch['label'].tolist()
            try:
                label = [int(x) for x in label]
                label = torch.tensor(label, dtype=torch.long)
            except:
                pass
            label = framework.check_gpu(label)

            sent1_bpe = corpus_df_batch['sent1_bpe'].to_list()
            sent2_bpe = corpus_df_batch['sent2_bpe'].to_list()

            lang1 = corpus_df_batch['sent1_lang'].to_list()
            lang2 = corpus_df_batch['sent2_lang'].to_list()

            key_word_idxs_1 = corpus_df_batch['sent1_bpe_keyword_idx'].apply(lambda x: x.split(' ')).to_list()
            key_word_idxs_1 = [[int(idx) for idx in idx_list] for idx_list in key_word_idxs_1]
            key_word_idxs_2 = corpus_df_batch['sent2_bpe_keyword_idx'].apply(lambda x: x.split(' ')).to_list()
            key_word_idxs_2 = [[int(idx) for idx in idx_list] for idx_list in key_word_idxs_2]
            word_ids_1, lengths_1, langs_1, positions_1 = generate_model_input(sent1_bpe, lang1, params, params['dico'])
            word_ids_2, lengths_2, langs_2, positions_2 = generate_model_input(sent2_bpe, lang2, params, params['dico'])
            # if c_batch % 100 == 0:
            #     logger.info("corpus: {}: {}".format(
            #         self.batch_size * c_batch,
            #         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            yield (
                      (word_ids_1, lengths_1, langs_1, positions_1, key_word_idxs_1),
                      (word_ids_2, lengths_2, langs_2, positions_2, key_word_idxs_2)
                  ), label


def generate_model_input(sentences, lang, params, dico):
    # check how many tokens are OOV
    # n_w = len([w for w in ' '.join(sentences).split()])
    # n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])
    # logger.info('Number of out-of-vocab words: %s/%s' % (n_oov, n_w))

    # todo 关键词位置大于128的，截断

    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]

    # Create batch
    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])
    # logger.info("max seq len: {}".format(slen))

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent
    word_ids = framework.check_gpu(word_ids)

    lengths = torch.LongTensor([len(sent) for sent in sentences])
    lengths = framework.check_gpu(lengths)

    # NOTE: No more language id (removed it in a later version)
    # langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs)
    # langs = None
    lang = [params.lang2id[l] for l in lang]
    langs = torch.tensor(lang, dtype=torch.long).unsqueeze(0).expand([slen, bs])
    langs = framework.check_gpu(langs)

    positions = torch.arange(slen)[:, None].repeat(1, bs)
    positions = framework.check_gpu(positions)

    return word_ids, lengths, langs, positions


if __name__ == '__main__':
    if model_xml_path.endswith("mlm_17_1280"):
        model_path = os.path.join(model_xml_path, u'mlm_17_1280.pth')
    elif model_xml_path.endswith("mlm_tlm_xnli15_1024"):
        model_path = os.path.join(model_xml_path, u'mlm_tlm_xnli15_1024.pth')

    reloaded = reload_model(model_path)

    model, params = load_model_xml(reloaded)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    USE_CUDA = True if torch.cuda.is_available() else False

    polysemt_xml_model = framework.framework(params=params, model=model, data_stream=data_stream, optimizer=optimizer,
                                             loss_func=loss_func, USE_CUDA=USE_CUDA)
    polysemt_xml_model.run_eval()
    polysemt_xml_model.run_train(epochs=30)

    torch.cuda.empty_cache()
    logger.info('END')
