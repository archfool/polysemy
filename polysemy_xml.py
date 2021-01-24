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

from src.utils import AttrDict
from src.data import dictionary
from src.model.transformer import TransformerModel

from init_config import *
from framework_tools import print_fun_time, check_gpu

import framework

def reload_model(model_path):
    # Reload a pretrained model
    reloaded = torch.load(model_path, map_location=torch.device('cpu'))
    # todo
    # model.load_state_dict(torch.load(
    #     params["reload_model"],
    #     map_location=lambda storage, loc: storage),
    #     False)
    return reloaded


def load_model_xml(reloaded):
    # extract para
    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

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
    model = check_gpu(model)

    # model.train_batch()
    # model.eval()
    model.load_state_dict(reloaded['model'])
    # todo
    # model.load_state_dict(torch.load(
    #     params["reload_model"],
    #     map_location=lambda storage, loc: storage),
    #     False)

    return model, params


class data_stream():
    def __init__(self):
        self.corpus_df = pd.read_csv(os.path.join(data_path, 'SemEval2021_Task2_corpus.csv'), sep='\001', encoding='utf-8')
        self.n_corpus = len(self.corpus[0])
        print("data num : {}".format(self.n_corpus))

    def __ceil__(self):
        pass


if __name__=='__main__':
    model_path = os.path.join(model_xml_path, u'mlm_17_1280.pth')

    reloaded = reload_model(model_path)

    model, params = load_model_xml(reloaded)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    USE_CUDA = True if torch.cuda.is_available() else False
    polysemt_xml_,odel = framework(params=params, model=model, dataset=None, optimizer=optimizer, loss_func=loss_func, USE_CUDA=USE_CUDA)

    print('END')