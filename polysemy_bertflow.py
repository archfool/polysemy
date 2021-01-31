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
import pickle

from src_xml.utils import AttrDict
from src_xml.data import dictionary
from src_xml.model.transformer import TransformerModel
# from PolysemyBertFlowModel import PolysemyBertFlowModel
import PolysemyBertFlowModel

from init_config import *
from util_tools import print_fun_time, logger

import framework


def get_params(model_init_path=None):
    params = {
        'model_name_prefix': 'polysemy_bertflow',
        'lang_set': ['en', 'fr'],
        'task_name': ['en-en'],
        "dim_embd": 1024,
        'layer_num': 2,
        'max_seq_len': 256,
        'model_save_dir': data_path,
        'model_init_path': model_init_path,
    }
    if params['model_init_path'] is None:
        params['epoch_num'] = 0
    else:
        params['epoch_num'] = int(params['model_init_path'].split("_")[-1])
    return params


def load_model(params):
    model = PolysemyBertFlowModel.PolysemyBertFlowModel(params)
    if params['model_init_path'] is not None:
        model_params = torch.load(params['model_init_path'], map_location='cpu')
        framework.load_model_params(model=model, model_params_from_file=model_params, frozen=None)
    model = framework.check_gpu(model)
    return model


class generate_data():
    def __init__(self, params, batch_size):
        split_per = 0.95
        self.batch_size = batch_size
        self.lang_set = params['lang_set']
        self.token_embd_list = []
        with open(corpus_embd_file_path_format.format('_'.join(self.lang_set)), "rb") as f:
            while True:
                try:
                    self.token_embd_list.append(pickle.load(f))
                except:
                    break
        self.corpus_df = pd.read_csv(corpus_csv_path, sep='\001', encoding='utf-8')
        df_filter = pd.Series([True] * len(self.corpus_df))
        for task in params['task_name']:
            df_filter = df_filter & (self.corpus_df['task_type'] == task)
        self.corpus_df = self.corpus_df[df_filter]
        self.corpus_label = self.corpus_df[self.corpus_df['data_set'] != 'test']
        self.corpus_unlabel = self.corpus_df[self.corpus_df['data_set'] == 'test']
        self.corpus_df_train = self.corpus_label[:int(len(self.corpus_label) * split_per)]
        self.corpus_df_dev = self.corpus_label[-int(len(self.corpus_label) * (1 - split_per)):]
        self.corpus_df_test = self.corpus_unlabel

        logger.info("load {} data of {}".format(len(self.token_embd_list), "/".join(self.lang_set)))

    def __call__(self, data_set):
        if data_set == "train":
            df = self.corpus_df_train
        elif data_set == "eval":
            df = self.corpus_df_dev
        else:
            df = pd.DataFrame()

        for i in range(math.ceil(len(df) / self.batch_size)):
            df_batch = df[:self.batch_size]
            df = df[self.batch_size:]
            token_embd_batch = np.concatenate([self.token_embd_list[idx] for idx in df_batch.index], axis=0)
            token_embd_batch = framework.check_gpu(torch.tensor(token_embd_batch, dtype=torch.float32))
            keyword_idx = [[[int(xx) for xx in x.split(' ')], [int(yy) for yy in y.split(' ')]]
                           for x, y in
                           zip(df_batch['sent1_token_keyword_idx'], df_batch['sent2_token_keyword_idx'])]
            label = np.array(df_batch['label']).astype(np.int)
            label = framework.check_gpu(torch.tensor(label, dtype=torch.long))
            yield (token_embd_batch, keyword_idx), label


if __name__ == '__main__':
    batch_size = 32
    params = get_params()
    # params = get_params(model_init_path = os.path.join(data_path, "polysemy_bertflow_en-en_10"))
    model = load_model(params)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    # USE_CUDA = True if torch.cuda.is_available() else False
    data_stream = generate_data(params=params, batch_size=batch_size)

    polysemt_bertflow_model = framework.framework(params=params, model=model, data_stream=data_stream,
                                                  optimizer=optimizer,
                                                  loss_func=loss_func)
    # polysemt_bertflow_model.run_eval()
    polysemt_bertflow_model.run_train(epochs=30)

    torch.cuda.empty_cache()
    logger.info('END')
