# -*- coding: utf-8 -*-
"""
@author: archfool
Created on 2021/1/31 上午9:49
"""

from logging import getLogger
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework import check_gpu


class PolysemyBertFlowModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim_embd = getattr(params, 'dim_embd', 1024)
        self.dim_head = getattr(params, 'dim_head', 64)
        self.dim_head_scale = math.sqrt(self.dim_head)
        self.layer_num = getattr(params, 'layer_num', 2)
        self.max_seq_len = getattr(params, 'max_seq_len', 256)

        self.inter_matrix = nn.ModuleList()
        for _ in range(4):
            self.inter_matrix.append(nn.Linear(self.dim_head, self.dim_head))
        # self.fc = nn.ModuleList([nn.Linear(4+1024*4, 1024), nn.Linear(1024, 2)])
        self.fc = nn.ModuleList([nn.Linear(516, 64), nn.Linear(64, 2)])
        # for fc in self.fc:
        #     torch.nn.init.constant_(fc.bias, 0.001)

        self.q_matrix_att = nn.Linear(self.dim_embd, self.dim_head)
        self.k_matrix_att = nn.Linear(self.dim_embd, self.dim_head)
        self.v_matrix_att = nn.Linear(self.dim_embd, self.dim_head)
        self.matrix_inter = nn.Linear(self.dim_embd, self.dim_head)

        self.inter_matrix = check_gpu(self.inter_matrix)

    def forward(self, mode, **kwargs):
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.fwd(**kwargs)
            # return self.predict(**kwargs)
        elif mode == 'infer':
            return self.fwd(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, input, langs=None, cache=None):
        token_embd_input, keyword_idx = input
        token_embd_input = check_gpu(token_embd_input)  # [batch_size, sent_id, layer_num, seq_num, dim_embd]
        # x = check_gpu(torch.from_numpy(x).type(torch.float32))  # [batch_size, sent_id, layer_num, seq_num, dim_embd]
        token_embd_input = token_embd_input.reshape([-1, 2, self.layer_num, self.max_seq_len, self.dim_embd])
        batch_size = token_embd_input.size()[0]
        sent_cls = token_embd_input[:, :, :, 0, :]
        keyword_maxpooling = torch.empty((batch_size, 2, self.layer_num, self.dim_embd), dtype=torch.float32, device='cuda')
        keyword_avgpooling = torch.empty((batch_size, 2, self.layer_num, self.dim_embd), dtype=torch.float32, device='cuda')
        attention_feature_list = []
        for batch_i in range(batch_size):
            tmp_keyword_embd_list = [[],[]]
            tmp_keyword_embd_q = [None, None]
            tmp_keyword_embd_k = [None, None]
            tmp_keyword_embd_v = [None, None]
            for sent_id in range(2):
                for layer_i in range(self.layer_num):
                    # extract keyword embd
                    tmp_sent_embd = token_embd_input[batch_i, sent_id, layer_i, :, :]
                    tmp_keyword_idx = keyword_idx[batch_i][sent_id - 1]
                    tmp_idx_gather_tensor = check_gpu(torch.tensor(tmp_keyword_idx, dtype=torch.long)) \
                        .unsqueeze(1).expand([-1, self.dim_embd])
                    tmp_keyword_embd = torch.gather(tmp_sent_embd, dim=0, index=tmp_idx_gather_tensor)
                    # cal pooling
                    keyword_maxpooling[batch_i, sent_id, layer_i, :] = torch.max(tmp_keyword_embd, dim=0).values
                    keyword_avgpooling[batch_i, sent_id, layer_i, :] = torch.mean(tmp_keyword_embd, dim=0)
                    # add to tmp_keyword_embd list
                    tmp_keyword_embd_list[sent_id].append(tmp_keyword_embd)
                tmp_keyword_embd_list[sent_id] = torch.cat(tmp_keyword_embd_list[sent_id], dim=0)
                tmp_keyword_embd_q[sent_id] = self.q_matrix_att(tmp_keyword_embd_list[sent_id])
                tmp_keyword_embd_k[sent_id] = self.k_matrix_att(tmp_keyword_embd_list[sent_id])
                tmp_keyword_embd_v[sent_id] = self.v_matrix_att(tmp_keyword_embd_list[sent_id])
            sent_1_attention_weight = torch.matmul(tmp_keyword_embd_q[0], tmp_keyword_embd_k[1].transpose(0,1))\
                          /self.dim_head_scale
            sent_1_attention_embd = torch.matmul(sent_1_attention_weight, tmp_keyword_embd_v[1])
            sent_1_attention_maxpooling = torch.max(sent_1_attention_embd, dim=0).values
            sent_1_attention_avgpooling = torch.mean(sent_1_attention_embd, dim=0)
            sent_2_attention_weight = torch.matmul(tmp_keyword_embd_q[1], tmp_keyword_embd_k[0].transpose(0,1))\
                          /self.dim_head_scale
            sent_2_attention_embd = torch.matmul(sent_2_attention_weight, tmp_keyword_embd_v[0])
            sent_2_attention_maxpooling = torch.max(sent_2_attention_embd, dim=0).values
            sent_2_attention_avgpooling = torch.mean(sent_2_attention_embd, dim=0)
            attention_feature = torch.cat([
                sent_1_attention_maxpooling,
                sent_1_attention_avgpooling,
                sent_2_attention_maxpooling,
                sent_2_attention_avgpooling
            ], dim=0).unsqueeze(0)
            attention_feature_list.append(attention_feature)
        sent_1_keyword_maxpooling = self.matrix_inter(torch.max(keyword_maxpooling[:, 0, :, :], dim=1).values)
        sent_2_keyword_maxpooling = self.matrix_inter(torch.max(keyword_maxpooling[:, 1, :, :], dim=1).values)
        sent_1_keyword_avgpooling = self.matrix_inter(torch.mean(keyword_avgpooling[:, 0, :, :], dim=1))
        sent_2_keyword_avgpooling = self.matrix_inter(torch.mean(keyword_avgpooling[:, 1, :, :], dim=1))
        inter_pooling_cosine = torch.cat([
            torch.cosine_similarity(self.inter_matrix[0](sent_1_keyword_maxpooling), sent_2_keyword_maxpooling).unsqueeze(1),
            torch.cosine_similarity(self.inter_matrix[1](sent_1_keyword_maxpooling), sent_2_keyword_avgpooling).unsqueeze(1),
            torch.cosine_similarity(self.inter_matrix[2](sent_1_keyword_avgpooling), sent_2_keyword_maxpooling).unsqueeze(1),
            torch.cosine_similarity(self.inter_matrix[3](sent_1_keyword_avgpooling), sent_2_keyword_avgpooling).unsqueeze(1),
        ], dim=1)
        feature_list = [
            torch.cat(attention_feature_list, dim=0),
            inter_pooling_cosine,
            sent_1_keyword_maxpooling,
            sent_1_keyword_avgpooling,
            sent_2_keyword_maxpooling,
            sent_2_keyword_avgpooling,
        ]
        feature = torch.cat(feature_list, dim=1)
        # feature = inter_pooling_cosine
        result = self.fc[0](feature)
        result = gelu(result)
        result = self.fc[1](result)
        result = gelu(result)
        result = F.softmax(result, dim=-1)
        # print(result)
        return result

    def predict(self, tensor, pred_mask, y, get_scores):
        pass

    def infer(self, tensor, pred_mask, y, get_scores):
        pass


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


if __name__ == "__main__":
    model_tmp = PolysemyBertFlowModel()
    a = model_tmp.forward("fwd", x=np.random.random((1024 * 2, 1024)).astype('float16'),
                          max_seq_len=256, dim_embd=1024, layer_num=2, keyword_idx=[[[0, 2], [1, 3]], [[0, 2], [1, 3]]])
    print("END")
