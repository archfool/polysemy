# -*- coding: utf-8 -*-
"""
Created on 2021/1/23 0023 下午 15:56
@author: ruanzhihao
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class framework():

    def __init__(self, params, model, data_stream,
                 optimizer=None, loss_func=None, eval_func=None, USE_CUDA=True):
        self.params = params
        self.model = model.cuda() if USE_CUDA else model
        self.data_stream = data_stream
        self.optimizer = self.build_optimizer() if optimizer is None else optimizer
        self.loss_func = self.build_loss_func() if loss_func is None else loss_func
        self.eval_func = eval_func
        self.USE_CUDA = USE_CUDA
        self.iter_count = 0

        # todo 加上每参数的大小权重信息
        total_params = sum(p.numel() for p in self.model.parameters())
        # print("%s" % model)
        print("Total Model Params:%s" % total_params)

    def __call__(self, *args, **kwargs):
        pass

    def build_optimizer(self):
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # optimizer = optim.Adam(self.model.parameters())
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.01)
        return optimizer

    def build_loss_func(self):
        # loss_func = nn.MSELoss()
        loss_func = nn.CrossEntropyLoss()
        return loss_func

    def run_train(self, epochs=1):
        # epoch
        for epoch_i in range(1, epochs + 1):
            # train_batch phrase
            self.model.train()
            iter_i = 0
            for iter_i, data_train in enumerate(self.data_stream(self.params)(), self.iter_count + 1):
                self.run_train_batch(data_train)
            self.iter_count += iter_i

            # eval phrase
            self.model.eval()
            self.model.run_eval()

    def run_train_batch(self, data):
        batch_input, batch_target = data
        input_var = batch_input
        target_var = batch_target
        if self.USE_CUDA:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # 每一次前馈就是一次函数闭包操作
        def closure():
            batch_output = self.model(input_var)
            loss = self.loss_func(batch_output, target_var)
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. x.grad += dloss/dx
            loss.backward()
            # 梯度裁剪 max_norm = 1~10
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            return loss

        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        self.optimizer.zero_grad()
        # optimizer.step updates the value of x using the gradient x.grad. For example of SGD : x += -lr * x.grad
        self.optimizer.step(closure)

    def run_eval(self):
        for idx, data_batch in enumerate(self.data_stream(self.params)()):
            tensor = self.model('polysemy', x=data_batch).contiguous()
            print(tensor.size())
            # tensor = self.model('polysemy', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
            # tensor_sent1 = self.model('polysemy', x=word_ids_1, lengths=lengths_1, langs=langs_1, causal=False) \
            #     .contiguous()
            # tensor_sent2 = self.model('polysemy', x=word_ids_2, lengths=lengths_2, langs=langs_2, causal=False) \
            #     .contiguous()

            # feature_tensor_batch = torch.tensor([]).cuda() if self.USE_CUDA else torch.tensor([])
            # for i, key_word in enumerate(key_word_idxs):
            #     tensor_single = tensor[:, i, :]
            #     index = torch.tensor(key_word, dtype=torch.long, device='cuda' if self.USE_CUDA else 'cpu') \
            #         .unsqueeze(1).expand([-1, tensor_single.size()[1]])
            #     key_word_tensor = torch.gather(tensor_single, dim=0,
            #                                    index=index)
            #     key_word_tensor_max_pooling = torch.max(key_word_tensor, dim=0).values
            #     key_word_tensor_avg_pooling = torch.mean(key_word_tensor, dim=0)
            #     sent_tensor = tensor_single[0, :]
            #     feature_tensor_single = torch.cat((key_word_tensor_max_pooling,
            #                                        key_word_tensor_avg_pooling,
            #                                        sent_tensor),
            #                                       dim=0)
            #     if i % 2 == 0:
            #         sent1_feature_tensor_single = feature_tensor_single
            #     else:
            #         sent_couple_feature_tensor_single = torch.cat(
            #             (sent1_feature_tensor_single, feature_tensor_single),
            #             dim=0).reshape([1, -1])
            #         feature_tensor_batch = torch.cat((feature_tensor_batch, sent_couple_feature_tensor_single),
            #                                          dim=0)

    def run_infer(self, data):
        pass


def load_model_params(model, model_params_from_file):
    model_params = {}
    # model_params_old = [x for x in model.named_parameters()]
    for para_name, para_value in model.named_parameters():
        if para_name in model_params_from_file:
            model_params[para_name] = model_params_from_file[para_name]
            print("{}: {}: {} **INIT FROM SAVED MODEL FILE**".format(para_name, para_value.size(), para_value.dtype))
        else:
            print("{}: {}: {}".format(para_name, para_value.size(), para_value.dtype))
    model.load_state_dict(model_params, strict=False)
    # model_params_new = [x for x in model.named_parameters()]


if __name__ == '__main__':
    print("END")

# b = torch.from_numpy(a)
# feature_tensor_single.detach().numpy()
# 模型中可学习的参数会由net.parameters()返回。
# params = list(net.parameters())
