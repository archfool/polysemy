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

    def __init__(self, params=None, model=None, dataset=None, optimizer=None, loss_func=None, USE_CUDA=True):
        self.params = params
        self.model = model.cuda() if USE_CUDA else model
        self.dataset = dataset
        self.optimizer = self.build_optimizer() if optimizer is None else optimizer
        self.loss_func = self.build_loss_func() if loss_func is None else loss_func
        self.eval_func = None
        self.USE_CUDA = USE_CUDA
        self.iter_count = 0

        # todo
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
            # 从dataloader 中拿数据
            iter_i = 0
            for iter_i, data_train in enumerate(self.dataset, self.iter_count + 1):
                self.run_train_batch(data_train)
            self.iter_count += iter_i

            # eval phrase
            # self.model.eval()
            # self.model.run_eval(data_eval)

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
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
            # x.grad += dloss/dx
            loss.backward()
            # todo
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
            #                                self.params["grad_clip"])
            return loss

        # loss 返回,准备优化
        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        self.optimizer.zero_grad()
        # optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs
        # x += -lr * x.grad
        self.optimizer.step(closure)

    def run_eval(self):
        for idx, (word_ids, lengths, langs) in enumerate(self.dataset()):
            tensor = self.model('polysemy', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
            if idx % 100 == 0:
                print(idx, tensor.size())

    def run_infer(self, data):
        pass


class yield_test():
    def __call__(delf):
        x = 10
        for i in range(x, 100):
            yield i


if __name__ == '__main__':
    a = yield_test()
    print(a())
    print("END")

# b = torch.from_numpy(a)
# feature_tensor_single.detach().numpy()
# 模型中可学习的参数会由net.parameters()返回。
# params = list(net.parameters())
