# -*- coding: utf-8 -*-
"""
Created on 2021/1/23 0023 下午 15:56
@author: ruanzhihao
"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from util_tools import logger


class framework():

    def __init__(self, params, model, data_stream,
                 optimizer=None, loss_func=None, eval_func=None, USE_CUDA=True):
        self.params = params
        # self.model = model.cuda() if USE_CUDA else model
        self.model = check_gpu(model)
        self.data_stream = data_stream
        self.optimizer = self.build_optimizer() if optimizer is None else optimizer
        self.loss_func = self.build_loss_func() if loss_func is None else loss_func
        self.eval_func = self.build_eval_func() if eval_func is None else eval_func
        # self.USE_CUDA = USE_CUDA
        self.iter_count = 0
        self.loss = 0
        self.score = 0
        # total_params = sum(p.numel() for p in self.model.parameters())/pow(2,20)
        total_storage_space = sum(p.numel() * int(str(p.dtype)[-2:]) / 8 for p in self.model.parameters()) / pow(2, 20)
        logger.info("Total Model Storage Space: {:.0f} MB".format(total_storage_space))

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

    def build_eval_func(self):
        eval_fun = metric_acc
        return eval_fun

    def run_train(self, epochs=1):
        # epoch
        for epoch_i in range(self.params['epoch_num'] + 1, self.params['epoch_num'] + epochs + 1):
            logger.info("epoch: {}".format(epoch_i))
            # train_batch phrase
            self.model.train()
            iter_i = 0
            for iter_i, data_train in enumerate(self.data_stream(data_set='train'), self.iter_count + 1):
                # data_train = [batch_size, (input_batch, target_batch)]
                self.run_train_batch(data_train)
                if iter_i % 100 == 0:
                    self.model.eval()
                    score_tmp = self.score
                    self.run_eval()
                    logger.info("step:{} loss: {:.4f} train_metric: {:.4f} dev_metric: {:.4f}"
                                .format(iter_i, self.loss, score_tmp, self.score))
            model_name = "_".join([self.params['model_name_prefix'],
                                   '_'.join(self.params['task_name']),
                                   str(epoch_i)
                                   ])
            self.save_model(model_name=model_name)
            self.iter_count = iter_i

            # # eval phrase
            # self.model.eval()
            # self.run_eval()

    def run_train_batch(self, data):
        input_batch, target_batch = data

        target_batch = check_gpu(target_batch)

        input_var = input_batch
        target_var = target_batch

        # if self.USE_CUDA:
        #     input_var = input_var.cuda()
        #     target_var = target_var.cuda()

        # 每一次前馈就是一次函数闭包操作
        def closure():
            # todo input mode name
            output_batch = self.model('fwd', input=input_batch)
            loss = self.loss_func(output_batch, target_batch)
            self.loss = loss
            self.score = self.eval_func(output=output_batch, label=target_batch)
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. x.grad += dloss/dx
            loss.backward()
            # 梯度裁剪 max_norm = 1~10
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            return loss

        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        self.optimizer.zero_grad()
        # optimizer.step updates the value of x using the gradient x.grad. For example of SGD : x += -lr * x.grad
        self.optimizer.step(closure)
        # loss.backward()和self.optimizer.step(closure)在原有模型空间的基础上，各增加50%的显存

    def run_eval(self):
        self.score = 0
        iter_i = 0
        for iter_i, data_batch in enumerate(self.data_stream(data_set='eval')):
            input_batch, target_batch = data_batch
            target_batch = check_gpu(target_batch)
            output_batch = self.model('predict', input=input_batch)
            score = self.eval_func(output=output_batch, label=target_batch)
            self.score += score
        self.score = self.score / (iter_i + 1)
        return self.score

    def run_infer(self, data):
        pass

    def save_model(self, model_name):
        torch.save(self.model.state_dict(), os.path.join(self.params['model_save_dir'], model_name))


def load_model_params(model, model_params_from_file, frozen=None):
    model_params = {}
    # todo the "FROZEN" tag is mistake
    for para_name, para_value in model.named_parameters():
        if para_name in model_params_from_file:
            param_tmp = model_params_from_file[para_name]
            if frozen is not None:
                param_tmp.requires_grad = not frozen
            model_params[para_name] = param_tmp
            logger.info("[{}]{}{}[{}] **INIT_FROM_FILE**".format(
                'Not Frozen' if param_tmp.requires_grad else 'Frozen',
                para_name,
                list(para_value.size()),
                str(para_value.dtype).split(".")[-1],
            ))
        else:
            param_tmp = para_value
            logger.info("[{}]{}{}[{}]".format(
                'Not Frozen' if param_tmp.requires_grad else 'Frozen',
                para_name,
                list(para_value.size()),
                str(para_value.dtype).split(".")[-1],
            ))
    model.load_state_dict(model_params, strict=False)


def metric_acc(output, label):
    prediction = torch.argmax(output, dim=1)
    correct_num = (prediction == label).sum().float()
    total_num = len(label)
    acc = correct_num / total_num
    return acc


def check_gpu(x):
    if x is None:
        return None
    elif torch.cuda.is_available():
        return x.cuda()
    else:
        return x


if torch.cuda.is_available():
    logger.info("USE GPU")
    GPU_OR_CPU = 'cuda'
else:
    logger.info("USE CPU")
    GPU_OR_CPU = 'cpu'

if __name__ == '__main__':
    logger.info("END")

# b = torch.from_numpy(a)
# feature_tensor_single.detach().numpy()
# 模型中可学习的参数会由net.parameters()返回。
# params = list(net.parameters())
