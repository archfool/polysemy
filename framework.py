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

        # total_params = sum(p.numel() for p in self.model.parameters())/pow(2,20)
        total_storage_space = sum(p.numel()*int(str(p.dtype)[-2:])/8 for p in self.model.parameters())/pow(2,20)
        print("Total Model Storage Space: {:.0f} MB".format(total_storage_space))

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
            for iter_i, data_train in enumerate(self.data_stream(self.params, data_set='train')(), self.iter_count + 1):
                self.run_train_batch(data_train)
            self.iter_count += iter_i

            # eval phrase
            self.model.eval()
            # self.model.run_eval()

    def run_train_batch(self, data):
        input_batch, target_batch = data
        input_var = input_batch
        target_var = target_batch
        # if self.USE_CUDA:
        #     input_var = input_var.cuda()
        #     target_var = target_var.cuda()

        # 每一次前馈就是一次函数闭包操作
        def closure():
            output_batch = self.model('polysemy_predict', x=input_batch)
            loss = self.loss_func(output_batch, target_batch)
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True. x.grad += dloss/dx
            loss.backward()
            # 梯度裁剪 max_norm = 1~10
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            print("loss: {}".format(loss))
            return loss

        # optimizer.zero_grad() clears x.grad for every parameter x in the optimizer.
        self.optimizer.zero_grad()
        # optimizer.step updates the value of x using the gradient x.grad. For example of SGD : x += -lr * x.grad
        self.optimizer.step(closure)

    def run_eval(self):
        for idx, data_batch in enumerate(self.data_stream(self.params, data_set='eval')()):
            input_batch, target_batch = data_batch
            # todo replace 'polysemy' with self.params['sub_model_name']['eval']
            tensor = self.model('polysemy_predict', x=input_batch).contiguous()

    def run_infer(self, data):
        pass


def load_model_params(model, model_params_from_file, frozen=None):
    model_params = {}
    for para_name, para_value in model.named_parameters():
        if para_name in model_params_from_file:
            param_tmp = model_params_from_file[para_name]
            if frozen is not None:
                param_tmp.requires_grad = not frozen
            model_params[para_name] = param_tmp
            print("[{}]{}{}[{}] **INIT FROM SAVED MODEL FILE**".format(
                'Not Frozon' if param_tmp.requires_grad else 'Frozon',
                para_name,
                list(para_value.size()),
                str(para_value.dtype).split(".")[-1],
            ))
        else:
            param_tmp = para_value
            print("[{}]{}{}[{}]".format(
                'Not Frozon' if param_tmp.requires_grad else 'Frozon',
                para_name,
                list(para_value.size()),
                str(para_value.dtype).split(".")[-1],
            ))
    model.load_state_dict(model_params, strict=False)


if torch.cuda.is_available():
    print("USE GPU")
    GPU_OR_CPU = 'cuda'
else:
    print("USE CPU")
    GPU_OR_CPU = 'cpu'


if __name__ == '__main__':
    print("END")

# b = torch.from_numpy(a)
# feature_tensor_single.detach().numpy()
# 模型中可学习的参数会由net.parameters()返回。
# params = list(net.parameters())
