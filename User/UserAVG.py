import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from User.User import Userbase
from utils.model_utils import LargeMargin
from Server.sgd import SGD
import copy


# Implementation for FedAvg clients


class UserAVG(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, localstep):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs, localstep)

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.local_server_model = copy.deepcopy(list(self.model.parameters()))
        self.dataiter = iter(self.trainloader)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def record_with_grad(self, step, grad_list):
        if step:
            self.optimizer.step()
        else:
            batch_grad = None
            for layer in self.model.parameters():
                batch_grad = layer.grad.view(-1) if batch_grad is None else torch.cat((batch_grad, layer.grad.view(-1)))
            total = batch_grad.shape[0]
            grad_list.append(batch_grad)
            # if len(grad_list) == 0:
            #     for cnt in range(0, total):
            #         grad_list.append(None)
            # for e in range(0, batch_grad.shape[0]):
            #     grad_list[e] = batch_grad[e].unsqueeze(0) if grad_list[e] is None else torch.cat(
            #         (grad_list[e], batch_grad[e].unsqueeze(0)))

    def compute_with_grad(self, step, grad_list):
        # if not step:
        #     result_tensor = None
        #     for tensor in grad_list:
        #         var_res = torch.var(tensor)
        #         result_tensor = var_res.unsqueeze(0) if result_tensor is None else torch.cat(
        #             (result_tensor, var_res.unsqueeze(0)))
        #     final_norm = torch.norm(result_tensor).cpu()
        #     print(result_tensor)
        #     print(final_norm)
        #     return final_norm
        avg_tensor = torch.stack(grad_list, dim=0)
        final_tensor = None
        total = 0
        count = 0
        for origin_tensor in grad_list:
            sub_tensor = avg_tensor - origin_tensor
            total += torch.norm(sub_tensor).item()
            count += 1
        return total / count

    def estimate(self, epochs):
        LOSS = 0

        batch_LargeMargin = []
        self.model.train()
        self.model.to(self.device)
        i = 0
        init_grad = []
        step = False

        for inputs, targets in self.trainloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            # LOSS+= loss.item()
            self.record_with_grad(step, init_grad)
        return self.compute_with_grad(step, init_grad)

    def train(self, epochs):
        LOSS = 0

        batch_LargeMargin = []
        self.model.train()
        self.model.to(self.device)
        i = 0
        init_grad = []
        step = False
        while True:
            # 不做梯度下降
            # 每一个参数的梯度记录
            # 一次epoch计算方差

            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                # LOSS+= loss.item()
                #self.record_with_grad(step, init_grad, i)
                self.optimizer.step()
                i += 1
                if i == self.localstep:
                    break
            if i == self.localstep:
                break
        #self.compute_with_grad(step, init_grad)
        '''
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batchidx,(inputs,targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss=self.loss(outputs, targets)
                loss.backward()

                #LOSS+= loss.item()
                self.optimizer.step()

                # dis = LargeMargin(self.model, targets, outputs)
                # batch_size = len(dis)
                # batch_sum = torch.sum(dis, axis=0).item()
                #
                # batch_LargeMargin.append(batch_sum)

            '''
