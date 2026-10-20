import sys

import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader

from User.UserAVG import UserAVG

from Server.Server import Server
from Server.sgd import SGD
from utils.model_utils import read_data, read_user_data, LargeMargin
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy


# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, num_glob_iters, local_epochs, iid,
                 select_users, datadir, beta, partition, numusers, localstep):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate,
                         num_glob_iters, local_epochs, num_users=numusers, times=1, select_users_num=select_users,
                         datadir=datadir, beta=beta, partition=partition)

        # Initialize data for all  users
        # data = read_data(dataset,iid)
        total_users = self.num_users
        self.optimizer = SGD(self.model.parameters(), lr=1)
        self.criterion = nn.CrossEntropyLoss()
        self.local_server_model = copy.deepcopy(list(self.model.parameters()))
        self.iid = iid
        self.local_epochs = local_epochs
        self.total_train_samples = 0

        for i in range(0, total_users):
            id = i
            train = self.users_trainset[i]
            user = UserAVG(device, id, train, model, batch_size, learning_rate, local_epochs, localstep)
            self.users.append(user)
            self.total_train_samples += user.train_samples

    def get_grads_difference(self):
        grad = []
        for param_1, param_0 in zip(self.model.parameters(), self.local_server_model):
            param = param_0.data - param_1.data
            # param=param_0.data-param_0.data
            grad = param.data.view(-1) if not len(grad) else torch.cat((grad, param.view(-1)))
        return grad

    def Evaluate(self):
        global best_acc
        self.model.eval()
        # self.model.train()
        device = self.device
        net = self.model
        criterion = self.criterion
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                if self.dataset == "Cifar100":
                    targets = targets.to("cpu").numpy()
                    for i in range(100):
                        targets[i] = self.targetdic.get(targets[i])
                    targets = torch.from_numpy(targets).to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                # loss.backward()
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print("Average Global Accurancy: ", 100. * correct / total)
            print("Average Global Trainning Loss: ", test_loss / 100)

    def aggregate_grads_AVG(self):

        global_para = self.model.state_dict()
        idx = 0
        total_data_nums = 0
        for user in self.selected_users:
            total_data_nums += user.train_samples
        for user in self.selected_users:
            net_para = user.model.cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] / len(self.selected_users)
            else:
                for key in net_para:
                    global_para[key] += net_para[key] / len(self.selected_users)
            idx = idx + 1
        self.model.load_state_dict(global_para)
    def aggregate_grads_AVG_new(self):

        user_updates = []
        #print("collect benign users' state and turn to 1 dimension vector")
        for user in self.selected_users:
            param = user.get_params()
            global_param = self.get_params()
            param = global_param-param
            user_updates = param[None, :] if len(user_updates) == 0 else torch.cat((user_updates, param[None, :]),
                                                                               0)
        aggregated_update=torch.mean(user_updates,0)

        global_para=self.model.state_dict()
        start_idx = 0
        for key in global_para:
            global_para[key] = global_para[key]-aggregated_update[start_idx:start_idx + len(global_para[key].data.view(-1))].reshape(
                global_para[key].data.shape)
            start_idx = start_idx + len(global_para[key].data.view(-1))
        self.model.load_state_dict(global_para)

    def train(self):
        loss = []

        # 每一个suser的model每一层参数
        # server的model的每一层参数
        # 计算每一轮的所有server和suser的差值的平均值和最大值

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.model=self.model.to(self.device)
            # calculate global variance before sending parameter
            if glob_iter % 1 == 0 and glob_iter!=0:
                gv_avg, gv_max = self.global_v()
                self.gv_avg_list.append(gv_avg)
                self.gv_max_list.append(gv_max)
            self.send_parameters()
            # calculate local variance after sending parameter
            if glob_iter % 1 == 0 and glob_iter!=0:
                lv_total = 0
                for user in self.users:
                    avg_single_user = user.estimate(self.local_epochs)
                    lv_total += avg_single_user
                lv_avg = lv_total / len(self.users)
                self.lv_avg_list.append(lv_avg)
            if glob_iter %1==0 and glob_iter!=0:
                print('lv_avg: ' + str(lv_avg) + ', gv_avg: ' + str(gv_avg) + ', gv_max: ' + str(gv_max))
            '''
            if glob_iter % 10 == 0:
                for user in self.users:
                    user.estimate(self.local_epochs)
            '''


            # Evaluate model each interation
            self.evaluate()
            self.select_users()

            for user in self.selected_users:
                user.train(self.local_epochs)  # * user.train_samples



            #self.aggregate_grads_AVG()
            self.aggregate_grads_AVG_new()

            #self.global_v()

        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_accandloss()
        self.save_variance()
        # self.save_model()

    def compute_global_and_local(self):
        lv_total = 0
        for user in self.users:
            avg_single_user = user.estimate(self.local_epochs)
            lv_total += avg_single_user
        lv_avg = lv_total / len(self.users)
        gv_avg, gv_max = self.global_v()
        print('lv_avg: ' + str(lv_avg) + ', gv_avg: ' + str(gv_avg) + ', gv_max: ' + str(gv_max))
        self.lv_avg_list.append(lv_avg)
        self.gv_avg_list.append(gv_avg)
        self.gv_max_list.append(gv_max)
    def global_v(self):
        server_model_param_list = None
        param_sum = 0
        param_max = -sys.maxsize
        for layer in self.model.parameters():
            if server_model_param_list is None:
                server_model_param_list = layer.data.view(-1)
            else:
                server_model_param_list = torch.cat((server_model_param_list, layer.data.view(-1)))

        # print(server_model_param_list)
        for user in self.selected_users:

            user_model_param_list = None

            for user_layer in user.model.parameters():
                if user_model_param_list is None:
                    user_model_param_list = user_layer.data.view(-1)
                else:

                    user_model_param_list = torch.cat((user_model_param_list, user_layer.data.view(-1)))

            # print(user_model_param_list)
            user_model_param_list = user_model_param_list.to(self.device)
            result_list = server_model_param_list - user_model_param_list
            print(result_list)
            cur_result = torch.norm(result_list).item()

            param_sum += cur_result
            # print(cur_result)
            param_max = max(param_max, cur_result)

        param_avg = param_sum / len(self.selected_users)
        print("Complete cauclate in one round")
        print('total: ' + str(param_sum))
        print('avg: ' + str(param_avg))
        print('max: ' + str(param_max))

        return param_avg, param_max
