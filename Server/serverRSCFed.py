import os
from experiments.paths import log_directory, metrics_directory
from datautils.partition_store import load_partition, partition_directory, partition_exists, save_partition

import torch
import copy
from User.userfixmatch import UserFixMatch
from User.userrscfed import UserRSCFed
from Server.serverbase import Server
from datautils.data_utils.data_utils import get_data_loader
from datautils.data_utils.ssl_dataset import SSL_Dataset, get_transforms
from datautils.data_utils.dataset import BasicDataset
from datautils.datasets import SVHN_truncated, CIFAR100_truncated_SUB,TINYIMAGENET_truncated
from datautils.utils import *
# from datautils.utils import *
import torch.nn as nn
# from utils.model_utils import read_data, read_user_data
import numpy as np
from Models.resnet import *
from Models.lenet import *
from Models.basic_nets import *
import math

from utils.utils import Averager
mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
mean['mnist'] = [0.5]
std['mnist'] = [0.5]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
std['svhn'] = [0.1751, 0.1771, 0.1744]
# Implementation for pFedMe Server


class RSCFed(Server):
    def __init__(self, args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_net,
                 name_optimizer, curtimes, dataset, datadir, partition, num_users, beta):
        super().__init__(args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_net,
                         name_optimizer,
                         curtimes, dataset, datadir, partition, num_users, beta)

        self.save_path_name=args.save_path_name

        root_path = str(partition_directory()) + os.sep
        file_name = dataset + "_" + str(num_users) + "_" + partition + "_" + str(beta)
        self.flag = file_name
        self.split_client = [2, 0, 18] #
        if self.dataset.upper() == 'STL10':
            self.split_client = [1, 0, 19]
        suffix = '.npy'
        path = root_path + file_name + suffix
        if not partition_exists(path):
            self.prepare_data(path)
            # return
        total_data = load_partition(path)
        self.server_train_transform, self.server_test_trainsform = get_transforms(self.dataset)

        self.newcurtime=curtimes
        # load backbone model
        '''
        net_builder = load_model(args, name_net)
        instance_model = net_builder(num_classes=self.num_classes)
        torch.cuda.set_device(args.gpu)
        instance_model = instance_model.cuda(args.gpu)
        '''
        if self.dataset.upper() == 'CIFAR10':
            instance_model = ResNet18()
            #instance_model = AlexNet(args.num_classes)
            #instance_model = get_basic_net(net="VGG19-BN", n_classes=10)
        if self.dataset.upper() == 'SVHN':
            instance_model = AlexNet(args.num_classes)
            #instance_model =  ResNet18()
            #instance_model = get_basic_net(net="VGG19-BN", n_classes=10)
        if self.dataset.upper() == 'STL10':
            instance_model = get_basic_net(net="ResNet20", n_classes=10)

        if self.dataset.upper() == 'CIFAR100':
            instance_model = get_basic_net(net="ResNet56", n_classes=100)
        if self.dataset.upper() == 'TINYIMAGENET':
            instance_model = get_basic_net(net="ResNet56", n_classes=200)
            # instance_model = get_basic_net(net="ResNet20",n_classes=100)
            # instance_model=get_basic_net(net="VGG19-BN",n_classes=100)
        torch.cuda.set_device(args.gpu)

        instance_model = instance_model.cuda(args.gpu)
        self.model = copy.deepcopy(instance_model)
        self.model_group = [copy.deepcopy(self.model) for i in range(self.num_total_users)]
        # 20221028 adding for RSCFed
        # self.sub_model_group = [copy.deepcopy(self.model) for i in range(self.args.S_M)]
        self.avg_model_temp = copy.deepcopy(self.model)
        self.sub_models = {}
        for i_M in range(self.args.S_M):
            self.sub_models[i_M] = copy.deepcopy(self.model)
        # self.split_client = [10, 10, 10]
        self.datasize_client = torch.zeros((self.num_total_users,)).cuda(args.gpu)

        self.supervisedloss=[]
        self.unsupervisedloss=[]
        self.globalacc=[]
        self.labelusers=[]
        self.unlabelusers=[]
        self.GlobalLabelDistribution={}
        # assign data and model to generate all users using instance
        for i in range(self.num_total_users):
            id_user = i

            lb_dset, ulb_dset = self.get_per_user_data(total_data, i, onehot=False)

            if id_user < self.split_client[0]:
                tag_client_lbulb = 0
            elif id_user < (self.split_client[0] + self.split_client[1]):
                tag_client_lbulb = 1
            else:
                tag_client_lbulb = 2

            # tag_client_lbulb = 0

            if tag_client_lbulb == 0:
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}
                loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                                          args.batch_size,
                                                          num_iters=1048576,
                                                          data_sampler=args.train_sampler,
                                                          num_workers=args.num_workers)

                loader_dict['train_ulb'] = None
                self.datasize_client[id_user] = args.batch_size
            elif tag_client_lbulb == 2:
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}
                loader_dict['train_lb'] = None
                loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                                           args.batch_size,
                                                           num_iters=1048576,
                                                           data_sampler=args.train_sampler,
                                                           num_workers=args.num_workers)



            else:
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}
                loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                                          args.batch_size,
                                                          num_epochs=None,
                                                          num_iters=None,
                                                          replacement=False,
                                                          data_sampler=args.train_sampler,
                                                          num_workers=args.num_workers)

                loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                                           args.batch_size,
                                                           num_epochs=None,
                                                           num_iters=None,
                                                           replacement=False,
                                                           data_sampler=args.train_sampler,
                                                           num_workers=args.num_workers)
                self.datasize_client[id_user] = len(lb_dset) + len(ulb_dset)



            user = UserRSCFed(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, instance_model, args, logger, device, aggregate_method_server,
                                name_algorithm, name_data_total, name_net, name_optimizer)

            self.users.append(user)


        # init prototype and prototype group
        self.len_feature = 128
        self.global_proto = torch.zeros((self.args.num_classes, self.len_feature)).cuda(args.gpu)
        self.global_proto_group = torch.zeros((self.args.num_classes, self.num_total_users, self.len_feature)).cuda(args.gpu)
        self.global_proto_group_count = torch.zeros((self.args.num_classes, self.args.num_total_users)).cuda(self.args.gpu)

        self.global_proto_list = []
        self.global_proto_group_list = []
        self.global_proto_group_count_list = []

    def prepare_data(self, filename):
        patition_users = self.num_users if self.dataset != 'stl10' else self.split_client[2]
        partition_res = partition_data(dataset=self.dataset, datadir=self.datadir, logdir=log_directory(),
                                       partition=self.partition, n_parties=patition_users, beta=self.beta)
        net_dataidx_map = partition_res[4]
        if self.dataset.upper() == "CIFAR10":
            dl_obj = CIFAR10_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset.upper() == 'STL10':
            dl_obj = STL10_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset.upper() == 'SVHN':
            dl_obj = SVHN_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset.upper() == 'CIFAR100':
            dl_obj = CIFAR100_truncated_SUB
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset.upper() == 'TINYIMAGENET':
            dl_obj = TINYIMAGENET_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        server_test_ds = dl_obj(self.datadir, train=False, transform=transform_test, download=True)

        if self.dataset == 'stl10':
            server_unlabel_ds = dl_obj(self.datadir, unlabel=True, transform=transform_train, download=True)
            server_train_ds = dl_obj(self.datadir, train=True, transform=transform_train, download=True)
        elif self.dataset == 'tinyimagenet':
            server_train_ds = dl_obj(os.path.join(self.datadir, 'train'), train=True, transform=transform_train)
        else:
            server_train_ds = dl_obj(self.datadir, train=True, transform=transform_train, download=True)
        data_map = {}
        if self.dataset == 'stl10':
            for i in range(self.split_client[0], self.num_users):
                lb_data, lb_target = server_unlabel_ds.get_index_data(net_dataidx_map[i - self.split_client[0]])
                user_data = {'data': lb_data, 'target': lb_target}
                data_map[i] = user_data
            avg_split = self.split_client[0]
            per_label_user = int(len(STL10_truncated.train_data) / avg_split)
            l = 0
            r = per_label_user
            for i in range(0, avg_split):
                lb_data, lb_target = server_train_ds.get_data_by_range(l, r)
                user_data = {'data': lb_data, 'target': lb_target}
                data_map[i] = user_data
                l = l + per_label_user
                r = r + per_label_user
        else:
            for i in range(0, self.num_users):
                lb_data, lb_target = server_train_ds.get_index_data(net_dataidx_map[i])
                user_data = {'data': lb_data, 'target': lb_target}

                data_map[i] = user_data

        save_partition(filename, data_map)

    def get_per_user_data(self, total_data, idx, strong_transofrm=None, onehot=False):
        lb_data, lb_target = total_data[idx]['data'], total_data[idx]['target']
        ulb_data = lb_data.__deepcopy__(None)
        ulb_target = lb_target.__deepcopy__(None)
        lb_dset = BasicDataset(self.name_algorithm, lb_data, lb_target, self.args.num_classes,
                               self.server_train_transform,
                               False, strong_transofrm, onehot, self.server_test_trainsform)
        ulb_dset = BasicDataset(self.name_algorithm, ulb_data, ulb_target, self.args.num_classes,
                                self.server_train_transform,
                                True, strong_transofrm, onehot)
        return lb_dset, ulb_dset

    def new_result_save(self):
        acc_path = metrics_directory(self.args)
        acc_name = acc_path + f"rscfed_" + self.flag + "_testacc.npy"
        loss_name = (acc_path + f"rscfed_" + self.flag +
                     "_idealloss.npy")
        precision_name = acc_path + f"rscfed_" + self.flag + "_precision.npy"
        recall_name = (acc_path + f"rscfed_" + self.flag + "_recall.npy")
        AUC_name = acc_path + f"rscfed_" + self.flag + "_AUC.npy"
        F1_name = (acc_path + f"rscfed_" + self.flag + "_F1.npy")
        localvariance_name = (acc_path + f"rscfed_" + self.flag +
                              "_localvariance.npy")
        globalvariance_name = (acc_path + f"rscfed_" + self.flag +
                               "_globalvariance.npy")
        gradnorm_name = (acc_path + f"rscfed_" + self.flag +
                         "_gradnorm.npy")
        np.save(acc_name, self.testacc)
        np.save(loss_name, self.idealloss)
        np.save(precision_name, self.precision)
        np.save(recall_name, self.recall)
        np.save(AUC_name, self.AUC)
        np.save(F1_name, self.F1)
        if self.dataset.upper()=='CIFAR100' or self.dataset.upper()=='TINYIMAGENET':
            top5_name=acc_path + f"rscfed_" + self.flag + "_top5.npy"
            np.save(top5_name, self.top5)



    def saveloss(self):
        save_path = "./Empirical/"   
        #su_name = save_path + f"cifar10_centralized_supervisedloss.npy"
        #un_name = save_path + f"cifar10_centralized_unsupervisedloss.npy"
        #acc_name=save_path+f"cifar10_centralized_globalacc.npy"
        #su_name = save_path + f"cifar10_federated__supervisedloss.npy"
        #un_name = save_path + f"cifar10_federated_unsupervisedloss.npy"
        #acc_name=save_path+f"cifar10_federated_globalacc.npy"
        #su_name = save_path + f"cifar10_federated_GI800_lr0.03_suloss.npy"
        #un_name = save_path + f"cifar10_federated_GI800_lr0.03_unsuloss.npy"
        #acc_name=save_path+f"cifar10_federated_GI800_lr0.03_globalacc.npy"
        #su_name = save_path + f"cifar10_federated_GI800_lr0.03_sample10_suloss.npy"
        #un_name = save_path + f"cifar10_federated_GI800_lr0.03_sample10_unsuloss.npy"
        #acc_name=save_path+f"cifar10_federated_GI800_lr0.03_sample10_globalacc.npy"
        #su_name = save_path + f"cifar10_federated_GI1600_lr0.001_sample10_suloss.npy"
        #un_name = save_path + f"cifar10_federated_GI1600_lr0.001_sample10_unsuloss.npy"
        acc_name=save_path+f"rscfed_"+self.save_name
        #np.save(su_name, self.supervisedloss)
        #np.save(un_name, self.unsupervisedloss)
        np.save(acc_name,self.globalacc)



    def process(self):
        # The complete process of Server.
        # if self.args.num_pretrain != 0:
        #     self.load_model(self.args.num_pretrain)
        #     self.load_proto(self.args.num_pretrain)
        #     # self.model_group = [copy.deepcopy(self.model) for i in range(self.num_total_users)]
        glob_iter_begin = 0


        for glob_iter in range(glob_iter_begin, glob_iter_begin + self.num_global_iters):
            supervisedloss=0
            unsupervisedloss=0
            self.logger.warning(f"-------------Round number: {glob_iter} -------------")
            print(f"-------------Round number: {glob_iter} -------------")
            '''
            # send all parameter for users
            self.send_parameters()

            # Personalized by sending parameter for selected users
            # self.send_parameters_group()

            # Evaluate gloal model on user for each interation
            self.logger.warning("Evaluate global model")
            # self.logger.warning("")
            # self.evaluate()
            self.evaluate_global_test_dataset(args=self.args)
            # self.logger.warning("Eval end")
            # self.evaluate_on_local_test_dataset()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.args, glob_iter, self.logger)  # * user.train_samples
            # self.logger.warning("Train end")
            # Evaluate gloal model on user for each interation
            # print("Evaluate persionalized model")
            # print("")
            # self.evaluate_personalized_model()

            # choose several users to send back upated model to server
            self.id_selected_users, self.selected_users = self.select_users(glob_iter, self.num_subset_users)
            # self.selected_users = self.select_users(glob_iter, self.num_users)
            '''
            # save proto
            self.global_proto_list.append(self.global_proto)
            self.global_proto_group_list.append(self.global_proto_group)
            self.global_proto_group_count_list.append(self.global_proto_group_count)

            # self.logger.warning("Evaluate global model")
            print(f"Evaluate global model")
            # self.logger.warning("")
            # self.evaluate()
            self.evaluate_global_test_dataset(args=self.args)
            self.start_training_round(glob_iter + 1)

            # self.aggregate_parameters()
            # id_selected_users = np.random.choice(len(self.users), num_users, replace=False)
            # renew_num_users = np.arange(0, self.num_total_users)

            # if self.args.work_mode == 0:
            #     renew_num_users = np.arange(0, self.num_total_users)
            # elif self.args.work_mode == 1:
            #     renew_num_users = np.arange(0, self.split_client[0])
            # elif self.args.work_mode == 2:
            #     renew_num_users = np.arange(self.split_client[0], self.split_client[0]+self.split_client[1])
            # elif self.args.work_mode == 3:
            #     renew_num_users = np.arange(self.split_client[0]+self.split_client[1], self.num_total_users)
            # elif self.args.work_mode == 4:
            #     renew_num_users = np.arange(0, self.split_client[0]+self.split_client[1])
            # elif self.args.work_mode == 5:
            #     renew_num_users = np.concatenate((np.arange(0, self.split_client[0]), np.arange(self.split_client[0]+self.split_client[1], self.num_total_users)))
            # elif self.args.work_mode == 6:
            #     renew_num_users = np.arange(self.split_client[0], self.num_total_users)
            renew_num_users = np.arange(0, self.num_total_users)

            for i_M in range(self.args.S_M):
                if self.args.selected_mode == 0:
                    renew_num_users = np.arange(0, self.num_total_users)
                    n_sam_clients = self.args.S_K
                    sam_clients = np.random.choice(
                        renew_num_users, n_sam_clients, replace=False
                    )
                elif self.args.selected_mode == 1:
                    # 20221029 adding
                    renew_num_users_0 = np.arange(0, self.split_client[0])
                    n_sam_clients_0 = math.ceil(len(renew_num_users_0) * (self.args.S_K / self.num_total_users))
                    n_sam_clients_0 = self.args.S_K_lb
                    sam_clients_0 = np.random.choice(
                        renew_num_users_0, n_sam_clients_0, replace=False
                    )
                    renew_num_users_1 = np.arange(self.split_client[0], self.num_total_users)
                    # n_sam_clients_1 = math.ceil(len(renew_num_users_1) * (self.num_subset_users / self.num_total_users))
                    n_sam_clients_1 = self.args.S_K - n_sam_clients_0
                    sam_clients_1 = np.random.choice(
                        renew_num_users_1, n_sam_clients_1, replace=False
                    )
                    sam_clients = np.concatenate((sam_clients_0, sam_clients_1))


                #print(f"selected client: {sam_clients}")
                # self.selected_users = [copy.deepcopy(self.users[n_sam_clients[i]]) for i in range(len(n_sam_clients))]

                # sam_clients = np.arange(10,20)
                local_models = {}
                local_protos = {}
                local_protos_counts = {}

                avg_loss = Averager()
                all_per_accs = []
                # self.send_parameters()
                for client in sam_clients:
                    user_temp = self.users[client]
                    local_model, per_accs, loss, local_proto, local_protos_count = user_temp.train(
                        down_model=copy.deepcopy(self.model),
                        down_proto=copy.deepcopy(self.global_proto),
                        args=self.args,
                        glob_iter=glob_iter,
                        logger=self.logger,
                    )

                    local_models[client] = copy.deepcopy(local_model)
                    local_protos[client] = local_proto.clone().detach()
                    local_protos_counts[client] = local_protos_count.clone().detach()

                self.update_sub_model_basic_ratio(r=glob_iter,
                                                  sub_model=self.sub_models[i_M],
                                                  local_models=local_models,
                                                  alpha=self.args.model_m,
                                                  )

            # update global model
            self.update_global(
                r=glob_iter,
                global_model=self.model,
                local_models=self.sub_models,
            )
            self.finish_training_round()

        # print(loss)
        #self.save_results()
        #self.saveloss()
        self.new_result_save()

    def update_sub_model_basic_ratio(self, r, sub_model, local_models, alpha=0):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        mean_state_dict = {}

        vs2 = []
        for client in local_models.keys():
            vs2.append(self.datasize_client[client])
        vs2 = torch.stack(vs2, dim=0)
        vs2_sum = vs2.sum(dim=-1, keepdim=False)
        if vs2_sum == 0:
            return
        else:
            vs2_sum_clamp = torch.clamp(vs2_sum, min=1e-6)
            vs2_cor = torch.div(vs2, vs2_sum_clamp)

        for name, param in self.avg_model_temp.state_dict().items():
            vs = []
            # vs2 = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
                # vs2.append(local_protos_counts[client])
            vs = torch.stack(vs, dim=0)
            # vs2 = torch.stack(vs2, dim=0)
            # vs2_sum = vs2.sum(dim=1, keepdim=False)
            # vs2_sum_sum = vs2_sum.sum(dim=0, keepdim=False)
            # vs2_sum_sum_clamp = torch.clamp(vs2_sum_sum, min=1e-6)
            # vs2_cor = torch.div(vs2_sum, vs2_sum_sum_clamp)
            # mean_value = torch.zeros_like(vs[0])
            mean_value = copy.deepcopy(vs2_cor[0] * vs[0])
            for index_vs in range(1, len(vs)):
                mean_value = mean_value + vs2_cor[index_vs] * vs[index_vs]
            # try:
            #     mean_value = vs.mean(dim=0)
            # except Exception:
            #     # for BN's cnt
            #     mean_value = (1.0 * vs).mean(dim=0).long()
            # mean_state_dict[name] = mean_value
            mean_state_dict[name] = alpha * self.avg_model_temp.state_dict()[name] + (1 - alpha) * mean_value

        self.avg_model_temp.load_state_dict(mean_state_dict, strict=False)

        dist_list = {}
        # for cli_idx in range(self.args.S_K):
        for client in local_models.keys():
            dist = self.model_dist(local_models[client], self.avg_model_temp)
            # dist_list.append(dist)
            dist_list[client] = dist

        # dist_scale_f = 10000
        dist_scale_f = self.args.dist_scale_f
        clt_freq_this_meta_uncer = [
            torch.exp(-dist_list[client] * dist_scale_f / self.datasize_client[client]) * self.datasize_client[client]
            for client in local_models.keys()]
        total_temp = sum(clt_freq_this_meta_uncer)
        print(f"total_temp is {total_temp}")
        if total_temp == 0:
            clt_freq_this_meta_dist = [1.0 /self.args.S_K  for i in range(self.args.S_K)]
        else:
            clt_freq_this_meta_dist = [clt_freq_this_meta_uncer[i] / total_temp for i in range(self.args.S_K)]

        input_ratio = torch.tensor(clt_freq_this_meta_dist).cuda(self.args.gpu)
        print(f"input_ratio is {input_ratio}")

        if self.args.re_input_ratio == 1:
            input_ratio_lb_percent = self.args.input_ratio_lb_percent
            input_ratio_ulb_percent = 1.0 - input_ratio_lb_percent
            input_ratio_lb = []
            input_ratio_ulb = []
            input_ratio_lb_index = []
            input_ratio_ulb_index = []
            index_client = 0
            for client in local_models.keys():
                if client < self.split_client[0]:
                    input_ratio_lb_index.append(index_client)
                    input_ratio_lb.append(input_ratio[index_client])
                else:
                    input_ratio_ulb_index.append(index_client)
                    input_ratio_ulb.append(input_ratio[index_client])
                index_client = index_client + 1
            # re_input_ratio = input_ratio
            if input_ratio_lb_index:
                if torch.sum(input_ratio[input_ratio_lb_index]) == 0:
                    input_ratio[input_ratio_lb_index] = input_ratio_lb_percent / len(input_ratio_lb_index)
                else:
                    input_ratio[input_ratio_lb_index] = input_ratio_lb_percent * input_ratio[
                        input_ratio_lb_index] / torch.sum(
                        input_ratio[input_ratio_lb_index])

            if input_ratio_ulb_index:
                if torch.sum(input_ratio[input_ratio_ulb_index]) == 0:
                    input_ratio[input_ratio_ulb_index] = input_ratio_ulb_percent / len(input_ratio_ulb_index)
                else:
                    if input_ratio_lb_index:
                        input_ratio_ulb_percent = 1.0 - input_ratio_lb_percent
                    else:
                        input_ratio_ulb_percent = 1.0
                    input_ratio[input_ratio_ulb_index] = input_ratio_ulb_percent * input_ratio[
                        input_ratio_ulb_index] / torch.sum(
                        input_ratio[input_ratio_ulb_index])


            print(f"re_input_ratio is {input_ratio}")


        # reweight 50% to lb client, 50% to ulb client


        # return input_ratio
        self.update_global_model_input_ratio(r=r,
                                             global_model=sub_model,
                                             local_models=local_models,
                                             input_ratio=input_ratio)


    def model_dist(self, w_1, w_2):
        # assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
        dist_total = torch.zeros(1).float().cuda(self.args.gpu)
        # for key in w_1:
        #     dist = torch.norm(w_1[key] - w_2[key])
        #     dist_total += dist.cuda(self.args.gpu)
        for name, param in w_1.state_dict().items():
            dist = torch.norm((param - w_2.state_dict()[name]).float())
            dist_total += dist

        return dist_total.item()

    def update_global_model_input_ratio(self, r, global_model, local_models, input_ratio=None):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        mean_state_dict = {}

        # vs2 = []
        # for client in local_models.keys():
        #     vs2.append(self.datasize_client[client])
        # vs2 = torch.stack(vs2, dim=0)
        vs2 = input_ratio
        vs2_sum = vs2.sum(dim=-1, keepdim=False)
        if vs2_sum == 0:
            return
        else:
            vs2_sum_clamp = torch.clamp(vs2_sum, min=1e-9)
            vs2_cor = torch.div(vs2, vs2_sum_clamp)


        for name, param in global_model.state_dict().items():
            vs = []
            # vs2 = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
                # vs2.append(local_protos_counts[client])
            vs = torch.stack(vs, dim=0)
            # vs2 = torch.stack(vs2, dim=0)
            # vs2_sum = vs2.sum(dim=1, keepdim=False)
            # vs2_sum_sum = vs2_sum.sum(dim=0, keepdim=False)
            # vs2_sum_sum_clamp = torch.clamp(vs2_sum_sum, min=1e-6)
            # vs2_cor = torch.div(vs2_sum, vs2_sum_sum_clamp)
            # mean_value = torch.zeros_like(vs[0])
            mean_value = copy.deepcopy(vs2_cor[0] * vs[0])
            for index_vs in range(1, len(vs)):
                mean_value = mean_value + vs2_cor[index_vs] * vs[index_vs]
            # try:
            #     mean_value = vs.mean(dim=0)
            # except Exception:
            #     # for BN's cnt
            #     mean_value = (1.0 * vs).mean(dim=0).long()
            # mean_state_dict[name] = mean_value
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)


    def update_global(self, r, global_model, local_models):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def update_global_model_basic_ratio(self, r, global_model, local_models, alpha=0):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        mean_state_dict = {}

        vs2 = []
        for client in local_models.keys():
            vs2.append(self.datasize_client[client])
        vs2 = torch.stack(vs2, dim=0)
        vs2_sum = vs2.sum(dim=-1, keepdim=False)
        if vs2_sum == 0:
            return
        else:
            vs2_sum_clamp = torch.clamp(vs2_sum, min=1e-6)
            vs2_cor = torch.div(vs2, vs2_sum_clamp)


        for name, param in global_model.state_dict().items():
            vs = []
            # vs2 = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
                # vs2.append(local_protos_counts[client])
            vs = torch.stack(vs, dim=0)
            # vs2 = torch.stack(vs2, dim=0)
            # vs2_sum = vs2.sum(dim=1, keepdim=False)
            # vs2_sum_sum = vs2_sum.sum(dim=0, keepdim=False)
            # vs2_sum_sum_clamp = torch.clamp(vs2_sum_sum, min=1e-6)
            # vs2_cor = torch.div(vs2_sum, vs2_sum_sum_clamp)
            # mean_value = torch.zeros_like(vs[0])
            mean_value = copy.deepcopy(vs2_cor[0] * vs[0])
            for index_vs in range(1, len(vs)):
                mean_value = mean_value + vs2_cor[index_vs] * vs[index_vs]
            # try:
            #     mean_value = vs.mean(dim=0)
            # except Exception:
            #     # for BN's cnt
            #     mean_value = (1.0 * vs).mean(dim=0).long()
            # mean_state_dict[name] = mean_value
            mean_state_dict[name] = alpha * global_model.state_dict()[name] + (1 - alpha) * mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def update_global_model_addcount(self, r, global_model, local_models, global_proto_group_count, local_protos_counts, alpha=0):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        mean_state_dict = {}

        vs2 = []
        for client in local_models.keys():
            vs2.append(local_protos_counts[client])
        vs2 = torch.stack(vs2, dim=0)
        vs2_sum = vs2.sum(dim=1, keepdim=False)
        vs2_sum_sum = vs2_sum.sum(dim=0, keepdim=False)
        if vs2_sum_sum == 0:
            return
        else:
            vs2_sum_sum_clamp = torch.clamp(vs2_sum_sum, min=1e-6)
            vs2_cor = torch.div(vs2_sum, vs2_sum_sum_clamp)


        for name, param in global_model.state_dict().items():
            vs = []
            # vs2 = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
                # vs2.append(local_protos_counts[client])
            vs = torch.stack(vs, dim=0)
            # vs2 = torch.stack(vs2, dim=0)
            # vs2_sum = vs2.sum(dim=1, keepdim=False)
            # vs2_sum_sum = vs2_sum.sum(dim=0, keepdim=False)
            # vs2_sum_sum_clamp = torch.clamp(vs2_sum_sum, min=1e-6)
            # vs2_cor = torch.div(vs2_sum, vs2_sum_sum_clamp)
            # mean_value = torch.zeros_like(vs[0])
            mean_value = copy.deepcopy(vs2_cor[0] * vs[0])
            for index_vs in range(len(vs)-1):
                mean_value = mean_value + vs2_cor[index_vs+1] * vs[index_vs+1]
            # try:
            #     mean_value = vs.mean(dim=0)
            # except Exception:
            #     # for BN's cnt
            #     mean_value = (1.0 * vs).mean(dim=0).long()
            # mean_state_dict[name] = mean_value
            mean_state_dict[name] = alpha * global_model.state_dict()[name] + (1 - alpha) * mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def update_global_model_addcount_expand(self, r, global_model, local_models, global_proto_group_count, local_protos_counts, alpha=0):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        mean_state_dict = {}

        vs2 = []
        for client in local_models.keys():
            if (client < self.split_client[0]):
                vs2.append(local_protos_counts[client] * (self.split_client[2] / self.split_client[0]))
            else:
                vs2.append(local_protos_counts[client])
        vs2 = torch.stack(vs2, dim=0)
        vs2_sum = vs2.sum(dim=1, keepdim=False)
        vs2_sum_sum = vs2_sum.sum(dim=0, keepdim=False)
        if vs2_sum_sum == 0:
            return
        else:
            vs2_sum_sum_clamp = torch.clamp(vs2_sum_sum, min=1e-6)
            vs2_cor = torch.div(vs2_sum, vs2_sum_sum_clamp)


        for name, param in global_model.state_dict().items():
            vs = []
            # vs2 = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
                # vs2.append(local_protos_counts[client])
            vs = torch.stack(vs, dim=0)
            # vs2 = torch.stack(vs2, dim=0)
            # vs2_sum = vs2.sum(dim=1, keepdim=False)
            # vs2_sum_sum = vs2_sum.sum(dim=0, keepdim=False)
            # vs2_sum_sum_clamp = torch.clamp(vs2_sum_sum, min=1e-6)
            # vs2_cor = torch.div(vs2_sum, vs2_sum_sum_clamp)
            # mean_value = torch.zeros_like(vs[0])
            mean_value = copy.deepcopy(vs2_cor[0] * vs[0])
            for index_vs in range(len(vs)-1):
                mean_value = mean_value + vs2_cor[index_vs+1] * vs[index_vs+1]
            # try:
            #     mean_value = vs.mean(dim=0)
            # except Exception:
            #     # for BN's cnt
            #     mean_value = (1.0 * vs).mean(dim=0).long()
            # mean_state_dict[name] = mean_value
            mean_state_dict[name] = alpha * global_model.state_dict()[name] + (1 - alpha) * mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def update_global_proto_simple(self, r, global_proto, local_protos, alpha=0.9):
        vs = []
        for client in local_protos.keys():
            vs.append(local_protos[client])
        vs = torch.stack(vs, dim=0)

        try:
            mean_value = vs.mean(dim=0)
        except Exception:
            # for BN's cnt
            mean_value = (1.0 * vs).mean(dim=0).long()
        global_proto = alpha * global_proto + (1 - alpha) * mean_value

        return global_proto

    def update_global_proto_group(self, r, global_proto_group, local_protos, local_protos_counts, alpha=0.5):
        for client in local_protos.keys():
            global_proto_group[:, client, :] = alpha * global_proto_group[:, client, :] + (1 - alpha) * local_protos[client]

        global_proto_center_mean = global_proto_group.mean(dim=1)  # keepdim=True
        global_proto_center_mean_temp = global_proto_center_mean.unsqueeze(2)
        global_proto_center_corr = torch.matmul(global_proto_group, global_proto_center_mean_temp)
        global_proto_center_corr = global_proto_center_corr.squeeze()
        global_proto_center_corr_norm = torch.nn.functional.softmax(global_proto_center_corr, dim=1)
        # print(f"global_iter: {r}, {global_proto_center_corr_norm}")
        global_proto_center_corr_norm = global_proto_center_corr_norm.unsqueeze(1)
        global_proto_center_rewt = torch.matmul(global_proto_center_corr_norm, global_proto_group)
        global_proto_center_rewt = global_proto_center_rewt.squeeze()

        return global_proto_center_mean, global_proto_center_rewt

    def update_global_proto_group_addcount(self, r, global_proto, global_proto_group, global_proto_group_count, local_protos, local_protos_counts, alpha=0, update_proto_mode=4):
        # local_protos_counts_subset = torch.zeros((self.args.num_classes, self.args.num_subset_users)).cuda(self.args.gpu)

        if update_proto_mode == 3:
            vs = []
            vs2 = []
            for client in local_protos.keys():
                vs.append(local_protos[client])
                vs2.append(local_protos_counts[client])
            vs = torch.stack(vs, dim=0)
            vs2 = torch.stack(vs2, dim=0)
            vs2_sum = vs2.sum(dim=0, keepdim=True)
            vs2_sum_clamp = torch.clamp(vs2_sum, min=1e-6)
            vs2_cor = torch.div(vs2, vs2_sum_clamp)
            vs2_repeat = vs2_cor.repeat(self.len_feature, 1, 1)
            global_proto_center_mean_count_pre = torch.mul(vs2_repeat.permute(1, 2, 0), vs)
            global_proto_center_mean_count_oneglobal = global_proto_center_mean_count_pre.sum(dim=0)
            global_proto_center_mean_count = alpha * global_proto + (1 - alpha) * global_proto_center_mean_count_oneglobal

            for client in local_protos.keys():
                global_proto_group[:, client, :] = alpha * global_proto_group[:, client, :] + (1 - alpha) * local_protos[client]
                global_proto_group_count[:, client] = local_protos_counts[client]

            return global_proto_center_mean_count, global_proto_group, global_proto_group_count

        if update_proto_mode == 4:
            for client in local_protos.keys():
                global_proto_group[:, client, :] = alpha * global_proto_group[:, client, :] + (1 - alpha) * local_protos[client]
                global_proto_group_count[:, client] = local_protos_counts[client]

            global_proto_group_count_sum_clamp = torch.clamp(global_proto_group_count.sum(dim=1, keepdim=True), min=1e-6)
            global_proto_group_count_cor = torch.div(global_proto_group_count, global_proto_group_count_sum_clamp)
            global_proto_group_count_repeat = global_proto_group_count_cor.repeat(self.len_feature, 1, 1)
            global_proto_center_rewt_count_pre = torch.mul(global_proto_group_count_repeat.permute(1, 2, 0), global_proto_group)
            global_proto_center_rewt_count = global_proto_center_rewt_count_pre.sum(dim=1)

            return global_proto_center_rewt_count, global_proto_group, global_proto_group_count

        if update_proto_mode == 5:
            for client in local_protos.keys():
                global_proto_group[:, client, :] = alpha * global_proto_group[:, client, :] + (1 - alpha) * local_protos[client]
                if (client < self.split_client[0]):
                    global_proto_group_count[:, client] = local_protos_counts[client] * (self.split_client[2] / self.split_client[0])
                else:
                    global_proto_group_count[:, client] = local_protos_counts[client]

            global_proto_group_count_sum_clamp = torch.clamp(global_proto_group_count.sum(dim=1, keepdim=True), min=1e-6)
            global_proto_group_count_cor = torch.div(global_proto_group_count, global_proto_group_count_sum_clamp)
            global_proto_group_count_repeat = global_proto_group_count_cor.repeat(self.len_feature, 1, 1)
            global_proto_center_rewt_count_pre = torch.mul(global_proto_group_count_repeat.permute(1, 2, 0), global_proto_group)
            global_proto_center_rewt_count = global_proto_center_rewt_count_pre.sum(dim=1)

            return global_proto_center_rewt_count, global_proto_group, global_proto_group_count

        if update_proto_mode == 6:
            for client in local_protos.keys():
                if (client < self.split_client[0]):
                    global_proto_group[:, client, :] = alpha * global_proto_group[:, client, :] + (1 - alpha) * local_protos[client]
                    global_proto_group_count[:, client] = local_protos_counts[client]

            global_proto_group_count_sum_clamp = torch.clamp(global_proto_group_count.sum(dim=1, keepdim=True), min=1e-6)
            global_proto_group_count_cor = torch.div(global_proto_group_count, global_proto_group_count_sum_clamp)
            global_proto_group_count_repeat = global_proto_group_count_cor.repeat(self.len_feature, 1, 1)
            global_proto_center_rewt_count_pre = torch.mul(global_proto_group_count_repeat.permute(1, 2, 0), global_proto_group)
            global_proto_center_rewt_count = global_proto_center_rewt_count_pre.sum(dim=1)

            return global_proto_center_rewt_count, global_proto_group, global_proto_group_count





  
