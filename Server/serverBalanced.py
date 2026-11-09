import os
from experiments.paths import log_directory, metrics_directory
from experiments.local_training import local_loader_options, uses_local_epochs
from datautils.partition_store import load_partition, partition_directory, partition_exists, save_partition

import torch
import copy
from User.userreweight_mycb import UserReweight_mycb
from Server.serverbase import Server
from datautils.data_utils.data_utils import get_data_loader

from datautils.data_utils.ssl_dataset import SSL_Dataset, get_transforms
from datautils.data_utils.dataset import BasicDataset
from datautils.datasets import SVHN_custom, CIFAR100_truncated_SUB,TINYIMAGENET_truncated
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


class Balanced(Server):
    def __init__(self, args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_net,
                 name_optimizer, curtimes, dataset, datadir, partition, num_users, beta):
        super().__init__(args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_net,
                         name_optimizer,
                         curtimes, dataset, datadir, partition, num_users, beta)
        self.save_path_name=args.save_path_name
        self.balanced_k=1
        self.ta_threshold=0.7
        root_path = str(partition_directory()) + os.sep
        file_name = dataset + "_" + str(num_users) + "_" + partition + "_" + str(beta)
        self.flag = file_name
        self.split_client = [2, 0, 18] #
        if self.num_total_users==50:
            self.split_client = [5, 0, 45]
        if self.num_total_users==100:
            self.split_client = [10, 0, 90]
        if self.dataset.upper() == 'STL10':
            self.split_client = [1, 0, 19]
        suffix = '.npy'
        path = root_path + file_name + suffix
        if not partition_exists(path):
            self.prepare_data(path)
            # return
        total_data = load_partition(path)
        self.server_train_transform, self.server_test_trainsform = get_transforms(self.dataset)

        self.newcurtime = str(curtimes)
        if self.dataset.upper() == 'CIFAR10':
            instance_model = ResNet18()
            #instance_model = AlexNet(args.num_classes)
            #instance_model = get_basic_net(net="VGG19-BN", n_classes=10)
        if self.dataset.upper() == 'CIFAR10' and self.num_total_users==100:
            instance_model = get_basic_net(net="ResNet20",n_classes=10)
        if self.dataset.upper() == 'CIFAR10' and self.num_total_users==50:
            instance_model = get_basic_net(net="ResNet20",n_classes=10)
        if self.dataset.upper() == 'MNIST':
            instance_model = LeNet5()
        if self.dataset.upper() == 'SVHN':
            #instance_model = AlexNet(args.num_classes)
            #instance_model =  ResNet18()
            instance_model = get_basic_net(net="VGG19-BN", n_classes=10)
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

        #self.split_client = [0, 1, 0]
        #self.split_client = [10, 0, 0]
        self.supervisedloss=[]
        self.unsupervisedloss=[]
        self.globalvariance = []
        self.localvariance = []
        self.globalacc=[]
        self.labelusers=[]
        self.unlabelusers=[]
        self.GlobalLabelDistribution={}

        self.classescount=np.zeros(args.num_classes,dtype=np.int32)
        self.unlabelcount=0
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
                for _, x_o, x_lb, y_lb in lb_dset:
                    self.classescount[y_lb]=self.classescount[y_lb]+1
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}
                loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                                          args.batch_size,
                                                          **local_loader_options(args, 1048576),
                                                          data_sampler=args.train_sampler,
                                                          num_workers=args.num_workers)

                loader_dict['train_ulb'] = None
            elif tag_client_lbulb == 2:
                for x_ulb_idx, x_ulb_w, x_ulb_s in ulb_dset:
                    self.unlabelcount=self.unlabelcount+1
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}

                loader_dict['train_lb'] = None
                loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                                           args.batch_size * args.uratio,
                                                           **local_loader_options(args, 1048576),
                                                           data_sampler=args.train_sampler,
                                                           num_workers=args.num_workers)

            else:
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}

                loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                                          args.batch_size,
                                                          num_iters=1048576,
                                                          data_sampler=args.train_sampler,
                                                          num_workers=args.num_workers)

                loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                                           args.batch_size * args.uratio,
                                                           num_iters=1048576,
                                                           data_sampler=args.train_sampler,
                                                           num_workers=args.num_workers)


            #user = UserFixMatch(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, instance_model, args, logger, device, aggregate_method_server,name_algorithm, name_data_total, name_net, name_optimizer)
            #user = UserReweight_cb(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, instance_model, args, logger, device, aggregate_method_server,name_algorithm, name_data_total, name_net, name_optimizer)
            user = UserReweight_mycb(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, instance_model, args, logger, device, aggregate_method_server,name_algorithm, name_data_total, name_net, name_optimizer,args.num_classes)
            #user = UserReweight_focal(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, instance_model, args, logger, device, aggregate_method_server,name_algorithm, name_data_total, name_net, name_optimizer)
            self.users.append(user)
            if tag_client_lbulb == 0:
                self.labelusers.append(user)
            if tag_client_lbulb == 2:
                self.unlabelusers.append(user)
        for user in self.users:
            user.set_global_distribution(self.classescount)

        self.logger.warning("class distribution")
        self.logger.warning(self.classescount)
        self.logger.warning("unlabelcount")
        self.logger.warning(self.unlabelcount)

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
            dl_obj = SVHN_custom
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
        acc_name = acc_path + f"fedcyon_" + self.flag + "_testacc.npy"
        loss_name = (acc_path + f"fedcyon_" + self.flag +
                     "_idealloss.npy")
        precision_name = acc_path + f"fedcyon_" + self.flag + "_precision.npy"
        recall_name = (acc_path + f"fedcyon_" + self.flag + "_recall.npy")
        AUC_name = acc_path + f"fedcyon_" + self.flag + "_AUC.npy"
        F1_name = (acc_path + f"fedcyon_" + self.flag + "_F1.npy")
        localvariance_name = (acc_path + f"fedcyon_" + self.flag +
                              "_localvariance.npy")
        globalvariance_name = (acc_path + f"fedcyon_" + self.flag +
                               "_globalvariance.npy")
        gradnorm_name = (acc_path + f"fedcyon_" + self.flag +
                         "_gradnorm.npy")
        np.save(acc_name, self.testacc)
        np.save(loss_name, self.idealloss)
        np.save(precision_name, self.precision)
        np.save(recall_name, self.recall)
        np.save(AUC_name, self.AUC)
        np.save(F1_name, self.F1)
        np.save(localvariance_name,self.localvariance)
        np.save(globalvariance_name,self.globalvariance)
        if self.dataset.upper()=='CIFAR100' or self.dataset.upper()=='TINYIMAGENET':
            top5_name=acc_path + f"fedcyon_" + self.flag + "_top5.npy"
            np.save(top5_name, self.top5)
    def saveloss(self):
        save_path = "./lossresult/Balanced_cifar10_federated_GI1600_lr0.001_sample10_k"   
        save_path =save_path+str(self.balanced_k)+"_"+"threshold_"+str(self.ta_threshold)+"_"
        #su_name = save_path + f"Balanced_cifar10_federated_GI800_lr0.03_suloss.npy"
        #un_name = save_path + f"Balanced_cifar10_federated_GI800_lr0.03_unsuloss.npy"
        #acc_name=save_path+f"Balanced_cifar10_federated_GI800_lr0.03_globalacc.npy"
        #su_name = save_path + f"Balanced_cifar10_federated_GI1600_lr0.001_suloss.npy"
        #un_name = save_path + f"Balanced_cifar10_federated_GI1600_lr0.001_unsuloss.npy"
        #acc_name=save_path+f"Balanced_cifar10_federated_GI1600_lr0.001_globalacc.npy"
        #su_name = save_path + f"Balanced_cifar10_federated_GI800_lr0.03_sample10_suloss.npy"
        #un_name = save_path + f"Balanced_cifar10_federated_GI800_lr0.03_sample10_unsuloss.npy"
        #acc_name=save_path+f"Balanced_cifar10_federated_GI800_lr0.03_sample10_globalacc.npy"
        su_name = save_path + f"suloss.npy"
        un_name = save_path + f"unsuloss.npy"
        acc_name=save_path+f"globalacc.npy"
        np.save(su_name, self.supervisedloss)
        np.save(un_name, self.unsupervisedloss)
        np.save(acc_name,self.globalacc)

    def process(self):
        # The complete process of Server.
        for glob_iter in range(self.num_global_iters):
            supervisedloss=0
            unsupervisedloss=0
            localvariance = 0.0
            self.logger.warning(f"-------------Round number: {glob_iter} -------------")
            print(f"-------------Round number: {glob_iter} -------------")

            self.logger.warning("Evaluate global model")
            print(f"Evaluate global model")

            self.evaluate_global_test_dataset(args=self.args)
            self.start_training_round(glob_iter + 1)

            renew_label_users=np.arange(0,self.split_client[0])

            renew_unlabel_users=np.arange(self.split_client[0]+self.split_client[1],self.num_total_users)

            n_label_sam_clients=int(self.split_client[0] * (self.num_subset_users / self.num_total_users))
            if n_label_sam_clients==0:
                n_label_sam_clients=1

            n_unlabel_sam_clients=int(self.split_client[2] * (self.num_subset_users / self.num_total_users))

            sam_label_clients = np.random.choice(
                renew_label_users, n_label_sam_clients, replace=False
            )

            sam_unlabel_clients = np.random.choice(
                renew_unlabel_users, n_unlabel_sam_clients, replace=False
            )

            local_models = {}
            local_models_lb = {}
            local_models_ulb = {}

            avg_loss = Averager()
            all_per_accs = []
            ta_label=[]
            ta_unlabel=[]

            for client in sam_label_clients:
                user_temp = self.users[client]

                local_model, per_accs, loss = user_temp.train(
                    down_model=copy.deepcopy(self.model),
                    args=self.args,
                    glob_iter=glob_iter,
                    logger=self.logger,
                )

                local_models[client] = copy.deepcopy(local_model)
                if self.users[client].tag_client_lbulb==0:
                    local_models_lb[client]= copy.deepcopy(local_model)
                    ta_label.append(self.users[client].ta)
                if self.users[client].tag_client_lbulb==2:
                    local_models_ulb[client]= copy.deepcopy(local_model)
                    ta_unlabel.append(self.users[client].ta)

                supervisedloss=supervisedloss+self.users[client].supervisedloss
                unsupervisedloss=unsupervisedloss+self.users[client].unsupervisedloss
                if self.users[client].maxlocalvariance > localvariance:
                    localvariance = self.users[client].maxlocalvariance
            for client in sam_unlabel_clients:
                user_temp = self.users[client]

                local_model, per_accs, loss = user_temp.train(
                    down_model=copy.deepcopy(self.model),
                    args=self.args,
                    glob_iter=glob_iter,
                    logger=self.logger,
                )

                local_models[client] = copy.deepcopy(local_model)
                if self.users[client].tag_client_lbulb==0:
                    local_models_lb[client]= copy.deepcopy(local_model)
                    ta_label.append(self.users[client].ta)
                if self.users[client].tag_client_lbulb==2:
                    local_models_ulb[client]= copy.deepcopy(local_model)
                    ta_unlabel.append(self.users[client].ta)

                supervisedloss=supervisedloss+self.users[client].supervisedloss
                unsupervisedloss=unsupervisedloss+self.users[client].unsupervisedloss



            if uses_local_epochs(self.args):
                supervisedloss = supervisedloss / len(sam_label_clients)
                unsupervisedloss = unsupervisedloss / len(sam_unlabel_clients)
            else:
                supervisedloss=supervisedloss/self.users[0].local_iterations
                unsupervisedloss=unsupervisedloss/self.users[0].local_iterations
                supervisedloss=supervisedloss/self.split_client[0]
                unsupervisedloss=unsupervisedloss/self.split_client[2]
            self.logger.warning(f"Supervisedloss: {supervisedloss}")
            self.logger.warning(f"Unsupervisedloss: {unsupervisedloss}")
            self.supervisedloss.append(supervisedloss.cpu().detach().numpy())
            self.unsupervisedloss.append(unsupervisedloss.cpu().detach().numpy())

            self.update_global_balanced(
                r=glob_iter,
                global_model=self.model,
                local_models_label=local_models_lb,
                local_models_unlabel=local_models_ulb,
                ta_label=ta_label,
                ta_unlabel=ta_unlabel
            )
            globalvariance = self.calculate_globalvariance(self.model, local_models)
            self.logger.warning(f"LocalVariance: {localvariance}")
            self.logger.warning(f"GlobalVariance: {globalvariance}")
            if isinstance(localvariance, float) == True:
                self.localvariance.append(localvariance)
            else:
                self.localvariance.append(localvariance.cpu().detach().numpy())
            if isinstance(globalvariance, float) == True:
                self.globalvariance.append(globalvariance)
            else:
                self.globalvariance.append(globalvariance.cpu().detach().numpy())
            self.finish_training_round()

        #self.save_results()
        #self.saveloss()
        self.new_result_save()
        # self.save_model()

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

    def update_global_decomposition(self, r, global_model, local_models_label,local_models_unlabel):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        mean_state_dict = {}
        label_ratio=0.5
        unlabel_ratio=0.5

        for name, param in global_model.state_dict().items():
            vs_label = []
            vs_unlabel=[]
            for client in local_models_label.keys():
                vs_label.append(local_models_label[client].state_dict()[name])
            vs_label = torch.stack(vs_label, dim=0)

            for client in local_models_unlabel.keys():
                vs_unlabel.append(local_models_unlabel[client].state_dict()[name])
            vs_unlabel = torch.stack(vs_unlabel, dim=0)

            try:
                mean_value_label = vs_label.mean(dim=0)
                mean_value_unlabel = vs_unlabel.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value_label = (1.0 * vs_label).mean(dim=0).long()
                mean_value_unlabel = (1.0 * vs_unlabel).mean(dim=0).long()
            mean_state_dict[name] = mean_value_label*label_ratio+mean_value_unlabel*unlabel_ratio

        global_model.load_state_dict(mean_state_dict, strict=False)
    def update_global_balanced(self, r, global_model, local_models_label,local_models_unlabel,ta_label,ta_unlabel):
        # global_model = local_models[0]
        # # self.model = local_models[0]
        k=self.balanced_k
        k=k*-1
        mean_state_dict = {}
        label_ratio=0
        for i in ta_label:
            label_ratio=label_ratio+i
        label_ratio=label_ratio/len(ta_label)
        label_ratio=math.exp(label_ratio*k)
        unlabel_ratio=0
        for i in ta_unlabel:
            unlabel_ratio=unlabel_ratio+i
        unlabel_ratio=unlabel_ratio/len(ta_unlabel)
        unlabel_ratio=math.exp(unlabel_ratio*k)
        sum=label_ratio+unlabel_ratio
        label_ratio=label_ratio/sum
        unlabel_ratio=unlabel_ratio/sum
        self.logger.warning("label_ratio")
        self.logger.warning(label_ratio)
        self.logger.warning("unlabel_ratio")
        self.logger.warning(unlabel_ratio)

        for name, param in global_model.state_dict().items():
            vs_label = []
            vs_unlabel=[]
            for client in local_models_label.keys():
                vs_label.append(local_models_label[client].state_dict()[name])
            vs_label = torch.stack(vs_label, dim=0)

            for client in local_models_unlabel.keys():
                vs_unlabel.append(local_models_unlabel[client].state_dict()[name])
            vs_unlabel = torch.stack(vs_unlabel, dim=0)

            try:
                mean_value_label = vs_label.mean(dim=0)
                mean_value_unlabel = vs_unlabel.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value_label = (1.0 * vs_label).mean(dim=0).long()
                mean_value_unlabel = (1.0 * vs_unlabel).mean(dim=0).long()
            mean_state_dict[name] = mean_value_label*label_ratio+mean_value_unlabel*unlabel_ratio

        global_model.load_state_dict(mean_state_dict, strict=False)
    def calculate_globalvariance(self, global_model, local_models):
        currentvariance = 0
        for client in local_models.keys():
            delta_grad = []
            for local_param, global_param in zip(local_models[client].parameters(), global_model.parameters()):
                delta = local_param - global_param
                delta_grad = delta.view(-1) if not len(delta_grad) else torch.cat((delta_grad, delta.view(-1)))
            deltanorm = torch.norm(delta_grad) ** 2
            if currentvariance < deltanorm:
                currentvariance = deltanorm
        return currentvariance

  
