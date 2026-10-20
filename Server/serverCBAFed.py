import os
from experiments.paths import log_directory, metrics_directory
from datautils.partition_store import load_partition, partition_directory, partition_exists, save_partition

import torch
import copy
from User.userfixmatch import UserFixMatch
from User.usercbafed import UserCBAFed
from Server.serverbase import Server
from datautils.data_utils.data_utils import get_data_loader
import torch.utils.data as data
from datautils.data_utils.ssl_dataset import SSL_Dataset, get_transforms
from datautils.data_utils.dataset import BasicDataset,StrongDataset
from datautils.datasets import SVHN_custom, CIFAR100_truncated,ImageFolder_custom
from datautils.utils import *
# from datautils.utils import *
import torch.nn as nn
# from utils.model_utils import read_data, read_user_data
import numpy as np
from Models.resnet import *
from Models.lenet import *
from Models.basic_nets import *

from utils.utils import Averager

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
mean['mnist'] = [0.5]
std['mnist'] = [0.5]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
std['svhn'] = [0.1751, 0.1771, 0.1744]


class CBAFed(Server):
    def __init__(self, args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_net,
                 name_optimizer,curtimes, dataset, datadir, partition, num_users, beta):
        super().__init__(args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_net,
                         name_optimizer,
                         curtimes, dataset, datadir, partition, num_users, beta)
        self.save_path_name = args.save_path_name

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
        print(path)
        total_data = load_partition(path)
        # for i in range(0, self.num_users):
        #     print(total_data[i]['data'][0].shape)
        # data_ssl_total = SSL_Dataset(args, alg=self.name_algorithm, data_dir=args.data_dir,
        #                              name=args.dataset, name_data_total=args.name_data_total,
        #                              num_classes=args.num_classes)
        ################################################################################################################
        # partition = partition_data(dataset=self.dataset, datadir=self.datadir, logdir=log_directory(),
        #                            partition=self.partition, n_parties=self.num_users, beta=self.beta)
        # net_dataidx_map = partition[4]
        # if args.dataset.upper()== "CIFAR10":
        #     dl_obj = CIFAR10_truncated
        #     transform_train = transforms.Compose([
        #         transforms.ToTensor(),
        #     ])
        #     # data prep for test set
        #     transform_test = transforms.Compose([
        #         transforms.ToTensor(),
        #     ])
        # elif args.dataset.upper() == 'STL10':
        #     dl_obj = STL10_truncated
        #
        self.server_train_transform, self.server_test_trainsform = get_transforms(self.dataset)
        # self.server_test_ds = dl_obj(self.datadir, train=False, transform=transform_test, download=True)
        # self.server_train_ds = dl_obj(self.datadir, train=True, transform=transform_train, download=True)
        # self.server_test_dl = data.DataLoader(dataset=self.server_test_ds, batch_size=100, shuffle=False, drop_last=False)
        # self.server_train_dl = data.DataLoader(dataset=self.server_train_ds, batch_size=100, shuffle=False, drop_last=False)
        ################################################################################################################
        self.newcurtime = str(curtimes)
        if self.dataset.upper() == 'CIFAR10':
            instance_model = ResNet18()
        if self.dataset.upper() == 'MNIST':
            instance_model = LeNet5()
        if self.dataset.upper() == 'SVHN':
            #instance_model = AlexNet(args.num_classes)
            #instance_model =  ResNet18()
            instance_model = get_basic_net(net="VGG19-BN", n_classes=10)
        if self.dataset.upper() == 'STL10':
            instance_model = get_basic_net(net="ResNet20",n_classes=10)

        if self.dataset.upper() == 'CIFAR100':
            instance_model = get_basic_net(net="ResNet56", n_classes=100)
        if self.dataset.upper() == 'TINYIMAGENET':
            instance_model = get_basic_net(net="ResNet56", n_classes=200)
            #instance_model = get_basic_net(net="ResNet20",n_classes=100)
            #instance_model=get_basic_net(net="VGG19-BN",n_classes=100)
        torch.cuda.set_device(args.gpu)

        instance_model = instance_model.cuda(args.gpu)
        self.model = copy.deepcopy(instance_model)

        self.supervisedloss = []
        self.unsupervisedloss = []
        self.globalvariance = []
        self.localvariance = []
        # assign data and model to generate all users using instance
        self.sup_label = torch.zeros(args.num_classes)
        for i in range(self.num_total_users):
            # train_ds = get_dataloader(self.dataset, self.datadir, net_dataidx_map[i])

            id_user = i
            # lb_dset, ulb_dset = data_ssl_total.get_per_user_data(id_user, onehot=False)
            lb_dset, ulb_dset = self.get_per_user_data(total_data, i, onehot=False)

            if id_user < self.split_client[0]:
                tag_client_lbulb = 0
            elif id_user < (self.split_client[0] + self.split_client[1]):
                tag_client_lbulb = 1
            else:
                tag_client_lbulb = 2
            if tag_client_lbulb == 0:
                for _, x_o, x_lb, y_lb in lb_dset:
                    #print(y_lb)
                    self.sup_label[y_lb] = self.sup_label[y_lb] + 1
                self.logger.warning(self.sup_label)
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}
                loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                                          args.batch_size,
                                                          num_iters=1048576,
                                                          data_sampler=args.train_sampler,
                                                          num_workers=args.num_workers)

                loader_dict['train_ulb'] = None
                #extra_loader=None

            elif tag_client_lbulb == 2:
                loader_dict = {}
                dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset}

                loader_dict['train_lb'] = None
                '''
                loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                                           args.batch_size * args.uratio,
                                                           num_iters=1048576,
                                                           data_sampler=args.train_sampler,
                                                           num_workers=args.num_workers)
                extra_loader=DataLoader(dataset=dset_dict['train_ulb'], batch_size=64,
                                                            shuffle=True,
                                                            num_workers=0,
                                                            drop_last=False)
                '''
                loader_dict['train_ulb']=data.DataLoader(dataset=dset_dict['train_ulb'],
                                                         batch_size=args.batch_size,
                                                         drop_last=False,
                                                         shuffle=True)


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


            user = UserCBAFed(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, instance_model, args, logger,
                                device, aggregate_method_server, name_algorithm, name_data_total, name_net,
                                name_optimizer)
            # user = UserReweight_mycb(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, instance_model, args, logger, device, aggregate_method_server,name_algorithm, name_data_total, name_net, name_optimizer,args.num_classes)
            self.users.append(user)

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
            dl_obj = CIFAR100_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset.upper() == 'TINYIMAGENET':
            dl_obj = ImageFolder_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

        #server_test_ds = dl_obj(self.datadir, train=False, transform=transform_test, download=True)
        
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
        ulb_dset= StrongDataset(self.dataset,ulb_data,ulb_target,is_labeled=False,is_testing=False,pre_w=40,pre_h=40)
        return lb_dset, ulb_dset

    def new_result_save(self):
        acc_path = metrics_directory(self.args)
        #acc_name = acc_path + f"fedavg_" + self.save_path_name + "_" + str(self.newcurtime) + "_" + self.flag + "_testacc.npy"
        #loss_name = acc_path + f"fedavg_" + self.save_path_name + "_" + str(self.newcurtime) + "_" + self.flag + "_idealloss.npy"
        #localvariance_name = acc_path + f"fedavg_" + self.save_path_name + "_" + str(self.newcurtime) + "_" + self.flag + "_localvariance.npy"
        #globalvariance_name = acc_path + f"fedavg_" + self.save_path_name + "_" + str(
            #self.newcurtime) + "_" + self.flag + "_globalvariance.npy"
        acc_name = acc_path + f"cbafed_"  +self.flag + "_testacc.npy"
        loss_name = (acc_path + f"cbafed_" +self.flag +
                     "_idealloss.npy")
        precision_name = acc_path + f"cbafed_" + self.flag + "_precision.npy"
        recall_name = (acc_path + f"cbafed_" + self.flag +"_recall.npy")
        AUC_name = acc_path + f"cbafed_" + self.flag + "_AUC.npy"
        F1_name = (acc_path + f"cbafed_" + self.flag +"_F1.npy")

        np.save(acc_name, self.testacc)
        np.save(loss_name, self.idealloss)
        np.save(precision_name, self.precision)
        np.save(recall_name, self.recall)
        np.save(AUC_name, self.AUC)
        np.save(F1_name, self.F1)
        if self.dataset.upper()=='CIFAR100' or self.dataset.upper()=='TINYIMAGENET':
            top5_name=acc_path + f"cbafed_" + self.flag + "_top5.npy"
            np.save(top5_name, self.top5)



    def process(self):
        # The complete process of Server.
        self.start_training_round(0, warmup=True)
        for i in range(1):
            self.logger.warning("Initializing pseudo labels")
            w_l = []
            n_l = []
            for i in range(self.split_client[0]):
                local_model, num = self.users[i].train(
                    down_model=copy.deepcopy(self.model),
                    args=self.args,
                    glob_iter=0,
                    logger=self.logger,
                )
                w_l.append(local_model)
                n_l.append(num)
        w= FedAvg(w_l,n_l)
        record_w = copy.deepcopy(w)
        self.model.load_state_dict(w)
        self.finish_training_round()
        T_base = 0.84

        T_lower = 0.03

        T_higher = 0.1
        T_upper = 0.85
        temp_sup_label = copy.deepcopy(self.sup_label)
        temp_sup_label = (temp_sup_label / sum(temp_sup_label)) * (self.args.num_classes / 10)
        temp_sup_label = temp_sup_label
        class_confident = temp_sup_label + T_base - temp_sup_label.std()
        class_confident[class_confident >= T_upper] = T_upper


        for glob_iter in range(self.num_global_iters):
            local_label = torch.zeros(self.args.num_classes)
            local_w = []
            local_num = []
            for user in self.users:
                user.set_class_confident(class_confident)
            self.logger.warning("confident thereshold")
            self.logger.warning(class_confident)
            supervisedloss = 0
            unsupervisedloss = 0
            localvariance = 0
            self.logger.warning(f"-------------Round number: {glob_iter} -------------")
            # print(f"-------------Round number: {glob_iter} -------------")

            self.logger.warning("Evaluate global model")
            # print(f"Evaluate global model")
            # Test global model
            self.evaluate_global_test_dataset(args=self.args)
            self.start_training_round(glob_iter + 1)
            # Sample clients
            renew_num_users = np.arange(0, self.num_total_users)
            n_sam_clients = int(len(renew_num_users) * (self.num_subset_users / self.num_total_users))
            sam_clients = np.random.choice(
                renew_num_users, n_sam_clients, replace=False
            )
            # Initialization of local model array

            avg_loss = Averager()
            all_per_accs = []
            # Local trianing of Sampled clients
            for client in sam_clients:
                if client < self.split_client[0]:
                    user_temp = self.users[client]
                    local_model, num = user_temp.train(
                        down_model=copy.deepcopy(self.model),
                        args=self.args,
                        glob_iter=glob_iter,
                        logger=self.logger,
                    )
                    # Collect local models for calculation of global variance
                    local_w.append(local_model)
                    local_num.append(num)
                else:
                    user_temp = self.users[client]
                    local_model, num,train_label = user_temp.train(
                        down_model=copy.deepcopy(self.model),
                        args=self.args,
                        glob_iter=glob_iter,
                        logger=self.logger,
                    )
                    # Collect local models for calculation of global variance
                    local_w.append(w)
                    local_num.append(num)
                    local_label = local_label + train_label
            local_label = local_label + self.sup_label
            local_label = (local_label / sum(local_label)) * (self.args.num_classes / 10)
            class_confident = local_label + T_base - local_label.std()
            class_confident[class_confident >= T_upper] = T_upper
            w = FedAvg(local_w, local_num)
            '''
            if glob_iter>0 and glob_iter%5==0:
                self.logger.warning('res weight connection 5 epoch')
                w_l = [record_w, copy.deepcopy(w)]
                n_l = [1., 1.]
                #print(n_l)
                w = FedAvg(w_l, n_l)
                record_w = copy.deepcopy(w)
            '''
            self.model.load_state_dict(w)
            self.finish_training_round()



        # self.save_results()
        self.new_result_save()
        # self.save_model()

    def update_global(self, r, global_model, local_models):
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
def FedAvg(w, client_freq):

    w_avg = copy.deepcopy(w[0])
    num_takepart = len(w)
    ratio_takepart = sum(client_freq[:num_takepart])
    ratio = [freq / ratio_takepart for freq in client_freq[:num_takepart]]

    for net_id in range(num_takepart):
        if net_id == 0:
            for key in w[net_id]:
                w_avg[key] = w[net_id][key] * ratio[net_id]
        else:
            for key in w[net_id]:
                w_avg[key] += w[net_id][key] * ratio[net_id]
    return w_avg
