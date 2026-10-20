import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics, LargeMargin
from torch.utils.data import DataLoader
from utils.model_utils import read_data, read_user_data
import copy
from datautils.utils import *
import torch.nn as nn
from datautils.datasets import *
from datautils.datasets import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom, SVHN_custom, \
    FashionMNIST_truncated, CustomTensorDataset, CelebA_custom, FEMNIST, Generated, genData


class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate,
                 num_glob_iters, local_epochs, num_users, times, select_users_num, datadir, beta, partition):

        # Set up the main attributes
        self.select_users_num = select_users_num
        self.device = device
        self.dataset = dataset
        self.datadir = datadir
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_glob_loss, self.rs_glob_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_train_loss = []
        self.times = times
        self.rs_glob_Jacobian_loss = []
        self.rs_glob_RiskJacobian_loss = []
        self.server_LargeMargin = []
        self.epoch_client_LargeMargin = []
        self.partition = partition
        self.users_trainset = []
        self.users_testset = []
        self.server_test_dl = []
        self.server_train_dl = []
        self.lv_avg_list = []
        self.gv_avg_list = []
        self.gv_max_list = []
        self.beta = beta
        self.loss = nn.CrossEntropyLoss()
        self.data_partition()

    def data_partition(self):
        partition = partition_data(dataset=self.dataset, datadir=self.datadir, logdir="/data/log",
                                   partition=self.partition, n_parties=self.num_users, beta=self.beta)
        net_dataidx_map = partition[4]
        if self.dataset == "cifar10":
            dl_obj = CIFAR10_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.dataset == 'stl10':
            dl_obj = STL10_truncated

        server_test_ds = dl_obj(self.datadir, train=False, transform=transform_test, download=True)
        server_train_ds = dl_obj(self.datadir, train=True, transform=transform_train, download=True)
        self.server_test_dl = data.DataLoader(dataset=server_test_ds, batch_size=100, shuffle=False, drop_last=False)
        self.server_train_dl = data.DataLoader(dataset=server_train_ds, batch_size=100, shuffle=False, drop_last=False)

        for i in range(self.num_users):
            train_ds = get_dataloader(self.dataset, self.datadir, net_dataidx_map[i])
            self.users_trainset.append(train_ds)
            # self.users_testset.append(test_ds)

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

        return copy.deepcopy(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        # if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):

        alg = self.algorithm
        alg = alg + "_" + str(self.num_users) + "_" + self.partition + "_" + str(self.beta)
        torch.save(self.model, os.path.join("ModelResult", self.dataset, alg + ".pt"))

    def load_model(self, model_path):
        assert (os.path.exists(model_path))
        print("model loaded!")
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if (self.select_users_num == len(self.users)):
            print("All users are selected")
            self.selected_users = self.users

        num_users = min(self.select_users_num, len(self.users))
        # selected_index=np.random.choice(self.users, self.select_users_num, replace=False)
        # print(selected_index)
        self.selected_users = np.random.choice(self.users, num_users, replace=False)
        # return np.random.choice(self.users, self.select_users_num, replace=False) #, p=pk)

    # Save loss, accurancy to h5 fiel
    def save_results(self):

        alg = self.algorithm
        alg = alg + "_" + str(self.num_users) + "_" + self.partition + "_" + str(self.beta)
        alg_acc = alg + "_testacc"
        alg_loss = alg + "_trainloss"
        alg_lv_avg = alg + "_lvavg"
        alg_gv_avg = alg + "_gvavg"
        alg_gv_max = alg + "_gvamax"

        base_dir = "/data/wudi/betelgeuse"
        dir_item = ["generalization", "convergence", "variance"]
        for item in dir_item:
            path = os.path.join(base_dir, "results", self.dataset, item)
            os.makedirs(path, exist_ok=True)
        np.save(os.path.join(base_dir, "results", self.dataset, "generalization", alg_acc), self.rs_glob_acc)
        np.save(os.path.join(base_dir, "results", self.dataset, "convergence", alg_loss), self.rs_train_loss)
        np.save(os.path.join(base_dir, "results", self.dataset, "variance", alg_lv_avg), self.lv_avg_list)
        np.save(os.path.join(base_dir, "results", self.dataset, "variance", alg_gv_avg), self.gv_avg_list)
        np.save(os.path.join(base_dir, "results", self.dataset, "variance", alg_gv_max), self.gv_max_list)

    def save_accandloss(self):
        alg = self.algorithm
        alg = alg + "_" + str(self.num_users) + "_" + self.partition + "_" + str(self.beta)
        alg_acc = alg + "_testacc"
        alg_loss = alg + "_trainloss"
        base_dir = "/data/wudi/betelgeuse"
        dir_item = ["generalization", "convergence", "variance"]
        for item in dir_item:
            path = os.path.join(base_dir, "results", self.dataset, item)
            os.makedirs(path, exist_ok=True)
        np.save(os.path.join(base_dir, "results", self.dataset, "generalization", alg_acc), self.rs_glob_acc)
        np.save(os.path.join(base_dir, "results", self.dataset, "convergence", alg_loss), self.rs_train_loss)
    def save_variance(self):

        alg = self.algorithm
        alg = alg + "_" + str(self.num_users) + "_" + self.partition + "_" + str(self.beta)
        alg_lv_avg = alg + "_lvavg"
        alg_gv_avg = alg + "_gvavg"
        alg_gv_max = alg + "_gvamax"

        base_dir = "/data/wudi/betelgeuse"
        dir_item = ["generalization", "convergence", "variance"]
        for item in dir_item:
            path = os.path.join(base_dir, "results", self.dataset, item)
            os.makedirs(path, exist_ok=True)
        np.save(os.path.join(base_dir, "results", self.dataset, "variance", alg_lv_avg), self.lv_avg_list)
        np.save(os.path.join(base_dir, "results", self.dataset, "variance", alg_gv_avg), self.gv_avg_list)
        np.save(os.path.join(base_dir, "results", self.dataset, "variance", alg_gv_max), self.gv_max_list)

    def server_test(self):
        '''tests self.latest_model on given clients
        '''
        self.model.eval()
        correct = 0
        total = 0
        for x, y in self.server_test_dl:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            _, predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            # test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        return 100. * correct / total

    def server_train_error_and_loss(self):

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.server_train_dl):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return test_loss / (batch_idx + 1), 100. * correct / total


    def evaluate(self):

        # stats = self.test()
        # glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        glob_acc = self.server_test()
        train_loss, train_acc = self.server_train_error_and_loss()
        # train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        # train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)
    def get_params(self):
        param=[]
        local_dict=self.model.cpu().state_dict()
        for key in local_dict:
            param = local_dict[key].data.view(-1) if not len(param) else torch.cat((param, local_dict[key].data.view(-1)))
        return param
