import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from experiments.recording import record_training_loss
# import needed library
import os
import logging
import random
import warnings
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.metrics import *


class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, model, args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_model, name_optimizer):

        # basic setting
        self.id_user = id_user
        self.tag_client_lbulb = tag_client_lbulb
        self.lb_dset = lb_dset
        self.ulb_dset = ulb_dset
        self.loader_dict = loader_dict
        self.model = copy.deepcopy(model)
        self.cpu_client_state = getattr(args, "cpu_client_state", False)
        self.pre_model = None if self.cpu_client_state else copy.deepcopy(model)
        self.args = args
        self.logger = logger
        self.device = device
        if tag_client_lbulb == 0:
            self.lb_dset_samples = len(lb_dset)
            self.ulb_dset_samples = 0
        elif tag_client_lbulb == 2:
            self.lb_dset_samples = 0
            self.ulb_dset_samples = len(ulb_dset)
        else:
            self.lb_dset_samples = len(lb_dset)
            self.ulb_dset_samples = len(ulb_dset)
        self.train_samples = self.lb_dset_samples + self.ulb_dset_samples

        self.aggregate_method_server = args.aggregate_method_server
        self.name_algorithm = args.name_algorithm
        self.runtimes = args.runtimes
        self.num_global_iters = args.num_global_iters
        self.local_epochs = args.local_epochs
        self.num_total_users = args.num_total_users
        self.num_subset_users = args.num_subset_users
        self.gpu = args.gpu
        self.local_iterations=args.local_iterations

        self.train_lb_dataloader = loader_dict['train_lb']
        self.train_ulb_dataloader = loader_dict['train_ulb']
        if loader_dict['train_lb'] is not None and not self.cpu_client_state:
            self.train_lb_iterator=iter(loader_dict['train_lb'])
        if loader_dict['train_ulb'] is not None and not self.cpu_client_state:
            self.train_ulb_iterator=iter(loader_dict['train_ulb'])

        self.local_model = [] if self.cpu_client_state else copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = [] if self.cpu_client_state else copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = [] if self.cpu_client_state else copy.deepcopy(list(self.model.parameters()))

    def record_training_loss(self, loss):
        record_training_loss(self, loss)

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()


    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()



