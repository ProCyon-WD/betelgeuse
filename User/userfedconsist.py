# import needed library
import copy
import os
import logging
import random
import warnings
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

from collections import Counter
from sklearn.metrics import *

from utils.utils import net_builder, get_logger, count_parameters, over_write_args_from_file
from utils.train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup, ce_loss, wd_loss, EMA, Bn_Controller
from User.fixmatch_utils import consistency_loss, Get_Scalar
# from datasets.ssl_dataset import SSL_Dataset, ImageNetLoader
# from datasets.data_utils import get_data_loader

from User.userbase import User
#from loss.soft_supconloss import SoftSupConLoss
#from loss.proto_supconloss import ProtoSupConLoss

class UserFedConsist(User):
    def __init__(self, id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, model, args, logger, device, aggregate_method_server,
                 name_algorithm, name_data_total, name_model, name_optimizer):
        super().__init__(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, model, args, logger, device, aggregate_method_server,
                         name_algorithm, name_data_total, name_model, name_optimizer)

        # args need xxxx
        # args.optim, args.lr, args.momentum, args.weight_decay

        # Init for different args
        # super().__init__
        self.local_epochs = args.local_epochs
        self.T = args.T
        self.p_cutoff = args.p_cutoff


        # momentum update param
        # self.loader = {}
        # self.num_classes = num_classes
        self.ema_m = args.ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        # self.model = model
        self.ema_model = None


        self.t_fn = Get_Scalar(self.T)  # temperature params function
        self.p_fn = Get_Scalar(self.p_cutoff)  # confidence cutoff function
        self.lambda_u = args.ulb_loss_ratio


        # self.tb_log = tb_log
        self.use_hard_label = args.hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        # self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()
        '''
        self.loss_contrast = SoftSupConLoss(temperature=1, base_temperature=1)
        self.loss_contrast_prototype = ProtoSupConLoss(temperature=1, base_temperature=1, num_classes=self.args.num_classes)
        '''
        self.lambda_con = 1.0
        self.print_loss = False  # False

        # dataset of user
        # self.loader_dict = loader_dict
        # self.ulb_dset = dset

        # backbone of user
        # self.model = model

        # Optimizer of user
        # SET Optimizer & LR Scheduler
        ## construct SGD and cosine lr scheduler
        num_train_iter = args.num_global_iters
        optimizer = get_optimizer(self.model, args.name_optimizer, args.lr, args.momentum, args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_train_iter,
                                                    num_warmup_steps=num_train_iter * 0)
        # ## set SGD and cosine lr on FixMatch
        self.optimizer = optimizer
        self.scheduler = scheduler

        if not torch.cuda.is_available():
            raise Exception('ONLY GPU TRAINING IS SUPPORTED')
        else:
            # model.model = torch.nn.DataParallel(model.model).cuda()
            self.model = self.model.cuda(self.device)

        import copy
        self.ema_model = copy.deepcopy(self.model)

        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)


    def train(self, down_model,  args, glob_iter, logger=None):
        # def train(self, args, glob_iter, logger=None):

        # ngpus_per_node = torch.cuda.device_count()

        self.model.load_state_dict(down_model.state_dict())


        self.model.train()

        # prototype setting
        len_feature = 128
        proto_epoch = torch.zeros((self.args.num_classes, len_feature)).cuda(self.args.gpu)
        proto_epoch_count = torch.zeros((self.args.num_classes,)).cuda(self.args.gpu)

        if self.tag_client_lbulb == 0:

            for i in range(self.args.local_epochs):

                scaler = GradScaler()
                amp_cm = autocast if args.amp else contextlib.nullcontext

                for iiii in range(self.local_iterations):
                    _, x_lb_o, x_lb, y_lb=next(self.train_lb_iterator)

                    num_lb = x_lb_o.shape[0]
                    x_lb_o = x_lb_o.cuda(args.gpu)
                    # x_lb_w, x_lb_s1, x_lb_s2 = x_lb_w.cuda(args.gpu), x_lb_s1.cuda(args.gpu), x_lb_s2.cuda(args.gpu)
                    y_lb = y_lb.cuda(args.gpu)
                    inputs = x_lb_o

                    # inference and calculate sup/unsup losses
                    with amp_cm():
                        #logits_x_lb_o, features_x_lb_o = self.model(inputs, ood_test=True)
                        logits_x_lb_o= self.model(inputs)
                        sup_loss = ce_loss(logits_x_lb_o, y_lb, reduction='mean')
                        total_loss = sup_loss

                        if self.print_loss:
                            print("\n")
                            print(f"ID user: {self.id_user}")
                            print(f"sup_loss: {sup_loss}")

                    self.record_training_loss(total_loss)
                    if args.amp:
                        scaler.scale(total_loss).backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        self.optimizer.step()
                    '''
                    if self.args.lr_scheduler:
                        # change lr
                        self.scheduler.step()
                    '''
                    self.ema.update()
                    self.model.zero_grad()



        elif self.tag_client_lbulb == 2:

            # model update
            # for i in range(self.local_epochs):
            for i in range(self.args.local_epochs):

                scaler = GradScaler()
                amp_cm = autocast if args.amp else contextlib.nullcontext

                # for (x_ulb_idx, x_ulb_w, x_ulb_s) in self.loader_dict['train_ulb']:
                # for j in range(args.local_epochs):
                #for (x_ulb_idx, x_ulb_o, x_ulb_w, _, _) in self.loader_dict['train_ulb']:
                for iiii in range(self.local_iterations):
                    x_ulb_idx, x_ulb_o, x_ulb_w=next(self.train_ulb_iterator)
                    # num_lb = x_lb.shape[0]
                    num_ulb = x_ulb_o.shape[0]
                    assert num_ulb == x_ulb_w.shape[0]

                    x_ulb_o = x_ulb_o.cuda(args.gpu)
                    x_ulb_w = x_ulb_w.cuda(args.gpu)

                    # inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    inputs = torch.cat((x_ulb_o, x_ulb_w))

                    with amp_cm():
                        # logits = self.model(inputs)
                        #logits, features = self.model(inputs, ood_test=True)
                        logits= self.model(inputs)
                        # logits_x_lb = logits[:num_lb]
                        # logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                        # sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                        # logits_x_ulb_w, logits_x_ulb_s = logits[:].chunk(2)
                        logits_x_ulb_o, logits_x_ulb_w = logits[:].chunk(2)
                        # features_x_ulb_o, features_x_ulb_w = features[:].chunk(2)

                        T = 1
                        p_cutoff = 0.5

                        unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_w,
                                                                               logits_x_ulb_o,
                                                                               'ce', T, p_cutoff,
                                                                               use_hard_labels=args.hard_label)


                        total_loss = unsup_loss

                        if self.print_loss:
                            print("\n")
                            print(f"ID user: {self.id_user}")
                            # print(f"sup_loss: {sup_loss}")
                            print(f"unsup_loss: {unsup_loss}")
                            # # print(f"loss_lb_contrast_sample: {loss_lb_contrast_sample}")
                            # # print(f"loss_lb_contrast_class_batch: {loss_lb_contrast_class_batch}")
                            # # print(f"loss_ulb_contrast_sample: {loss_ulb_contrast_sample}")
                            # print(f"loss_ulb_contrast_class_batch: {loss_ulb_contrast_class_batch}")
                            # # print(f"loss_lb_contrast_class_prototype: {loss_lb_contrast_class_prototype}")
                            # print(f"loss_ulb_contrast_class_prototype: {loss_ulb_contrast_class_prototype}")
                            # print(f"total_loss: {total_loss}")



                    # parameter updates
                    self.record_training_loss(total_loss)
                    if args.amp:
                        scaler.scale(total_loss).backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        self.optimizer.step()
                    '''
                    if self.args.lr_scheduler:
                        # change lr
                        self.scheduler.step()
                    '''
                    self.ema.update()
                    self.model.zero_grad()

            # change lr
            # self.scheduler.step()

        else:

            # model update
            # for i in range(self.local_epochs):
            for i in range(self.args.local_epochs+1):


                scaler = GradScaler()
                amp_cm = autocast if args.amp else contextlib.nullcontext

                selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
                selected_label = selected_label.cuda(args.gpu)

                classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)

                # for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                #                                                           self.loader_dict['train_ulb']):
                # for (_, x_o, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                #                                                               self.loader_dict['train_ulb']):


                # for j in range(args.local_epochs):
                for (x_lb_idx, x_lb_o, x_lb_w, x_lb_s1, x_lb_s2, y_lb), (x_ulb_idx, x_ulb_o, x_ulb_w, x_ulb_s1, x_ulb_s2) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
                    # 20220818 revise
                    # idx, img_o, img_w, img_s1, img_s2, target
                    # (x_lb_idx, x_lb_o, x_lb_w, x_lb_s1, x_lb_s2, y_lb) = next(iter(self.loader_dict['train_lb']))
                    # (x_ulb_idx, x_ulb_o, x_ulb_w, x_ulb_s1, x_ulb_s2) = next(iter(self.loader_dict['train_ulb']))
                    # (_, x_o, x_lb, y_lb) = next(iter(self.loader_dict['train_lb']))
                    # (x_ulb_idx, x_ulb_w, x_ulb_s) = next(iter(self.loader_dict['train_ulb']))



                    num_lb = x_lb_w.shape[0]
                    num_ulb = x_ulb_w.shape[0]
                    assert num_ulb == x_ulb_s1.shape[0]

                    # 20220818 revise
                    x_lb_w, x_lb_s1, x_lb_s2 = x_lb_w.cuda(args.gpu), x_lb_s1.cuda(args.gpu), x_lb_s2.cuda(args.gpu)
                    y_lb = y_lb.cuda(args.gpu)
                    x_ulb_w, x_ulb_s1, x_ulb_s2 = x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(args.gpu), x_ulb_s2.cuda(args.gpu)


                    # x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
                    # y_lb = y_lb.cuda(args.gpu)

                    pseudo_counter = Counter(selected_label.tolist())
                    if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                        for i in range(args.num_classes):
                            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

                    # inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    inputs = torch.cat((x_lb_w, x_lb_s1, x_lb_s2, x_ulb_w, x_ulb_s1, x_ulb_s2))

                    # inference and calculate sup/unsup losses
                    with amp_cm():

                        logits, features = self.model(inputs, ood_test=True)
                        # logits_x_lb = logits[:num_lb]
                        # logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                        logits_x_lb_w, logits_x_lb_s1, logits_x_lb_s2 = logits[:num_lb*3].chunk(3)
                        logits_x_ulb_w, logits_x_ulb_s1, logits_x_ulb_s2 = logits[num_lb*3:].chunk(3)
                        # features contrastive
                        # features_x_lb = features[:num_lb]
                        # features_x_ulb_w, features_x_ulb_s = logits[num_lb:].chunk(2)
                        features_x_lb_w, features_x_lb_s1, features_x_lb_s2 = features[:num_lb*3].chunk(3)
                        features_x_ulb_w, features_x_ulb_s1, features_x_ulb_s2 = features[num_lb*3:].chunk(3)

                        sup_loss = ce_loss(logits_x_lb_w, y_lb, reduction='mean')

                        # hyper-params for update
                        T = self.t_fn(self.it)
                        p_cutoff = self.p_fn(self.it)

                        unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s1,
                                                                               logits_x_ulb_w,
                                                                               'ce', T, p_cutoff,
                                                                               use_hard_labels=args.hard_label)

                        # if not self.da:
                        #     probs_u_w = torch.softmax(logits_x_lb_w.detach() / T, dim=-1)
                        # else:
                        #     probs_u_w = self._da_pseudo_label(logits_u_w)
                        probs_x_lb_w = torch.softmax(logits_x_lb_w.detach() / T, dim=-1)
                        probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach() / T, dim=-1)

                        # pseudo label and scores for u_w
                        max_probs_lb_w, p_targets_lb_w = torch.max(probs_x_lb_w, dim=-1)
                        max_probs_lb_w = torch.ones_like(max_probs_lb_w)
                        max_probs_ulb_w, p_targets_ulb_w = torch.max(probs_x_ulb_w, dim=-1)
                        # filter out low confidence pseudo label by self.cfg.threshold
                        mask_lb_w = max_probs_lb_w.ge(p_cutoff).float()
                        mask_ulb_w = max_probs_ulb_w.ge(p_cutoff).float()
                        # Lu = (self.loss_u(logits_u_s, p_targets_u, reduction='none') *
                        #       mask).mean()

                        # for supervised contrastive
                        labels_lb_w = p_targets_lb_w
                        labels_ulb_w = p_targets_ulb_w



                        features_x_lb_s1s2 = torch.cat([features_x_lb_s1.unsqueeze(1), features_x_lb_s2.unsqueeze(1)], dim=1)
                        loss_lb_contrast_sample = self.loss_contrast(features_x_lb_s1s2, max_probs_lb_w)
                        if glob_iter < self.args.proto_warmup:
                            loss_lb_contrast_class_batch = 0
                        else:
                            loss_lb_contrast_class_batch = self.loss_contrast(features_x_lb_s1s2, max_probs_lb_w, y_lb)

                        # loss_lb_contrast_class_prototype = self.loss_contrast(features)
                        # loss_lb_contrast_class_prototype = self.loss_contrast_prototype(features_x_lb_s1s2, max_probs_lb_w, y_lb, prototypes=global_proto)

                        features_x_ulb_s1s2 = torch.cat([features_x_ulb_s1.unsqueeze(1), features_x_ulb_s2.unsqueeze(1)], dim=1)
                        loss_ulb_contrast_sample = self.loss_contrast(features_x_ulb_s1s2, max_probs_ulb_w)
                        if glob_iter < self.args.proto_warmup:
                            loss_ulb_contrast_class_batch = 0
                        else:
                            loss_ulb_contrast_class_batch = self.loss_contrast(features_x_ulb_s1s2, max_probs_ulb_w, labels_ulb_w)
                        # loss_ulb_contrast_class_prototype = self.loss_contrast(features)
                        # loss_ulb_contrast_class_prototype = self.loss_contrast_prototype(features_x_ulb_s1s2, max_probs_ulb_w, labels_ulb_w, prototypes=global_proto)

                        if (i == self.args.local_epochs):
                            # cal prototype
                            proto_epoch, proto_epoch_count = self.update_local_prototype(features=features_x_lb_s1s2.detach(),
                                                                                         labels=y_lb,
                                                                                         mask=mask_lb_w,
                                                                                         num_classes=self.args.num_classes,
                                                                                         device=self.args.gpu,
                                                                                         proto_epoch=proto_epoch,
                                                                                         proto_epoch_count=proto_epoch_count)
                            proto_epoch, proto_epoch_count = self.update_local_prototype(features=features_x_ulb_s1s2.detach(),
                                                                                         labels=labels_ulb_w,
                                                                                         mask=mask_ulb_w,
                                                                                         num_classes=self.args.num_classes,
                                                                                         device=self.args.gpu,
                                                                                         proto_epoch=proto_epoch,
                                                                                         proto_epoch_count=proto_epoch_count)
                            continue

                        if glob_iter < self.args.proto_warmup:
                            loss_lb_contrast_class_prototype = 0
                            loss_ulb_contrast_class_prototype = 0
                        else:
                            loss_lb_contrast_class_prototype = self.loss_contrast_prototype(features_x_lb_s1s2,
                                                                                            max_probs_lb_w, y_lb,
                                                                                            prototypes=global_proto.detach())
                            loss_ulb_contrast_class_prototype = self.loss_contrast_prototype(features_x_ulb_s1s2,
                                                                                             max_probs_ulb_w, labels_ulb_w,
                                                                                             prototypes=global_proto.detach())

                        # total_loss = sup_loss + self.lambda_u * unsup_loss

                        if self.args.loss_mode == 0:
                            total_loss = sup_loss + self.lambda_u * unsup_loss
                        elif self.args.loss_mode == 1:
                            total_loss = sup_loss + self.lambda_u * unsup_loss \
                                         + self.lambda_con * loss_lb_contrast_sample \
                                         + self.lambda_con * loss_ulb_contrast_sample
                        elif self.args.loss_mode == 2:
                            total_loss = sup_loss + self.lambda_u * unsup_loss \
                                         + self.lambda_con * loss_lb_contrast_class_batch \
                                         + self.lambda_con * loss_ulb_contrast_class_batch
                        elif self.args.loss_mode == 3:
                            total_loss = sup_loss + self.lambda_u * unsup_loss \
                                         + loss_lb_contrast_class_prototype + loss_ulb_contrast_class_prototype
                        elif self.args.loss_mode == 4:
                            total_loss = sup_loss + self.lambda_u * unsup_loss \
                                         + self.lambda_con * loss_lb_contrast_sample + self.lambda_con * loss_lb_contrast_class_batch \
                                         + self.lambda_con * loss_ulb_contrast_sample + self.lambda_con * loss_ulb_contrast_class_batch
                        elif self.args.loss_mode == 5:
                            total_loss = sup_loss + self.lambda_u * unsup_loss \
                                         + self.lambda_con * loss_lb_contrast_sample \
                                         + self.lambda_con * loss_ulb_contrast_sample \
                                         + loss_lb_contrast_class_prototype + loss_ulb_contrast_class_prototype
                        elif self.args.loss_mode == 6:
                            total_loss = sup_loss + self.lambda_u * unsup_loss \
                                         + self.lambda_con * loss_lb_contrast_class_batch \
                                         + self.lambda_con * loss_ulb_contrast_class_batch \
                                         + loss_lb_contrast_class_prototype + loss_ulb_contrast_class_prototype

                        elif self.args.loss_mode == 7:
                            total_loss = sup_loss + self.lambda_u * unsup_loss \
                                         + self.lambda_con * loss_lb_contrast_sample + self.lambda_con * loss_lb_contrast_class_batch \
                                         + self.lambda_con * loss_ulb_contrast_sample + self.lambda_con * loss_ulb_contrast_class_batch \
                                         + loss_lb_contrast_class_prototype + loss_ulb_contrast_class_prototype

                        # total_loss = sup_loss + self.lambda_u * unsup_loss \
                        #              + self.lambda_con * loss_lb_contrast_sample + self.lambda_con * loss_lb_contrast_class_batch \
                        #              + self.lambda_con * loss_ulb_contrast_sample + self.lambda_con * loss_ulb_contrast_class_batch \
                        #              + loss_lb_contrast_class_prototype + loss_ulb_contrast_class_prototype

                        if self.print_loss:
                            print("\n")
                            print(f"ID user: {self.id_user}")
                            print(f"sup_loss: {sup_loss}")
                            print(f"unsup_loss: {unsup_loss}")
                            print(f"loss_lb_contrast_sample: {loss_lb_contrast_sample}")
                            print(f"loss_lb_contrast_class_batch: {loss_lb_contrast_class_batch}")
                            print(f"loss_ulb_contrast_sample: {loss_ulb_contrast_sample}")
                            print(f"loss_ulb_contrast_class_batch: {loss_ulb_contrast_class_batch}")
                            print(f"loss_lb_contrast_class_prototype: {loss_lb_contrast_class_prototype}")
                            print(f"loss_ulb_contrast_class_prototype: {loss_ulb_contrast_class_prototype}")
                            print(f"total_loss: {total_loss}")


                    # parameter updates
                    self.record_training_loss(total_loss)
                    if args.amp:
                        scaler.scale(total_loss).backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        self.optimizer.step()

                    if self.args.lr_scheduler:
                        # change lr
                        self.scheduler.step()
                    self.ema.update()
                    self.model.zero_grad()
                    # # inference and calculate sup/unsup losses
                    # with amp_cm():
                    #     logits = self.model(inputs)
                    #     logits_x_lb = logits[:num_lb]
                    #     logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                    #     sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                    #
                    #     # hyper-params for update
                    #     T = self.t_fn(self.it)
                    #     p_cutoff = self.p_fn(self.it)
                    #
                    #     unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s,
                    #                                                            logits_x_ulb_w,
                    #                                                            'ce', T, p_cutoff,
                    #                                                            use_hard_labels=args.hard_label)
                    #
                    #     total_loss = sup_loss + self.lambda_u * unsup_loss
                    #
                    # # parameter updates
                    # if args.amp:
                    #     scaler.scale(total_loss).backward()
                    #     if (args.clip > 0):
                    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    #     scaler.step(self.optimizer)
                    #     scaler.update()
                    # else:
                    #     total_loss.backward()
                    #     if (args.clip > 0):
                    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                    #     self.optimizer.step()
                    #
                    # if self.args.lr_scheduler:
                    #     # change lr
                    #     self.scheduler.step()
                    # self.ema.update()
                    # self.model.zero_grad()

                    # print(f"ID_user, {self.id_user}, Global: {glob_iter}, lr: {self.scheduler.get_last_lr()}")
                    # print(f"ID_user, {self.id_user}, Global: {glob_iter}, lr: {self.optimizer.param_groups[0]['lr']}")
            # change lr
            # self.scheduler.step()


            '''
            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000
            '''

            # eval_dict = self.evaluate(args=args)
            # eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
            # return
        self.scheduler.step()
        per_accs = 0
        loss = total_loss.item()
        return self.model, per_accs, loss

    def update_local_prototype(self, features, labels, mask, num_classes, device, proto_epoch, proto_epoch_count):
        pos = torch.nonzero(mask == 1.).cuda(device)
        # num_classes = self.args.num_classes
        proto_batch_count = torch.zeros((num_classes,)).cuda(device)
        # out of epoch
        # proto_epoch_count = torch.zeros((num_classes,)).cuda(self.args.gpu)
        proto_batch_label = [[] for _ in range(num_classes)]
        for index in pos:
            one_label = labels[index]
            proto_batch_label[one_label].append(index)
            proto_batch_count[one_label] += 1
        proto_batch = torch.zeros((num_classes, features.shape[-1])).cuda(device)
        for k in range(num_classes):
            if proto_batch_label[k]:
                current_features = features[proto_batch_label[k], :, :]
                proto_batch[k, :] = torch.mean(torch.mean(current_features, dim=0), dim=0)
                proto_epoch[k, :] = (proto_epoch_count[k] / torch.clamp((proto_epoch_count[k] + proto_batch_count[k]), min=1e-9)) * proto_epoch[k, :] + \
                                    (proto_batch_count[k] / torch.clamp((proto_epoch_count[k] + proto_batch_count[k]), min=1e-9)) * proto_batch[k, :]


        proto_epoch_count = proto_epoch_count + proto_batch_count

        return proto_epoch, proto_epoch_count

    def update_local_prototype_w(self, features, labels, mask, num_classes, device, proto_epoch, proto_epoch_count):
        pos = torch.nonzero(mask == 1.).cuda(device)
        # num_classes = self.args.num_classes
        proto_batch_count = torch.zeros((num_classes,)).cuda(device)
        # out of epoch
        # proto_epoch_count = torch.zeros((num_classes,)).cuda(self.args.gpu)
        proto_batch_label = [[] for _ in range(num_classes)]
        for index in pos:
            one_label = labels[index]
            proto_batch_label[one_label].append(index)
            proto_batch_count[one_label] += 1
        proto_batch = torch.zeros((num_classes, features.shape[-1])).cuda(device)
        for k in range(num_classes):
            if proto_batch_label[k]:
                current_features = features[proto_batch_label[k], :, :]
                proto_batch[k, :] = torch.mean(torch.mean(current_features, dim=0), dim=0)
                proto_epoch[k, :] = (proto_epoch_count[k] / torch.clamp((proto_epoch_count[k] + proto_batch_count[k]), min=1e-9)) * proto_epoch[k, :] + \
                                    (proto_batch_count[k] / torch.clamp((proto_epoch_count[k] + proto_batch_count[k]), min=1e-9)) * proto_batch[k, :]


        proto_epoch_count = proto_epoch_count + proto_batch_count

        return proto_epoch, proto_epoch_count

    def contrast_left_out(self, max_probs):
        contrast_mask = max_probs.ge(self.args.p_cutoff).float()
        contrast_mask2 = torch.clone(contrast_mask)
        contrast_mask2[contrast_mask == 0] = -1
        select_elements = torch.eq(
            contrast_mask2.reshape([-1, 1]), contrast_mask.reshape([-1, 1]).T).float()
        select_elements += torch.eye(contrast_mask.shape[0]).cuda(self.args.gpu)
        select_elements[select_elements > 1] = 1
        select_matrix = torch.ones(
            contrast_mask.shape[0]).cuda(self.args.gpu) * select_elements
        return select_matrix

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=5000,
                        help='evaluation frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    # main(args)
