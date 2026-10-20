# import needed library
import contextlib

import torch
import torch.nn.parallel
from torch.cuda.amp import autocast, GradScaler

from collections import Counter
from sklearn.metrics import *

from utils.utils import net_builder, count_parameters, over_write_args_from_file
from utils.train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup, ce_loss, wd_loss, EMA, Bn_Controller
from User.fixmatch_utils import consistency_loss, Get_Scalar
# from datasets.ssl_dataset import SSL_Dataset, ImageNetLoader
# from datasets.data_utils import get_data_loader

from User.userbase import User


class UserFixMatch(User):
    def __init__(self, id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, model, args, logger, device, aggregate_method_server,
                 name_algorithm, name_data_total, name_model, name_optimizer):
        super().__init__(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, model, args, logger, device, aggregate_method_server,
                         name_algorithm, name_data_total, name_model, name_optimizer)
        # Init for different args
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
        self.basic_lr=args.lr
        self.bn_controller = Bn_Controller()
        num_train_iter = args.num_global_iters
        optimizer = get_optimizer(self.model, args.name_optimizer, args.lr, args.momentum, args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_train_iter,
                                                    num_warmup_steps=num_train_iter * 0)
        ## set SGD and cosine lr on FixMatch
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
        self.supervisedloss=0
        self.unsupervisedloss=0
        self.ta=0
        self.ta_threshold=0.95
        self.maxlocalvariance=0
    def optimizer_update_bysquare(self,glob_iter):
        # update learning rate
        if glob_iter % 5 ==0:
            for params in self.optimizer.param_groups:
                params['lr'] = (1 / 1 + glob_iter ** 0.5) * self.basic_lr



    def train(self, down_model, args, glob_iter, logger=None):

        self.model.load_state_dict(down_model.state_dict())
        self.supervisedloss=0
        self.unsupervisedloss=0
        self.maxlocalvariance=0
        grads=[]
        self.ta=0
        self.model.train()

        if self.tag_client_lbulb == 0:
            for i in range(1):
                scaler = GradScaler()
                amp_cm = autocast if args.amp else contextlib.nullcontext
                for iiii in range(self.local_iterations):
                    _, x_o, x_lb, y_lb=next(self.train_lb_iterator)
                    num_lb = x_lb.shape[0]
                    x_o = x_o.cuda(args.gpu)
                    x_lb = x_lb.cuda(args.gpu)
                    y_lb = y_lb.cuda(args.gpu)
                    inputs = x_o
                    with amp_cm():
                        logits = self.model(inputs)
                        logits_x_lb = logits[:num_lb]
                        self.calculate_ta(logits_x_lb,self.local_iterations,num_lb)
                        sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                        total_loss = sup_loss
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
                    self.ema.update()
                    model_grad=[]
                    for param in self.model.parameters():
                        model_grad=param.grad.detach().view(-1) if not len(model_grad) else torch.cat((model_grad,param.grad.detach().view(-1)))
                    grads.append(model_grad)
                    self.optimizer.step()
                    self.model.zero_grad()

                    self.supervisedloss=self.supervisedloss+sup_loss
        elif self.tag_client_lbulb == 2:
            for i in range(1):
                scaler = GradScaler()
                amp_cm = autocast if args.amp else contextlib.nullcontext
                selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
                selected_label = selected_label.cuda(args.gpu)
                classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)

                for iiii in range(self.local_iterations):
                    x_ulb_idx, x_ulb_w, x_ulb_s=next(self.train_ulb_iterator)

                    num_ulb = x_ulb_w.shape[0]
                    assert num_ulb == x_ulb_s.shape[0]
                    x_ulb_w, x_ulb_s = x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)

                    pseudo_counter = Counter(selected_label.tolist())
                    if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                        for i in range(args.num_classes):
                            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                    inputs = torch.cat((x_ulb_w, x_ulb_s))

                    with amp_cm():
                        logits = self.model(inputs)
                        logits_x_ulb_w, logits_x_ulb_s = logits[:].chunk(2)
                        self.calculate_ta(logits_x_ulb_w,self.local_iterations,num_ulb)
                        T = self.t_fn(self.it)
                        p_cutoff = self.p_fn(self.it)

                        unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s,
                                                                               logits_x_ulb_w,
                                                                               'ce', T, p_cutoff,
                                                                               use_hard_labels=args.hard_label)
                        total_loss = self.lambda_u*unsup_loss
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
                    self.ema.update()
                    model_grad=[]
                    for param in self.model.parameters():
                        model_grad=param.grad.detach().view(-1) if not len(model_grad) else torch.cat((model_grad,param.grad.detach().view(-1)))
                    grads.append(model_grad)
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.ema.update()
                    self.model.zero_grad()
                    self.unsupervisedloss=self.unsupervisedloss+unsup_loss

        else:
            for i in range(1):
                scaler = GradScaler()
                amp_cm = autocast if args.amp else contextlib.nullcontext
                selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
                selected_label = selected_label.cuda(args.gpu)
                classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)

                for iiii in range(self.local_iterations):
                    _, x_o, x_lb, y_lb=next(self.train_lb_iterator)
                    x_ulb_idx, x_ulb_w, x_ulb_s=next(self.train_ulb_iterator)
                    num_lb = x_lb.shape[0]
                    num_ulb = x_ulb_w.shape[0]
                    assert num_ulb == x_ulb_s.shape[0]
                    x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
                    y_lb = y_lb.cuda(args.gpu)
                    pseudo_counter = Counter(selected_label.tolist())
                    if max(pseudo_counter.values()) < len(self.ulb_dset):
                        for i in range(args.num_classes):
                            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

                    # inference and calculate sup/unsup losses
                    with amp_cm():
                        logits = self.model(inputs)
                        logits_x_lb = logits[:num_lb]
                        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                        sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                        # hyper-params for update
                        T = self.t_fn(self.it)
                        p_cutoff = self.p_fn(self.it)

                        unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s,
                                                                               logits_x_ulb_w,
                                                                               'ce', T, p_cutoff,
                                                                               use_hard_labels=args.hard_label)

                        total_loss = sup_loss + self.lambda_u * unsup_loss

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
                    self.ema.update()
                    model_grad=[]
                    for param in self.model.parameters():
                        model_grad=param.grad.detach().view(-1) if not len(model_grad) else torch.cat((model_grad,param.grad.detach().view(-1)))
                    grads.append(model_grad)
                    self.optimizer.step()
                    #self.scheduler.step()
                    self.model.zero_grad()
                    self.supervisedloss=self.supervisedloss+sup_loss
                    self.unsupervisedloss=self.unsupervisedloss+self.lambda_u * unsup_loss
        self.scheduler.step()
        tensor_grads=torch.stack(grads,0)
        mean=torch.mean(tensor_grads)
        for grad in grads:
            deviate=torch.norm(grad-mean)**2
            if deviate>self.maxlocalvariance:
                self.maxlocalvariance=deviate
        per_accs = 0
        loss = total_loss.item()
        return self.model, per_accs, loss

    def calculate_ta(self,model_output,local_iterations,batchsize):
        pseudo_label = torch.softmax(model_output, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        selected=max_probs.ge(self.ta_threshold).float()
        summation=torch.sum(selected)
        self.ta=self.ta+summation/(local_iterations*batchsize)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
