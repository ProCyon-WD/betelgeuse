# import needed library
import contextlib
import numpy as np
import torch
import torch.nn.parallel
from torch.cuda.amp import autocast, GradScaler
import math
from collections import Counter
from sklearn.metrics import *

from utils.utils import net_builder, count_parameters, over_write_args_from_file
from utils.train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup, ce_loss, wd_loss, EMA, Bn_Controller,My_CB_loss
from User.fixmatch_utils import consistency_loss, Get_Scalar
# from datasets.ssl_dataset import SSL_Dataset, ImageNetLoader
# from datasets.data_utils import get_data_loader

from User.userbase import User
from experiments.local_training import local_batches, singleton_batchnorm, uses_local_epochs

class UserReweight_mycb(User):
    def __init__(self, id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, model, args, logger, device, aggregate_method_server,
                 name_algorithm, name_data_total, name_model, name_optimizer,num_classes):
        super().__init__(id_user, tag_client_lbulb, lb_dset, ulb_dset, loader_dict, model, args, logger, device, aggregate_method_server,
                         name_algorithm, name_data_total, name_model, name_optimizer)

        # args need xxxx
        # args.optim, args.lr, args.momentum, args.weight_decay

        # Init for different args
        # super().__init__
        self.local_epochs = args.local_epochs
        self.T = args.T
        self.p_cutoff = args.p_cutoff
        self.num_classes=num_classes
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


        num_train_iter = args.num_global_iters
        optimizer = get_optimizer(self.model, args.name_optimizer, args.lr, args.momentum, args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_train_iter,
                                                    num_warmup_steps=num_train_iter * 0)
        ## set SGD and cosine lr on FixMatch
        self.optimizer = optimizer
        self.scheduler = scheduler

        if not torch.cuda.is_available() and not self.cpu_client_state:
            raise Exception('ONLY GPU TRAINING IS SUPPORTED')
        elif not self.cpu_client_state:
            # model.model = torch.nn.DataParallel(model.model).cuda()
            self.model = self.model.cuda(self.device)

        import copy
        self.ema_model = None if self.cpu_client_state else copy.deepcopy(self.model)

        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)
        self.supervisedloss=0
        self.unsupervisedloss=0
        self.ta=0
        self.ta_threshold = args.ta_threshold if self.cpu_client_state else 0.95
        self.mask_ratio = 0.0
        #self.global_distribution=global_distribution

    def set_global_distribution(self,global_distribution,k=-1):
        self.global_distribution=global_distribution

        weights = np.array(global_distribution, dtype=np.float64)
        weights=weights/sum(weights)

        for i in range(self.num_classes):
            weights[i] = math.exp(weights[i] * k)

        weights = weights / np.sum(weights) * self.num_classes * 2
        print(weights)
        self.loss_weight = torch.tensor(weights, dtype=torch.float32, device=next(self.model.parameters()).device)



    def move_training_state(self, device):
        """Move only an active client's training state; retain optimizer parameter identities."""
        for parameter in self.model.parameters():
            parameter.data = parameter.data.to(device)
            if parameter.grad is not None:
                parameter.grad = parameter.grad.to(device)
        self.model.to(device)
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
        self.ema.shadow = {name: value.to(device) for name, value in self.ema.shadow.items()}
        if hasattr(self, "loss_weight"):
            self.loss_weight = self.loss_weight.to(device)
        if torch.device(device).type == "cpu":
            self.train_lb_iterator = None
            self.train_ulb_iterator = None
            for name in ("supervisedloss", "unsupervisedloss", "ta", "maxlocalvariance"):
                value = getattr(self, name, None)
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.detach().cpu())

    def train(self, down_model, args, glob_iter, logger=None):
    # def train(self, args, glob_iter, logger=None):

        # ngpus_per_node = torch.cuda.device_count()
        
        if self.cpu_client_state:
            self.move_training_state(self.device)
            if self.train_lb_dataloader is not None and not uses_local_epochs(args):
                self.train_lb_iterator = iter(self.train_lb_dataloader)
            if self.train_ulb_dataloader is not None and not uses_local_epochs(args):
                self.train_ulb_iterator = iter(self.train_ulb_dataloader)
        self.model.load_state_dict(down_model.state_dict())
        self.supervisedloss=0
        self.unsupervisedloss=0
        self.maxlocalvariance = 0
        self.ta=0
        self.mask_ratio = 0.0
        self.model.train()
        grads = []
        gradient_moments = []
        epoch_mode = uses_local_epochs(args)
        if epoch_mode and self.tag_client_lbulb not in (0, 2):
            raise ValueError("FedCyon epoch mode supports pure labeled/unlabeled clients only")
        self.local_train_steps = 0
        self.local_train_samples = 0


        if self.tag_client_lbulb == 0:

            for i in range(1):

                scaler = GradScaler(enabled=args.amp)
                amp_cm = autocast if args.amp else contextlib.nullcontext



                for _, x_o, x_lb, y_lb in local_batches(self.train_lb_dataloader, args, None if epoch_mode else self.train_lb_iterator):
                #for _, x_o, x_lb, y_lb in self.loader_dict['train_lb']:

                    num_lb = x_lb.shape[0]
                    # num_ulb = x_ulb_w.shape[0]
                    # assert num_ulb == x_ulb_s.shape[0]

                    # x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
                    x_o = x_o.to(self.device)
                    x_lb = x_lb.to(self.device)
                    y_lb = y_lb.to(self.device)


                    inputs = x_o

                    # inference and calculate sup/unsup losses
                    with amp_cm(), singleton_batchnorm(self.model, epoch_mode and num_lb == 1):
                        logits = self.model(inputs)
                        logits_x_lb = logits[:num_lb]
                        self.calculate_ta(logits_x_lb,self.local_iterations,num_lb)
                        # logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)


                        #sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                        #sup_loss=focal_loss(logits_x_lb, y_lb, reduction='mean')
                        sup_loss=My_CB_loss(y_lb, logits_x_lb, self.global_distribution, self.num_classes, loss_type="softmax", k=-1,reduction='mean',weight=self.loss_weight)
                        #sup_loss=CB_loss(y_lb, logits_x_lb, self.global_distribution, 10, loss_type="focal", beta=0.99, gamma=2.0)
                        # # hyper-params for update
                        # T = self.t_fn(self.it)
                        # p_cutoff = self.p_fn(self.it)
                        #
                        # unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s,
                        #                                                        logits_x_ulb_w,
                        #                                                        'ce', T, p_cutoff,
                        #                                                        use_hard_labels=args.hard_label)

                        # total_loss = sup_loss + self.lambda_u * unsup_loss
                        total_loss = sup_loss

                    # parameter updates
                    self.record_training_loss(total_loss, num_lb if epoch_mode else None)
                    if args.amp:
                        scaler.scale(total_loss).backward()
                        if (args.clip > 0):
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        if not self.cpu_client_state:
                            model_grad = []
                            for param in self.model.parameters():
                                model_grad = param.grad.detach().view(-1) if not len(model_grad) else torch.cat(
                                    (model_grad, param.grad.detach().view(-1)))
                            if epoch_mode:
                                gradient_moments.append((model_grad.sum(), model_grad.square().sum(), model_grad.numel()))
                            else:
                                grads.append(model_grad)
                        self.optimizer.step()


                    # self.scheduler.step()
                    self.ema.update()
                    self.model.zero_grad()
                    self.local_train_steps += 1
                    self.local_train_samples += num_lb
                    self.supervisedloss=self.supervisedloss+sup_loss.detach() * (num_lb if epoch_mode else 1)

        elif self.tag_client_lbulb == 2:

            # model update
            # for i in range(self.local_epochs):
            for i in range(1):


                scaler = GradScaler(enabled=args.amp)
                amp_cm = autocast if args.amp else contextlib.nullcontext

                selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
                selected_label = selected_label.to(self.device)

                classwise_acc = torch.zeros((args.num_classes,)).to(self.device)

                for x_ulb_idx, x_ulb_w, x_ulb_s in local_batches(self.train_ulb_dataloader, args, None if epoch_mode else self.train_ulb_iterator):

                #for (x_ulb_idx, x_ulb_w, x_ulb_s) in self.loader_dict['train_ulb']:
                    # num_lb = x_lb.shape[0]
                    num_ulb = x_ulb_w.shape[0]
                    assert num_ulb == x_ulb_s.shape[0]

                    # x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
                    # y_lb = y_lb.cuda(args.gpu)
                    x_ulb_w, x_ulb_s = x_ulb_w.to(self.device), x_ulb_s.to(self.device)

                    pseudo_counter = Counter(selected_label.tolist())
                    if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                        for i in range(args.num_classes):
                            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

                    # inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    inputs = torch.cat((x_ulb_w, x_ulb_s))

                    # inference and calculate sup/unsup losses
                    with amp_cm():
                        logits = self.model(inputs)
                        # logits_x_lb = logits[:num_lb]
                        # logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                        # sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                        logits_x_ulb_w, logits_x_ulb_s = logits[:].chunk(2)
                        self.calculate_ta(logits_x_ulb_w,self.local_iterations,num_ulb)
                        # hyper-params for update
                        T = self.t_fn(self.it)
                        p_cutoff = self.p_fn(self.it)

                        unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_x_ulb_s,
                                                                               logits_x_ulb_w,
                                                                               'ce', T, p_cutoff,
                                                                               use_hard_labels=args.hard_label)

                        # total_loss = sup_loss + self.lambda_u * unsup_loss
                        total_loss = unsup_loss
                        if epoch_mode:
                            self.mask_ratio += mask.detach().item() * num_ulb
                        elif self.cpu_client_state:
                            self.mask_ratio += mask.detach().item() / self.local_iterations

                    # parameter updates
                    self.record_training_loss(total_loss, num_ulb if epoch_mode else None)
                    if args.amp:
                        scaler.scale(total_loss).backward()
                        if (args.clip > 0):
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        if not self.cpu_client_state:
                            model_grad = []
                            for param in self.model.parameters():
                                model_grad = param.grad.detach().view(-1) if not len(model_grad) else torch.cat(
                                    (model_grad, param.grad.detach().view(-1)))
                            if epoch_mode:
                                gradient_moments.append((model_grad.sum(), model_grad.square().sum(), model_grad.numel()))
                            else:
                                grads.append(model_grad)
                        self.optimizer.step()

                    #self.scheduler.step()
                    self.ema.update()
                    self.model.zero_grad()
                    self.local_train_steps += 1
                    self.local_train_samples += num_ulb
                    self.unsupervisedloss=self.unsupervisedloss+unsup_loss.detach() * (num_ulb if epoch_mode else 1)

        else:

            # model update
            # for i in range(self.local_epochs):
            for i in range(1):


                scaler = GradScaler(enabled=args.amp)
                amp_cm = autocast if args.amp else contextlib.nullcontext

                selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
                selected_label = selected_label.to(self.device)

                classwise_acc = torch.zeros((args.num_classes,)).to(self.device)

                # for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],
                #                                                           self.loader_dict['train_ulb']):
                for iiii in range(self.local_iterations):
                    _, x_o, x_lb, y_lb=next(self.train_lb_iterator)
                    x_ulb_idx, x_ulb_w, x_ulb_s=next(self.train_ulb_iterator)
                #for (_, x_o, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) in zip(self.loader_dict['train_lb'],self.loader_dict['train_ulb']):


                    num_lb = x_lb.shape[0]
                    num_ulb = x_ulb_w.shape[0]
                    assert num_ulb == x_ulb_s.shape[0]

                    x_lb, x_ulb_w, x_ulb_s = x_lb.to(self.device), x_ulb_w.to(self.device), x_ulb_s.to(self.device)
                    y_lb = y_lb.to(self.device)

                    pseudo_counter = Counter(selected_label.tolist())
                    if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
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
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if (args.clip > 0):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                        self.optimizer.step()

                    #self.scheduler.step()
                    self.ema.update()
                    self.model.zero_grad()
                    self.supervisedloss=self.supervisedloss+sup_loss
                    self.unsupervisedloss=self.unsupervisedloss+self.lambda_u * unsup_loss

        if epoch_mode:
            if self.local_train_samples == 0:
                raise ValueError("Epoch training requires a nonempty local dataset")
            self.ta = self.ta / self.local_train_samples
            self.supervisedloss = self.supervisedloss / self.local_train_samples
            self.unsupervisedloss = self.unsupervisedloss / self.local_train_samples
            self.mask_ratio = self.mask_ratio / self.local_train_samples
            if gradient_moments:
                mean = sum(total for total, squared, count in gradient_moments) / sum(count for total, squared, count in gradient_moments)
                self.maxlocalvariance = torch.stack([
                    squared - 2 * mean * total + count * mean.square()
                    for total, squared, count in gradient_moments]).clamp_min(0).max()
        self.scheduler.step()
        if grads:
            tensor_grads=torch.stack(grads,0)
            mean=torch.mean(tensor_grads)
            for grad in grads:
                deviate=torch.norm(grad-mean)**2
                if deviate>self.maxlocalvariance:
                    self.maxlocalvariance=deviate


        per_accs = 0
        loss = float(self.supervisedloss + self.unsupervisedloss) if epoch_mode else total_loss.item()
        return self.model, per_accs, loss

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
    def calculate_ta(self,model_output,local_iterations,batchsize):
        pseudo_label = torch.softmax(model_output, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        selected=max_probs.ge(self.ta_threshold).float()
        summation=torch.sum(selected)
        if uses_local_epochs(self.args):
            self.ta = self.ta + summation
        else:
            self.ta = self.ta + summation / (local_iterations * batchsize)


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
