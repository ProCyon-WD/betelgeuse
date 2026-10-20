import torch
import os
import numpy as np
import h5py
# from utils.model_utils import Metrics
import copy
import torch.nn.functional as F
from sklearn.metrics import *
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

# from FLAlgorithms.servers.attentions_simple import call_attentions_simple
# from FLAlgorithms.servers.serverutils import call_attentions_simple_amp

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
mean['mnist']=[0.5]
std['mnist']=[0.5]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
std['svhn'] = [0.1751, 0.1771, 0.1744]

def target_transform(target):
    return int(target)

class Server:
    def __init__(self, args, logger, device, aggregate_method_server, name_algorithm, name_data_total, name_net, name_optimizer,
                 curtimes, dataset, datadir, partition, num_users, beta, noise = 0, initialize_evaluation=True):
        # args
        self.aggregate_method_server = args.aggregate_method_server
        self.name_algorithm = args.name_algorithm
        self.runtimes = args.runtimes
        self.num_global_iters = args.num_global_iters
        self.local_epochs = args.local_epochs
        self.num_total_users = args.num_total_users
        self.num_subset_users = args.num_subset_users
        self.num_classes = args.num_classes
        self.gpu = args.gpu
        self.print_fn = print if logger is None else logger.info
        self.curtimes = curtimes
        # other
        self.users = []
        self.top5=[]
        self.id_selected_users = []
        self.selected_users = []
        # FedSSL sertting
        self.args = args
        self.logger = logger
        self.device = device
        self.aggregate_method_server = aggregate_method_server
        self.name_algorithm = name_algorithm
        self.name_data_total = name_data_total
        self.name_net = name_net
        self.name_optimizer = name_optimizer
        self.noise = noise
        # save results of train and test
        self.evaluate_result = []
        self.rs_train_acc = []
        self.rs_train_loss = []
        self.rs_glob_acc = []
        self.rs_glob_loss = []
        self.rs_train_acc_per = []
        self.rs_train_loss_per = []
        self.rs_glob_acc_per = []
        self.rs_glob_loss_per = []
        self.testacc=[]
        self.F1=[]
        self.precision=[]
        self.recall=[]
        self.AUC=[]
        self.idealloss=[]
        self.grad_norm = []
        # , dataset, datadir, partition, num_users, beta
        self.dataset = dataset
        self.datadir = datadir
        self.partition = partition
        self.num_users = num_users
        self.beta = beta
        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
        }

        self.loader_dict_eval = {}
        if not initialize_evaluation:
            return
        if self.dataset.upper()=='CIFAR10':
            transform_test =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['cifar10'], std['cifar10'])])
            testset = torchvision.datasets.CIFAR10(root=self.datadir, train=False, download=False,transform=transform_test)
            self.loader_dict_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if self.dataset.upper()=='CIFAR100':
            transform_test =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['cifar10'], std['cifar10'])])
            testset = torchvision.datasets.CIFAR100(root=self.datadir, train=False, download=False,transform=transform_test)
            self.loader_dict_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if self.dataset.upper()=='MNIST':
            transform_test =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['mnist'], std['mnist'])])
            testset = torchvision.datasets.MNIST(root=self.datadir, train=False, download=False,transform=transform_test)
            self.loader_dict_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if self.dataset.upper()=='SVHN':
            transform_test =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['svhn'], std['svhn'])])
            testset = torchvision.datasets.SVHN(root=self.datadir, split='train',download=False, transform=torchvision.transforms.ToTensor())
            self.loader_dict_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, num_workers=args.num_workers)
        self.loader_dict_ideal_eval = {}
        if self.dataset.upper()=='CIFAR10':
            transform_test =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['cifar10'], std['cifar10'])])
            testset = torchvision.datasets.CIFAR10(root=self.datadir, train=True, download=False,transform=transform_test)
            self.loader_dict_ideal_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if self.dataset.upper()=='CIFAR100':
            transform_test =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['cifar10'], std['cifar10'])])
            testset = torchvision.datasets.CIFAR100(root=self.datadir, train=True, download=False,transform=transform_test)
            self.loader_dict_ideal_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if self.dataset.upper()=='MNIST':
            transform_test =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['mnist'], std['mnist'])])
            testset = torchvision.datasets.MNIST(root=self.datadir, train=True, download=False,transform=transform_test)
            self.loader_dict_ideal_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if self.dataset.upper()=='SVHN':
            transform_train =transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean['svhn'], std['svhn'])])
            trainset = torchvision.datasets.SVHN(root=self.datadir, split='train',
                                                 download=False, transform=torchvision.transforms.ToTensor())
            self.loader_dict_ideal_eval['eval'] = torch.utils.data.DataLoader(trainset, batch_size=args.eval_batch_size,
                                                                              num_workers=args.num_workers)
        if self.dataset.upper() == 'STL10':
            transform_train = transforms.Compose([transforms.ToTensor()])
            testset = torchvision.datasets.STL10(root=self.datadir, split='test',
                                                 download=False, transform=torchvision.transforms.ToTensor())
            self.loader_dict_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size,
                                                                              num_workers=args.num_workers)
        if self.dataset.upper()=='TINYIMAGENET':
            transform_test =transforms.Compose([transforms.ToTensor()])
            data_dir=self.datadir
            testset=ImageFolder(os.path.join(data_dir, "val"), transform_test)
            self.loader_dict_eval['eval'] = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size,
                                                                              num_workers=args.num_workers)

    def start_training_round(self, round_number, warmup=False):
        recorder = getattr(self, "training_recorder", None)
        if recorder is not None:
            recorder.begin_round(round_number, warmup=warmup)

    def finish_training_round(self):
        recorder = getattr(self, "training_recorder", None)
        return recorder.end_round() if recorder is not None else None

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def evaluate_global_test_dataset(self, eval_loader=None, args=None):
        self.model.train()
        if eval_loader is None:
            eval_loader = self.loader_dict_eval['eval']
        if self.dataset != 'stl10' and self.dataset != 'tinyimagenet':
            eval_ideal_loader=self.loader_dict_ideal_eval['eval']
        total_loss = 0.0
        total_num = 0.0
        total_ideal_loss=0.0
        total_ideal_num=0.0
        y_true = []
        y_pred = []
        y_logits = []
        for batch_idx, (x, y) in enumerate(eval_loader):
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        avg_grad_norm = 0

        if self.dataset.upper() != 'STL10':
            grad_sum = 0
            batch_sum = 0
            for batch_idx,(x,y)in enumerate(eval_ideal_loader):

                self.model.zero_grad()

                x, y = x.cuda(args.gpu), y.cuda(args.gpu)
                num_batch = x.shape[0]
                total_ideal_num+= num_batch
                logits = self.model(x)
                loss = F.cross_entropy(logits, y, reduction='mean')
                loss.backward()
                ############################

                batch_grad = None
                for layer in self.model.parameters():
                    #print(layer)
                    #print(layer.grad)

                    batch_grad = layer.grad.view(-1) if batch_grad is None else torch.cat(
                        (batch_grad, layer.grad.view(-1)))
                model_norm = torch.norm(batch_grad).item()
                batch_sum += 1
                grad_sum += model_norm
                ###############################
                total_ideal_loss += loss.detach() * num_batch
            avg_grad_norm = grad_sum / batch_sum

        else:
            total_ideal_num = 1.0
            total_ideal_loss = torch.tensor(1.0)

        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')
        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.model.train()
        evaluate_result_once =  {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}
        self.logger.warning(f"Average Global Accurancy: {top1}")
        self.logger.warning(f"Average Global Loss: {total_loss / total_num}") # {total_ideal_loss / total_ideal_num}
        #self.logger.warning(f"Average Ideal Train Loss: {total_ideal_loss / total_ideal_num}")
        self.logger.warning(f"Average Global F1:  {F1}")
        self.logger.warning(f"Average Global AUC: {AUC}")
        #self.logger.warning(f"Average Grad Norm: {avg_grad_norm}")
        if len(self.testacc) >0:
            self.logger.warning(f"Best Global Accuracy: {np.max(self.testacc)}")
        #self.evaluate_result.append(evaluate_result_once)
        self.testacc.append(top1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.F1.append(F1)
        self.AUC.append(AUC)
        if self.dataset.upper()=='CIFAR100' or self.dataset.upper()=='TINYIMAGENET':
            self.top5.append(top5)
        self.idealloss.append((total_ideal_loss / total_ideal_num).cpu().numpy())
        # print("avg grad norm = " + str(avg_grad_norm))
        #self.grad_norm.append(avg_grad_norm)
    def save_results(self):
        alg = self.aggregate_method_server + "_" + self.name_algorithm + "_" + \
              self.name_data_total[:-4] + "_" + self.name_net
        alg = alg + "_sub" + str(self.num_subset_users)
        alg = alg + "_" + str(self.curtimes)
        if (len(self.evaluate_result) != 0):
            save_path = "./result/exp_results/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = save_path + f"{alg}.npy"
            np.save(file_name, self.evaluate_result)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
