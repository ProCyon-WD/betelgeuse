import torch

from .data_utils import split_ssl_data, sample_labeled_data
from datautils.data_utils.dataset import BasicDataset
from collections import Counter
import torchvision
import numpy as np
from torchvision import transforms
import json
import os

import random
from .augmentation.randaugment import RandAugment

from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
# from datasets.DistributedProxySampler import DistributedProxySampler

import gc
import sys
import copy
from PIL import Image

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['svhn'] = [0.4380, 0.4440, 0.4730]
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
mean['imagenet'] = [0.485, 0.456, 0.406]
mean['tinyimagenet'] = [0.485, 0.456, 0.406]
mean['mnist'] = [0.5]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['svhn'] = [0.1751, 0.1771, 0.1744]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
std['imagenet'] = [0.229, 0.224, 0.225]
std['tinyimagenet'] = [0.229, 0.224, 0.225]
std['mnist'] = [0.5]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImagenetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ulb, num_labels=-1):
        super().__init__(root, transform)
        self.ulb = ulb
        self.num_labels = num_labels
        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.ulb:
            self.strong_transform = copy.deepcopy(transform)
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_transformed = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (index, sample_transformed, target) if not self.ulb else (
            index, sample_transformed, self.strong_transform(sample))

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        lb_idx = {}

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.num_labels != -1:
                    fnames = fnames[:self.num_labels]
                if self.num_labels != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        if self.num_labels != -1:
            with open('./sampled_label_idx.json', 'w') as f:
                json.dump(lb_idx, f)
        del lb_idx
        gc.collect()
        return instances


class ImageNetLoader:
    def __init__(self, root_path, num_labels=-1, num_class=1000):
        self.root_path = os.path.join(root_path, 'imagenet')
        self.num_labels = num_labels // num_class

    def get_transform(self, train, ulb):
        if train:
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        else:
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean["imagenet"], std["imagenet"])])
        return transform

    def get_lb_train_data(self):
        transform = self.get_transform(train=True, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=False,
                               num_labels=self.num_labels)
        return data

    def get_ulb_train_data(self):
        transform = self.get_transform(train=True, ulb=True)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=True)
        return data

    def get_lb_test_data(self):
        transform = self.get_transform(train=False, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "val"), transform=transform, ulb=False)
        return data

def get_transforms(name):
    crop_size = 96 if name.upper() == 'STL10' else 224 if name.upper() == 'IMAGENET' else 32
    return get_transform(mean[name], std[name], crop_size, True), get_transform(mean[name], std[name], crop_size, False)

def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(crop_size, padding=4, padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def load_data_total(data_dir, name_dset, name_data_total):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    # if(dataset == "Cifar10"):
    #     clients, groups, train_data, test_data = read_cifa_data()
    #     return clients, groups, train_data, test_data

    # train_data_dir = os.path.join('data', name_data_total, 'data', 'train')
    # test_data_dir = os.path.join('data', name_data_total, 'data', 'test')
    # data_dir_specific = os.path.join(data_dir,name_dset, name_data_total)
    data_dir = '/home/wudi/fedssl/data/data_generate_result/'
    name_data_total = 'data_total_cifar10_NonIID_alpha0.1_num20_2018_prolb0.1.npy'
    data_dir_specific = os.path.join(data_dir, name_data_total)
    print(name_data_total)
    # data_dir = "../data/data_generate_result/"
    # name_data_total = "data_total_svhn_NonIID_alpha0.3_num60_pro123.npy"
    # name_data_total = "data_total_cifar10_NonIID_alpha0.3_num60_pro123.npy"
    # name_data_total = "data_total_cifar10_IID_alpha0.3_num60_pro123.npy"
    # name_data_total = "data_total_cifar100_NonIID_alpha0.3_num60_pro123.npy"
    # name_data_total = "data_total_cifar100_IID_alpha0.3_num60_pro123.npy"
    # name_data_total = "data_total_svhn_NonIID_alpha0.3_num60_pro123.npy"
    # name_data_total = "data_total_svhn_IID_alpha0.3_num60_pro123.npy"
    # data_dir_specific = os.path.join(data_dir, name_data_total)

    data_total_temp = np.load(data_dir_specific, allow_pickle=True)
    # data_total is a dict, the last element is test data and targets,
    # other element is the data of each client.
    data_total = data_total_temp.item()
    print(data_total)
    print(type(data_total[0]))
    # clients = []
    # train_lb = []
    # train_ulb = []
    # lb_ulb = []
    # eval = []
    # # clients = []
    # # groups = []
    # # train_data = {}
    # # test_data = {}
    #
    # # train_files = os.listdir(train_data_dir)
    # # train_files = [f for f in train_files if f.endswith('.json')]
    # data_files = os.listdir(data_dir_specific)
    # data_files = [f for f in data_files if f.endswith('.json')]
    # for f in data_files:
    #     file_path = os.path.join(data_dir_specific, f)
    #     with open(file_path, 'r') as inf:
    #         cdata = json.load(inf)
    #     clients.extend(cdata)
    #     if 'hierarchies' in cdata:
    #         groups.extend(cdata['hierarchies'])
    #     train_data.update(cdata['user_data'])
    #
    # test_files = os.listdir(test_data_dir)
    # test_files = [f for f in test_files if f.endswith('.json')]
    # for f in test_files:
    #     file_path = os.path.join(test_data_dir, f)
    #     with open(file_path, 'r') as inf:
    #         cdata = json.load(inf)
    #     test_data.update(cdata['user_data'])
    #
    # clients = list(sorted(train_data.keys()))

    return data_total


# class SSL_Dataset:
#     """
#     SSL_Dataset class gets dataset from torchvision.datasets,
#     separates labeled and unlabeled data,
#     and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
#     """
#
#     def __init__(self,
#                  args,
#                  alg='fixmatch',
#                  name='cifar10',
#                  train=True,
#                  num_classes=10,
#                  data_dir='./data'):
#         """
#         Args
#             alg: SSL algorithms
#             name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
#             train: True means the dataset is training dataset (default=True)
#             num_classes: number of label classes
#             data_dir: path of directory, where data is downloaed or stored.
#         """
#         self.args = args
#         self.alg = alg
#         self.name = name
#         self.train = train
#         self.num_classes = num_classes
#         self.data_dir = data_dir
#         crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
#         self.transform = get_transform(mean[name], std[name], crop_size, train)
#
#     def get_data(self, svhn_extra=True):
#         """
#         get_data returns data (images) and targets (labels)
#         shape of data: B, H, W, C
#         shape of labels: B,
#         """
#         dset = getattr(torchvision.datasets, self.name.upper())
#         if 'CIFAR' in self.name.upper():
#             dset = dset(self.data_dir, train=self.train, download=True)
#             data, targets = dset.data, dset.targets
#             return data, targets
#         elif self.name.upper() == 'SVHN':
#             if self.train:
#                 if svhn_extra:  # train+extra
#                     dset_base = dset(self.data_dir, split='train', download=True)
#                     data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
#                     dset_extra = dset(self.data_dir, split='extra', download=True)
#                     data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
#                     data = np.concatenate([data_b, data_e])
#                     targets = np.concatenate([targets_b, targets_e])
#                     del data_b, data_e
#                     del targets_b, targets_e
#                 else:  # train_only
#                     dset = dset(self.data_dir, split='train', download=True)
#                     data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
#             else:  # test
#                 dset = dset(self.data_dir, split='test', download=True)
#                 data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
#             return data, targets
#         elif self.name.upper() == 'STL10':
#             split = 'train' if self.train else 'test'
#             dset_lb = dset(self.data_dir, split=split, download=True)
#             dset_ulb = dset(self.data_dir, split='unlabeled', download=True)
#             data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
#             ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])
#             return data, targets, ulb_data
#
#     def get_dset(self, is_ulb=False,
#                  strong_transform=None, onehot=False):
#         """
#         get_dset returns class BasicDataset, containing the returns of get_data.
#
#         Args
#             is_ulb: If True, returned dataset generates a pair of weak and strong augmented images.
#             strong_transform: list of strong_transform (augmentation) if use_strong_transform is True。
#             onehot: If True, the label is not integer, but one-hot vector.
#         """
#
#         if self.name.upper() == 'STL10':
#             data, targets, _ = self.get_data()
#         else:
#             data, targets = self.get_data()
#         num_classes = self.num_classes
#         transform = self.transform
#
#         return BasicDataset(self.alg, data, targets, num_classes, transform,
#                             is_ulb, strong_transform, onehot)
#
#     def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
#                      strong_transform=None, onehot=False):
#         """
#         get_ssl_dset split training samples into labeled and unlabeled samples.
#         The labeled data is balanced samples over classes.
#
#         Args:
#             num_labels: number of labeled data.
#             index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
#             include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
#             strong_transform: list of strong transform (RandAugment in FixMatch)
#             onehot: If True, the target is converted into onehot vector.
#
#         Returns:
#             BasicDataset (for labeled data), BasicDataset (for unlabeld data)
#         """
#         # Supervised top line using all data as labeled data.
#         if self.alg == 'fullysupervised':
#             lb_data, lb_targets = self.get_data()
#             lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
#                                    self.transform, False, None, onehot)
#             return lb_dset, None
#
#         if self.name.upper() == 'STL10':
#             lb_data, lb_targets, ulb_data = self.get_data()
#             if include_lb_to_ulb:
#                 ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
#             lb_data, lb_targets, _ = sample_labeled_data(self.args, lb_data, lb_targets, num_labels, self.num_classes)
#             ulb_targets = None
#         else:
#             data, targets = self.get_data()
#             lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
#                                                                         num_labels, self.num_classes,
#                                                                         index, include_lb_to_ulb)
#         # output the distribution of labeled data for remixmatch
#         count = [0 for _ in range(self.num_classes)]
#         for c in lb_targets:
#             count[c] += 1
#         dist = np.array(count, dtype=float)
#         dist = dist / dist.sum()
#         dist = dist.tolist()
#         out = {"distribution": dist}
#         output_file = r"./data_statistics/"
#         output_path = output_file + str(self.name) + '_' + str(num_labels) + '.json'
#         if not os.path.exists(output_file):
#             os.makedirs(output_file, exist_ok=True)
#         with open(output_path, 'w') as w:
#             json.dump(out, w)
#         # print(Counter(ulb_targets.tolist()))
#         lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
#                                self.transform, False, None, onehot)
#
#         ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets, self.num_classes,
#                                 self.transform, True, strong_transform, onehot)
#         # print(lb_data.shape)
#         # print(ulb_data.shape)
#         return lb_dset, ulb_dset


class SSL_Dataset:
    """
    SSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 args,
                 alg='fixmatch',
                 data_dir='./data',
                 name='cifar10',
                 name_data_total=None,
                 num_classes=10):
        """
        Args
            alg: SSL algorithms
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        self.args = args
        self.alg = alg
        self.name = name
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.name_data_total = name_data_total
        self.num_total_users = args.num_total_users

        crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
        self.transform = get_transform(mean[name], std[name], crop_size, train=True)
        self.transform_test = get_transform(mean[name], std[name], crop_size, train=False)

        # load data
        self.data_total = load_data_total(self.data_dir, self.name, self.name_data_total)


    def get_per_user_data(self, id_user, strong_transform=None, onehot=False):
        #
        data_per_user = self.data_total[id_user]
        lb_data = data_per_user["lb_data"]
        lb_targets = data_per_user["lb_targets"]
        ulb_data = data_per_user["ulb_data"]
        ulb_targets = data_per_user["ulb_targets"]
        # lb_ulb_data = data_per_user["lb_ulb_data"]
        # lb_ulb_targets = data_per_user["lb_ulb_targets"]
        # eval_data = data_per_user["eval_data"]
        # eval_targets = data_per_user["eval_targets"]



        is_ulb = False
        lb_dset = BasicDataset(self.alg, lb_data, lb_targets,
                               self.num_classes, self.transform, is_ulb, strong_transform, onehot, self.transform_test)
        is_ulb = True
        ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets,
                                self.num_classes, self.transform, is_ulb, strong_transform, onehot)
        # is_ulb = False
        # lb_ulb_dset = BasicDataset(self.alg, lb_ulb_data, lb_ulb_targets,
        #                            self.num_classes, self.transform_test, is_ulb, strong_transform, onehot)
        # is_ulb = False
        # eval_dset = BasicDataset(self.alg, eval_data, eval_targets,
        #                          self.num_classes, self.transform_test, is_ulb, strong_transform, onehot)

        return lb_dset, ulb_dset,

    def get_total_test_data(self, id_user=61, strong_transform=None, onehot=False):
        #
        id_user = self.num_total_users
        data_per_user = self.data_total[id_user]
        # lb_data = data_per_user["lb_data"]
        # lb_targets = data_per_user["lb_targets"]
        # ulb_data = data_per_user["ulb_data"]
        # ulb_targets = data_per_user["ulb_targets"]
        # lb_ulb_data = data_per_user["lb_ulb_data"]
        # lb_ulb_targets = data_per_user["lb_ulb_targets"]
        test_data = data_per_user["test_data"]
        test_targets = data_per_user["test_targets"]



        # is_ulb = False
        # lb_dset = BasicDataset(self.alg, lb_data, lb_targets,
        #                        self.num_classes, self.transform, is_ulb, strong_transform, onehot)
        # is_ulb = True
        # ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets,
        #                         self.num_classes, self.transform, is_ulb, strong_transform, onehot)
        # is_ulb = False
        # lb_ulb_dset = BasicDataset(self.alg, lb_ulb_data, lb_ulb_targets,
        #                            self.num_classes, self.transform_test, is_ulb, strong_transform, onehot)
        is_ulb = False
        test_dset = BasicDataset(self.alg, test_data, test_targets,
                                 self.num_classes, self.transform_test, is_ulb, strong_transform, onehot)

        return test_dset

    # def get_data(self, svhn_extra=True):
    #     """
    #     get_data returns data (images) and targets (labels)
    #     shape of data: B, H, W, C
    #     shape of labels: B,
    #     """
    #     dset = getattr(torchvision.datasets, self.name.upper())
    #     if 'CIFAR' in self.name.upper():
    #         dset = dset(self.data_dir, train=self.train, download=True)
    #         data, targets = dset.data, dset.targets
    #         return data, targets
    #     elif self.name.upper() == 'SVHN':
    #         if self.train:
    #             if svhn_extra:  # train+extra
    #                 dset_base = dset(self.data_dir, split='train', download=True)
    #                 data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
    #                 dset_extra = dset(self.data_dir, split='extra', download=True)
    #                 data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
    #                 data = np.concatenate([data_b, data_e])
    #                 targets = np.concatenate([targets_b, targets_e])
    #                 del data_b, data_e
    #                 del targets_b, targets_e
    #             else:  # train_only
    #                 dset = dset(self.data_dir, split='train', download=True)
    #                 data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
    #         else:  # test
    #             dset = dset(self.data_dir, split='test', download=True)
    #             data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
    #         return data, targets
    #     elif self.name.upper() == 'STL10':
    #         split = 'train' if self.train else 'test'
    #         dset_lb = dset(self.data_dir, split=split, download=True)
    #         dset_ulb = dset(self.data_dir, split='unlabeled', download=True)
    #         data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    #         ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])
    #         return data, targets, ulb_data
    #
    # def get_dset(self, is_ulb=False,
    #              strong_transform=None, onehot=False):
    #     """
    #     get_dset returns class BasicDataset, containing the returns of get_data.
    #
    #     Args
    #         is_ulb: If True, returned dataset generates a pair of weak and strong augmented images.
    #         strong_transform: list of strong_transform (augmentation) if use_strong_transform is True。
    #         onehot: If True, the label is not integer, but one-hot vector.
    #     """
    #
    #     if self.name.upper() == 'STL10':
    #         data, targets, _ = self.get_data()
    #     else:
    #         data, targets = self.get_data()
    #     num_classes = self.num_classes
    #     transform = self.transform
    #
    #     return BasicDataset(self.alg, data, targets, num_classes, transform,
    #                         is_ulb, strong_transform, onehot)
    #
    # def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
    #                  strong_transform=None, onehot=False):
    #     """
    #     get_ssl_dset split training samples into labeled and unlabeled samples.
    #     The labeled data is balanced samples over classes.
    #
    #     Args:
    #         num_labels: number of labeled data.
    #         index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
    #         include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
    #         strong_transform: list of strong transform (RandAugment in FixMatch)
    #         onehot: If True, the target is converted into onehot vector.
    #
    #     Returns:
    #         BasicDataset (for labeled data), BasicDataset (for unlabeld data)
    #     """
    #     #
    #     data_per_user = self.data_total[f"client{id_user}"]
    #     self.lb_data = data_per_user["lb_data"]
    #     self.lb_targets = data_per_user["lb_targets"]
    #     self.ulb_data = data_per_user["ulb_data"]
    #     self.ulb_targets = data_per_user["ulb_targets"]
    #     self.lb_ulb_data = data_per_user["lb_ulb_data"]
    #     self.lb_ulb_targets = data_per_user["lb_ulb_targets"]
    #     self.eval_data = data_per_user["eval_data"]
    #     self.eval_targets = data_per_user["eval_targets"]
    #
    #
    #     # Supervised top line using all data as labeled data.
    #     if self.alg == 'fullysupervised':
    #         lb_data, lb_targets = self.get_data()
    #         lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
    #                                self.transform, False, None, onehot)
    #         return lb_dset, None
    #
    #     if self.name.upper() == 'STL10':
    #         lb_data, lb_targets, ulb_data = self.get_data()
    #         if include_lb_to_ulb:
    #             ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
    #         lb_data, lb_targets, _ = sample_labeled_data(self.args, lb_data, lb_targets, num_labels, self.num_classes)
    #         ulb_targets = None
    #     else:
    #         data, targets = self.get_data()
    #         lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
    #                                                                     num_labels, self.num_classes,
    #                                                                     index, include_lb_to_ulb)
    #     # output the distribution of labeled data for remixmatch
    #     count = [0 for _ in range(self.num_classes)]
    #     for c in lb_targets:
    #         count[c] += 1
    #     dist = np.array(count, dtype=float)
    #     dist = dist / dist.sum()
    #     dist = dist.tolist()
    #     out = {"distribution": dist}
    #     output_file = r"./data_statistics/"
    #     output_path = output_file + str(self.name) + '_' + str(num_labels) + '.json'
    #     if not os.path.exists(output_file):
    #         os.makedirs(output_file, exist_ok=True)
    #     with open(output_path, 'w') as w:
    #         json.dump(out, w)
    #     # print(Counter(ulb_targets.tolist()))
    #     lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
    #                            self.transform, False, None, onehot)
    #
    #     ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets, self.num_classes,
    #                             self.transform, True, strong_transform, onehot)
    #     # print(lb_data.shape)
    #     # print(ulb_data.shape)
    #
    #
    #     #
    #     self.lb_data = data_per_user["lb_data"]
    #     self.lb_targets = data_per_user["lb_targets"]
    #     self.ulb_data = data_per_user["ulb_data"]
    #     self.ulb_targets = data_per_user["ulb_targets"]
    #     self.lb_ulb_data = data_per_user["lb_ulb_data"]
    #     self.lb_ulb_targets = data_per_user["lb_ulb_targets"]
    #     self.eval_data = data_per_user["eval_data"]
    #     self.eval_targets = data_per_user["eval_targets"]
    #
    #     lb_dset = BasicDataset(self.alg, self.lb_data, self.lb_targets,
    #                            self.num_classes, self.transform, False, None, onehot)
    #     ulb_dset = BasicDataset(self.alg, self.ulb_data, self.ulb_targets,
    #                             self.num_classes, self.transform, True, strong_transform, onehot)
    #     lb_ulb_dset = BasicDataset(self.alg, self.lb_ulb_data, self.lb_ulb_targets,
    #                             self.num_classes, self.transform, False, None, onehot)
    #     eval_dset = BasicDataset(self.alg, self.eval_data, self.eval_targets,
    #                             self.num_classes, self.transform, False, None, onehot)
    #
    #     return lb_dset, ulb_dset, lb_ulb_dset, eval_dset
