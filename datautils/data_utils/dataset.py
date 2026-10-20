from torchvision import transforms
from torch.utils.data import Dataset
from datautils.data_utils.data_utils import get_onehot
from datautils.data_utils.augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import torch
import copy


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 basic_transform=None,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        self.strong_transform = strong_transform
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

        self.basic_transform = basic_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images

        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            # img_o = transforms.ToTensor()(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                if self.basic_transform is None:
                    return idx, img_w, target
                else:
                    img_o = self.basic_transform(img)
                    return idx, img_o, img_w, target

            else:
                if self.alg == 'fixmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'flexmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'pimodel':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'pseudolabel':
                    return idx, img_w
                elif self.alg == 'vat':
                    return idx, img_w
                elif self.alg == 'meanteacher':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'uda':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'mixmatch':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1)
                elif self.alg == 'fullysupervised':
                    return idx

    def __len__(self):
        return len(self.data)
class StrongDataset(Dataset):
    def __init__(self, dataset_type, data_np, label_np, pre_w, pre_h, lab_trans=None,
                 data_idxs=None,
                 is_labeled=False,
                 is_testing=False):
        """
        Args:

            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(StrongDataset, self).__init__()

        self.images = data_np
        self.labels = label_np
        self.is_labeled = is_labeled
        self.dataset_type = dataset_type
        self.is_testing = is_testing

        self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
        self.resize_trans = transforms.Compose([transforms.Resize((256, 256))])
        weak_trans = transforms.Compose([
            transforms.RandomCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        strong_trans = transforms.Compose([
            transforms.RandomCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        if not is_testing:
            if is_labeled == True:
                self.transform = lab_trans
            else:
                #self.data_idxs = data_idxs
                self.weak_trans = weak_trans
                self.strong_trans = strong_trans
        else:
            self.transform = lab_trans

        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = self.images[index]
        if self.dataset_type == 'skin':
            img_path = self.images[index]
            image = Image.open(img_path).convert('RGB')
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')

        image_resized = self.resize(image)
        image_resized_trans = self.resize_trans(image)
        label = self.labels[index]

        if not self.is_testing:
            if self.is_labeled == True:
                if self.transform is not None:
                    image = self.transform(image_resized).squeeze()
                    # image=image[:,:224,:224]
                    return index, image, torch.FloatTensor([label])
            else:
                if self.weak_trans is not None:
                    weak_aug = self.weak_trans(image_resized)
                    strong_aug = self.weak_trans(image_resized)

                    for idx in range(len(weak_aug)):
                        weak_aug[idx] = weak_aug[idx].squeeze()
                        strong_aug[idx] = strong_aug[idx].squeeze()
                    return index, weak_aug, strong_aug
        else:
            image = self.transform(image_resized)
            return index, image, torch.FloatTensor([label])
            # return index, weak_aug, strong_aug, torch.FloatTensor([label])

    def __len__(self):
        return len(self.labels)
