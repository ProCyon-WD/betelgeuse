"""FedPDM and SAGE on CIFAR-10/SVHN using the common indexed-data training loop."""

from Models.basic_nets import get_basic_net
from Models.resnet import ResNet18
from Server.serverImageNet100Baselines import ImageNet100Baseline
from datautils.small_image_fssl import prepare_small_images, small_image_transforms


class SmallImageBaseline(ImageNet100Baseline):
    def _prepare_data(self, basic):
        return prepare_small_images(self.args, basic)

    def _transforms(self):
        return small_image_transforms(self.dataset)

    def _build_model(self):
        if self.args.name_net == "cifar_resnet18":
            return ResNet18()
        if self.args.name_net == "vgg19_bn":
            return get_basic_net("VGG19-BN", n_classes=10)
        raise ValueError("Small-image baselines require cifar_resnet18 or vgg19_bn")
