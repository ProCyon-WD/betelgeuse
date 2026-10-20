"""CIFAR-10/SVHN pure-client data with shared images and explicit test splits."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from torchvision import datasets, transforms

from datautils.data_utils.augmentation.randaugment import RandAugment
from datautils.imagenet100 import partition_indices
from datautils.partition_store import load_partition, partition_directory, partition_exists, save_partition
from experiments.paths import project_path


def small_image_transforms(dataset):
    statistics = {
        "cifar10": ([125.3 / 255, 123.0 / 255, 113.9 / 255], [63.0 / 255, 62.1 / 255, 66.7 / 255]),
        "svhn": ([0.4380, 0.4440, 0.4730], [0.1751, 0.1771, 0.1744]),
    }
    mean, deviation = statistics[dataset]
    normalization = [transforms.ToTensor(), transforms.Normalize(mean, deviation)]
    augmentations = [transforms.RandomCrop(32, padding=4, padding_mode="reflect")]
    if dataset == "cifar10":
        augmentations.append(transforms.RandomHorizontalFlip())
    weak = transforms.Compose(augmentations + normalization)
    strong = transforms.Compose(copy.deepcopy(augmentations) + [RandAugment(3, 5)] + copy.deepcopy(normalization))
    return weak, strong, transforms.Compose(copy.deepcopy(normalization))


def raw_dataset(args, training, transform=None):
    if args.dataset == "cifar10":
        return datasets.CIFAR10(args.datadir, train=training, transform=transform, download=args.download_data)
    if args.dataset == "svhn":
        return datasets.SVHN(args.datadir, split="train" if training else "test",
                             transform=transform, download=args.download_data)
    raise ValueError("Small-image FSSL supports cifar10 and svhn only")


def prepare_small_images(args, basic):
    explicit = bool(args.partition_file)
    if explicit:
        filename = project_path(args.partition_file)
        if filename.name.split("_", 1)[0] != args.dataset:
            raise ValueError("partition_file dataset prefix does not match --dataset")
        if not partition_exists(filename):
            raise FileNotFoundError("Explicit client partition does not exist: {}".format(filename))
    else:
        metadata = {"dataset": args.dataset, "partition": args.partition, "beta": args.beta,
                    "seed": args.partition_seed, "clients": args.num_total_users,
                    "min_samples": args.min_client_samples, "format": "small-image-fssl.v1"}
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        filename = partition_directory() / (args.dataset + "_fssl_" + digest + ".npz")
        if not partition_exists(filename):
            source = raw_dataset(args, True)
            images = source.data if args.dataset == "cifar10" else source.data.transpose(0, 2, 3, 1)
            targets = np.asarray(source.targets if args.dataset == "cifar10" else source.labels, dtype=np.int64)
            indices = partition_indices(targets, args.num_total_users, args.partition, args.beta,
                                        args.partition_seed, args.min_client_samples)
            save_partition(filename, {client: {"data": images[selection], "target": targets[selection]}
                                      for client, selection in enumerate(indices)})
    partition = load_partition(filename)
    if set(partition) != set(range(args.num_total_users)):
        raise ValueError("partition_file client IDs/count do not match num_total_users")
    training = []
    for client in range(args.num_total_users):
        entry = partition[client]
        targets = np.asarray(entry["target"])
        if (targets.ndim != 1 or targets.dtype.kind not in "iu" or len(targets) < args.min_client_samples
                or np.any(targets < 0) or np.any(targets >= args.num_classes)
                or entry["data"].dtype != np.uint8
                or entry["data"].shape != (len(targets), 32, 32, 3)):
            raise ValueError("Invalid small-image client data: {}".format(client))
        training.append(SimpleNamespace(images=entry["data"], targets=targets))
    evaluation = raw_dataset(args, False, basic)
    labeled_targets = np.concatenate([client.targets for client in training[:args.num_labeled_users]])
    counts = np.bincount(labeled_targets, minlength=args.num_classes)
    summary = {"partition_file": str(filename.resolve()), "datadir": args.datadir,
               "partition_source": "explicit_shared_cache" if explicit else "seeded_shared_cache",
               "partition_seed": None if explicit else args.partition_seed,
               "partition_parameters_applied": not explicit,
               "evaluation_split": "test", "train_images": sum(len(client.targets) for client in training),
               "val_images": len(evaluation), "client_sizes": [len(client.targets) for client in training],
               "labeled_clients": list(range(args.num_labeled_users)), "labeled_class_counts": counts.tolist(),
               "classes_without_labels": np.flatnonzero(counts == 0).tolist()}
    return training, evaluation, summary
