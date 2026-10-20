"""ImageNet-100 manifests and reproducible, index-only client partitions."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from datautils.partition_store import atomic_output, partition_directory


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
MANIFEST_FORMAT = "betelgeuse.imagenet100.manifest.v1"
PARTITION_FORMAT = "betelgeuse.imagenet100.indices.v1"


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_classes(filename):
    classes = [line.strip() for line in Path(filename).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if len(classes) != 100 or len(set(classes)) != 100:
        raise ValueError("ImageNet-100 requires exactly 100 distinct synsets in --imagenet_classes")
    if any(re.fullmatch(r"n[0-9]{8}", synset) is None for synset in classes):
        raise ValueError("Each ImageNet-100 class must be a synset such as n02869837")
    return classes


def _folder_samples(root, split, classes):
    samples = []
    for label, synset in enumerate(classes):
        directory = root / split / synset
        if not directory.is_dir():
            hint = (" Flat validation images require --imagenet_val_manifest "
                    "(filename synset per line)." if split == "val" else "")
            raise ValueError("Missing class directory: {}.{}".format(directory, hint))
        images = sorted(image for image in directory.rglob("*")
                        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError("No images found in {}".format(directory))
        samples.extend([image.relative_to(root).as_posix(), label] for image in images)
    return samples


def _validation_samples(root, filename, classes):
    class_to_idx = {synset: label for label, synset in enumerate(classes)}
    samples = []
    seen = set()
    for line_number, line in enumerate(Path(filename).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or re.fullmatch(r"n[0-9]{8}", fields[1]) is None:
            raise ValueError("Invalid validation manifest line {}: expected filename synset, "
                             "not numeric ILSVRC class IDs".format(line_number))
        relative, synset = fields
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Validation filenames must be relative to <datadir>/val")
        if synset not in class_to_idx:
            continue
        image = root / "val" / relative
        if not image.is_file():
            raise ValueError("Missing validation image: {}".format(image))
        name = image.relative_to(root).as_posix()
        if name in seen:
            raise ValueError("Duplicate validation image: {}".format(name))
        seen.add(name)
        samples.append([name, class_to_idx[synset]])
    if set(label for _, label in samples) != set(range(100)):
        raise ValueError("Validation manifest must contain images for all selected 100 classes")
    return sorted(samples)


def build_manifest(datadir, classes_file, val_manifest=None, cache_dir=None):
    root = Path(datadir).expanduser().resolve()
    classes = read_classes(classes_file)
    payload = {"format": MANIFEST_FORMAT, "classes": classes,
               "train": _folder_samples(root, "train", classes),
               "val": (_validation_samples(root, val_manifest, classes) if val_manifest
                       else _folder_samples(root, "val", classes))}
    fingerprint = _digest(payload)
    directory = Path(cache_dir) if cache_dir is not None else partition_directory()
    filename = directory / "_shared" / "imagenet100" / (fingerprint + ".json")
    if filename.exists():
        if json.loads(filename.read_text(encoding="utf-8")) != payload:
            raise ValueError("ImageNet-100 manifest checksum mismatch: {}".format(filename))
    else:
        with atomic_output(filename, "w") as stream:
            json.dump(payload, stream, separators=(",", ":"))
    return payload, filename, fingerprint


def partition_indices(labels, num_users, partition, beta, seed, min_samples=10):
    labels = np.asarray(labels, dtype=np.int64)
    if num_users < 2 or min_samples < 1 or len(labels) < num_users * min_samples:
        raise ValueError("Not enough training samples for the requested clients/min_client_samples")
    if partition not in ("homo", "noniid-labeldir"):
        raise ValueError("ImageNet-100 supports homo and noniid-labeldir partitions")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    generator = np.random.RandomState(seed)
    if partition == "homo":
        return [indices.astype(np.int64) for indices in np.array_split(generator.permutation(len(labels)), num_users)]
    for attempt in range(1000):
        clients = [[] for _ in range(num_users)]
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            generator.shuffle(indices)
            proportions = generator.dirichlet(np.repeat(beta, num_users))
            proportions *= np.array([len(client) < len(labels) / num_users for client in clients])
            if proportions.sum() == 0:
                proportions = np.ones(num_users)
            proportions /= proportions.sum()
            boundaries = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client, assigned in zip(clients, np.split(indices, boundaries)):
                client.extend(assigned.tolist())
        if min(map(len, clients)) >= min_samples:
            result = []
            for client in clients:
                generator.shuffle(client)
                result.append(np.asarray(client, dtype=np.int64))
            return result
    raise ValueError("Cannot construct this Dirichlet partition after 1000 attempts; "
                     "increase beta or reduce clients/min_client_samples")


def load_or_create_partition(manifest, fingerprint, args, cache_dir=None):
    directory = Path(cache_dir) if cache_dir is not None else partition_directory()
    metadata = {"format": PARTITION_FORMAT, "manifest_sha256": fingerprint,
                "partition": args.partition, "beta": args.beta, "seed": args.partition_seed,
                "num_users": args.num_total_users, "num_labeled_users": args.num_labeled_users,
                "min_samples": args.min_client_samples}
    filename = directory / "partitions" / "imagenet100" / (_digest(metadata) + ".npz")
    labels = np.asarray([label for _, label in manifest["train"]], dtype=np.int64)
    if filename.exists():
        with np.load(filename, allow_pickle=False) as archive:
            if json.loads(archive["metadata"].tobytes().decode()) != metadata:
                raise ValueError("ImageNet-100 partition metadata mismatch")
            clients = [archive["client_{}".format(client)] for client in range(args.num_total_users)]
    else:
        clients = partition_indices(labels, args.num_total_users, args.partition, args.beta,
                                    args.partition_seed, args.min_client_samples)
        arrays = {"client_{}".format(client): indices for client, indices in enumerate(clients)}
        arrays["metadata"] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
        with atomic_output(filename) as stream:
            np.savez_compressed(stream, **arrays)
    if any(indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < args.min_client_samples
           for indices in clients):
        raise ValueError("Invalid ImageNet-100 client indices")
    combined = np.concatenate(clients)
    if not np.array_equal(np.sort(combined), np.arange(len(labels))):
        raise ValueError("Client indices must cover every training image exactly once")
    return clients, filename


def prepare_imagenet100(args):
    manifest, manifest_file, fingerprint = build_manifest(
        args.datadir, args.imagenet_classes, args.imagenet_val_manifest)
    clients, partition_file = load_or_create_partition(manifest, fingerprint, args)
    labels = np.asarray([label for _, label in manifest["train"]], dtype=np.int64)
    labeled_indices = np.concatenate(clients[:args.num_labeled_users])
    distribution = np.bincount(labels[labeled_indices], minlength=100)
    summary = {"manifest": str(manifest_file), "manifest_sha256": fingerprint,
               "partition_file": str(partition_file), "datadir": str(Path(args.datadir).resolve()),
               "class_to_idx": {synset: label for label, synset in enumerate(manifest["classes"])},
               "train_images": len(labels), "val_images": len(manifest["val"]),
               "partition_seed": args.partition_seed, "client_sizes": [len(indices) for indices in clients],
               "labeled_clients": list(range(args.num_labeled_users)),
               "labeled_class_counts": distribution.tolist(),
               "classes_without_labels": [manifest["classes"][label]
                                          for label in np.flatnonzero(distribution == 0)]}
    return manifest, clients, summary


class IndexedImageFiles:
    """Read a single RGB image on demand; never materialize an image bank."""

    def __init__(self, root, samples, indices=None):
        self.root = str(root)
        self.samples = samples
        self.indices = np.arange(len(samples)) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        relative, _ = self.samples[int(self.indices[index])]
        with Image.open(Path(self.root) / relative) as image:
            return image.convert("RGB")


class ImageNet100Dataset:
    def __init__(self, root, manifest, split, transform, indices=None):
        self.images = IndexedImageFiles(root, manifest[split], indices)
        self.targets = np.asarray([manifest[split][int(index)][1] for index in self.images.indices], dtype=np.int64)
        self.classes = manifest["classes"]
        self.class_to_idx = {synset: label for label, synset in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        return (self.transform(image) if self.transform else image), int(self.targets[index])


def imagenet100_transforms():
    from torchvision import transforms
    from datautils.data_utils.augmentation.randaugment import RandAugment

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    weak = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), normalize])
    strong = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                 RandAugment(3, 5), transforms.ToTensor(), normalize])
    basic = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                transforms.ToTensor(), normalize])
    return weak, strong, basic
