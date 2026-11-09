"""Common startup, measurements and repeat-run orchestration for SSL methods."""

import gc
import importlib
import os
import random

from experiments.config import SERVER_TYPES, configuration_json, parse_args, server_type_spec
from experiments.paths import PROJECT_ROOT, project_path, source_directory


def resolve_training_device(args):
    import torch

    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
        return torch.device("cpu")
    if not torch.cuda.is_available() or args.gpu < 0:
        raise RuntimeError("CUDA training requires a CUDA-enabled PyTorch installation and --gpu >= 0. "
                           "For FedCyon/FedPDM on ImageNet-100 on a Mac, use --device cpu --amp false.")
    return torch.device("cuda:{}".format(args.gpu))


def run(args):
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import numpy as np
    import torch
    from utils.utils import get_logger
    from experiments.recording import TrainingRecorder

    save_path = project_path(args.save_dir) / args.save_name
    datadir = getattr(args, "datadir", None) or source_directory(args.dataset, getattr(args, "source_dir", None))
    seed_offset = getattr(args, "seed_offset", SERVER_TYPES[args.method][3])
    if args.resume and args.load_path is None:
        raise ValueError("Resume of training requires --load_path")
    if save_path.exists() and not args.overwrite:
        raise FileExistsError("Model directory already exists: {}".format(save_path))
    device = resolve_training_device(args)
    module_name, class_name = server_type_spec(args)
    server_type = getattr(importlib.import_module(module_name), class_name)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "resolved_config.json").write_text(configuration_json(args) + "\n", encoding="utf-8")
    logger = get_logger(args.save_name, str(save_path), "WARNING")
    logger.warning("USE DEVICE: %s for training", device)
    if device.type == "cpu":
        logger.warning("CPU intra-op threads: %s", torch.get_num_threads())
    for current_run in range(1, args.runtimes + 1):
        logger.warning("---------------Running time:------------ %s", current_run)
        seed = current_run + seed_offset
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = True
        server_args = [args, logger, device, args.aggregate_method_server, args.name_algorithm,
                       args.name_data_total, args.name_net, args.name_optimizer, current_run,
                       args.dataset, str(datadir), args.partition,
                       20 if args.method == "rscfed" and args.dataset != "imagenet100" else args.num_total_users, args.beta]
        if args.method == "fixmatch" and args.dataset != "imagenet100":
            server_args.append(args.noise)
        server = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            with TrainingRecorder(args, device, current_run, seed) as recorder:
                server = server_type(*server_args)
                recorder.attach(server)
                server.process()
        finally:
            del server
            gc.collect()
        logger.warning("Device %s training is FINISHED", device)


def cli(argv=None, method=None):
    args = parse_args(argv, method)
    if args.dry_run:
        print(configuration_json(args))
        return
    if args.prepare_data:
        import json
        if args.dataset == "imagenet100":
            from datautils.imagenet100 import prepare_imagenet100
            _, _, summary = prepare_imagenet100(args)
        else:
            from datautils.small_image_fssl import prepare_small_images, small_image_transforms
            _, _, basic = small_image_transforms(args.dataset)
            _, _, summary = prepare_small_images(args, basic)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    os.chdir(PROJECT_ROOT)
    run(args)
