"""Shared command-line options and backward-compatible YAML configuration."""

import argparse
import json
import math
import os
from pathlib import Path

from experiments.paths import project_path, source_directory


SERVER_TYPES = {
    "fedcyon": ("Server.serverBalanced", "Balanced", "FedCyon", 10),
    "fixmatch": ("Server.serverFixMatch", "FixMatch", "FixMatch", 10),
    "fedconsist": ("Server.serverFedConsist", "FedConsist", "meanteacher", 10),
    "cbafed": ("Server.serverCBAFed", "CBAFed", "CBAFed", 10),
    "rscfed": ("Server.serverRSCFed", "RSCFed", "meanteacher", 20),
    "isofed": ("Server.serverIsoFed", "IsoFed", "IsoFed", 10),
    "fedpdm": ("Server.serverImageNet100Baselines", "ImageNet100Baseline", "fedpdm", 10),
    "sage": ("Server.serverImageNet100Baselines", "ImageNet100Baseline", "sage", 10),
}


def indexed_backend(args):
    return args.dataset == "imagenet100" or (args.method in ("fedpdm", "sage")
                                            and args.dataset in ("cifar10", "svhn"))


def server_type_spec(args):
    if args.dataset == "imagenet100":
        if args.method == "fedcyon":
            return "Server.serverImageNet100", "ImageNet100FedCyon"
        if args.method in ("fixmatch", "fedconsist", "rscfed"):
            return "Server.serverImageNet100Legacy", "ImageNet100Legacy"
    if args.method in ("fedpdm", "sage") and args.dataset in ("cifar10", "svhn"):
        return "Server.serverSmallImageBaselines", "SmallImageBaseline"
    return SERVER_TYPES[args.method][:2]


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in ("yes", "true", "t", "y", "1", "store_true"):
        return True
    if normalized in ("no", "false", "f", "n", "0", "store_false"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(filename):
    try:
        import yaml
    except ImportError as error:
        raise ValueError("Reading YAML requires PyYAML: pip install PyYAML") from error
    path = Path(filename).expanduser()
    if not path.is_file():
        path = project_path(filename)
    if not path.is_file() and (Path(filename).parent.name == "learning_config" or Path(filename).parent == Path(".")):
        path = project_path("config/experiments") / Path(filename).name
    if not path.is_file():
        raise ValueError("Configuration file does not exist: {}".format(filename))
    with path.open(encoding="utf-8") as stream:
        try:
            values = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError("Invalid YAML configuration: {}".format(error)) from error
    if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
        raise ValueError("Configuration must be a YAML mapping with string keys.")
    values = dict(values)
    if "sn" in values:
        values.setdefault("save_name", values.pop("sn"))
    return values, str(path.resolve())


def build_parser(method):
    parser = argparse.ArgumentParser(description="Federated semi-supervised learning experiments")
    parser.add_argument("--method", choices=SERVER_TYPES, default=method)
    parser.add_argument("--c", "--config", dest="c", default=None, help="YAML experiment configuration")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration without importing training libraries")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare indexed ImageNet-100 or small-image baseline data without CUDA/training")

    federated = parser.add_argument_group("Federated training")
    federated.add_argument("--aggregate_method_server", default="FedAvg", choices=["pFedMe", "PerAvg", "FedAvg", "FedAvgAdd", "pFedAT", "pFedATold", "pFedAMP", "pFedAMPold"])
    federated.add_argument("--name_algorithm", default=SERVER_TYPES[method][2], help="Client loss/augmentation algorithm, not server selection")
    federated.add_argument("--runtimes", type=int, default=1)
    federated.add_argument("--num_global_iters", type=int, default=5)
    federated.add_argument("--local_epochs", type=int, default=1)
    federated.add_argument("--local_iterations", type=int, default=5)
    federated.add_argument("--num_total_users", type=int, default=60)
    federated.add_argument("--num_subset_users", type=int, default=12)
    federated.add_argument("--gpu", type=int, default=0, help="CUDA device index; current server implementations require CUDA")
    federated.add_argument("--work_mode", type=int, default=0, choices=range(7))

    outputs = parser.add_argument_group("Saving and loading")
    outputs.add_argument("--save_dir", default="outputs/saved_models")
    outputs.add_argument("-sn", "--save_name", default="fixmatch")
    outputs.add_argument("--resume", type=str2bool, nargs="?", const=True, default=False)
    outputs.add_argument("--load_path", default=None)
    outputs.add_argument("-o", "--overwrite", type=str2bool, nargs="?", const=True, default=False)

    training = parser.add_argument_group("Semi-supervised training")
    training.add_argument("-nl", "--num_labels", type=int, default=40)
    training.add_argument("-bsz", "--batch_size", type=int, default=64)
    training.add_argument("--uratio", type=int, default=7)
    training.add_argument("--eval_batch_size", type=int, default=1024)
    training.add_argument("--hard_label", type=str2bool, default=True)
    training.add_argument("--T", type=float, default=1.0 if method in ("fixmatch", "cbafed", "isofed") else 0.5)
    training.add_argument("--p_cutoff", type=float, default=0.5 if method in ("fixmatch", "cbafed", "isofed") else 0.95)
    training.add_argument("--ema_m", type=float, default=0.999)
    training.add_argument("--ulb_loss_ratio", type=float, default=1.0)

    data = parser.add_argument_group("Data and partitions")
    data.add_argument("--data_dir", default="data/data_generate_result")
    data.add_argument("--source_dir", default=os.environ.get("BETELGEUSE_SOURCE_DIR", "data/datasource"), help="Root containing raw dataset directories")
    data.add_argument("--datadir", default=None, help="Override raw dataset directory for this experiment")
    data.add_argument("--metrics_dir", default="outputs/metrics")
    data.add_argument("--partition_file", default=None, help="FedPDM/SAGE CIFAR-10/SVHN: reuse a shared client partition, or 'none' for a seeded partition")
    data.add_argument("--download_data", type=str2bool, default=False, help="FedPDM/SAGE CIFAR-10/SVHN: allow torchvision raw data downloads")
    data.add_argument("-ds", "--dataset", default="cifar10")
    data.add_argument("-ndt", "--name_data_total", default="data_total_cifar10_NonIID_alpha0.3_num60_pro123.npy")
    data.add_argument("--train_sampler", default="RandomSampler")
    data.add_argument("-nc", "--num_classes", type=int, default=10)
    data.add_argument("--num_workers", type=int, default=0)
    data.add_argument("--beta", type=float, default=10)
    data.add_argument("--partition", default="noniid-labeldir")
    data.add_argument("--noise", type=float, default=0)
    data.add_argument("--imagenet_classes", default="config/datasets/imagenet100_cmc.txt", help="Ordered list of 100 ImageNet synsets")
    data.add_argument("--imagenet_val_manifest", default=None, help="Optional flat-validation mapping: filename synset, relative to val/")
    data.add_argument("--partition_seed", type=int, default=2026)
    data.add_argument("--num_labeled_users", type=int, default=2, help="Indexed data paths: first N clients have labels")
    data.add_argument("--min_client_samples", type=int, default=10)
    data.add_argument("--eval_interval", type=int, default=10, help="Indexed paths: evaluate every N completed rounds and at the end")

    model = parser.add_argument_group("Model")
    model.add_argument("--name_net", default="WideResNet")
    model.add_argument("--net_from_name", type=str2bool, default=False)
    model.add_argument("--depth", type=int, default=28)
    model.add_argument("--widen_factor", type=int, default=2)
    model.add_argument("--leaky_slope", type=float, default=0.1)
    model.add_argument("--dropout", type=float, default=0)
    model.add_argument("--pretrained", type=str2bool, default=False, help="ImageNet-100: explicitly opt in to ImageNet-1K supervised weights")

    optimizer = parser.add_argument_group("Optimization")
    optimizer.add_argument("--name_optimizer", default="SGD")
    optimizer.add_argument("--lr", type=float, default=0.03)
    optimizer.add_argument("--momentum", type=float, default=0.9)
    optimizer.add_argument("--weight_decay", type=float, default=0.0005)
    optimizer.add_argument("--amp", type=str2bool, default=False)
    optimizer.add_argument("--clip", type=float, default=0)
    optimizer.add_argument("--save_path_name", default="noname")

    methods = parser.add_argument_group("Method-specific settings")
    methods.add_argument("--seed_offset", type=int, default=SERVER_TYPES[method][3])
    methods.add_argument("--S_M", type=int, default=4)
    methods.add_argument("--S_K", type=int, default=12)
    methods.add_argument("--S_K_lb", type=int, default=2)
    methods.add_argument("--selected_mode", type=int, default=0)
    methods.add_argument("--re_input_ratio", type=int, default=1)
    methods.add_argument("--conlb_loss_ratio", type=float, default=1.0)
    methods.add_argument("--model_m", type=float, default=0)
    methods.add_argument("--dist_scale_f", type=int, default=1)
    methods.add_argument("--input_ratio_lb_percent", type=float, default=0.7)
    methods.add_argument("--ta_threshold", type=float, default=0.95, help="ImageNet-100: confidence threshold for FedCyon TA")
    methods.add_argument("--sage_half_gap", type=float, default=0.05, help="SAGE: confidence gap where the local target weight is 0.5")
    methods.add_argument("--pdm_prior_ema", type=float, default=0.9)
    methods.add_argument("--pdm_sinkhorn_power", type=float, default=3.0)
    methods.add_argument("--pdm_sinkhorn_iters", type=int, default=30)
    methods.add_argument("--pdm_logit_strength", type=float, default=1.0)
    methods.add_argument("--pdm_rampup_rounds", type=int, default=20)
    return parser


def parse_args(argv=None, method=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--method", choices=SERVER_TYPES, default=method or "fedcyon")
    preliminary.add_argument("--c", "--config", dest="c")
    preliminary.add_argument("-ds", "--dataset", default=None)
    selected, _ = preliminary.parse_known_args(argv)
    parser = build_parser(selected.method)
    if method is not None and selected.method != method:
        parser.error("This compatibility entrypoint selects {}; use train.py to select another method.".format(method))
    defaults = {}
    if selected.c:
        try:
            defaults, config_path = load_config(selected.c)
            reserved = {"method", "c", "dry_run", "prepare_data"}.intersection(defaults)
            if reserved:
                raise ValueError("These options belong on the command line: {}".format(", ".join(sorted(reserved))))
            for option in parser._actions:
                if option.dest in defaults and option.type is str2bool:
                    defaults[option.dest] = str2bool(defaults[option.dest])
            if defaults.get("load_path") in ("None", "null", ""):
                defaults["load_path"] = None
            defaults["c"] = config_path
        except (ValueError, argparse.ArgumentTypeError) as error:
            parser.error(str(error))
    dataset = selected.dataset or defaults.get("dataset", "cifar10")
    if dataset.lower() == "imagenet100":
        parser.set_defaults(num_classes=100, name_net="resnet18", name_algorithm="fixmatch",
                            num_total_users=20, num_subset_users=20, batch_size=16, uratio=1,
                            eval_batch_size=64, save_name=selected.method + "_imagenet100", T=1.0, p_cutoff=0.55,
                            hard_label=False)
        if selected.method in ("fedpdm", "sage"):
            parser.set_defaults(name_algorithm=selected.method, p_cutoff=0.95,
                                hard_label=selected.method == "fedpdm", ema_m=0.99)
        elif selected.method in ("fixmatch", "fedconsist", "rscfed"):
            parser.set_defaults(name_algorithm=selected.method, p_cutoff=0.95 if selected.method == "fixmatch" else 0.5,
                                hard_label=selected.method == "fixmatch")
    elif selected.method in ("fedpdm", "sage") and dataset.lower() in ("cifar10", "svhn"):
        parser.set_defaults(num_classes=10, name_net="cifar_resnet18" if dataset.lower() == "cifar10" else "vgg19_bn",
                            name_algorithm=selected.method, num_total_users=20, num_subset_users=20,
                            batch_size=64, uratio=1, eval_batch_size=256, T=1.0, p_cutoff=0.95,
                            hard_label=selected.method == "fedpdm", ema_m=0.99,
                            save_name=selected.method + "_" + dataset.lower())
    args = parser.parse_args(argv, namespace=argparse.Namespace(**defaults))
    args.dataset = args.dataset.lower()
    if args.partition_file is not None and str(args.partition_file).lower() in ("none", "null", ""):
        args.partition_file = None
    if args.runtimes < 1 or args.num_total_users < 1:
        parser.error("runtimes and num_total_users must be positive")
    for key in ("save_dir", "data_dir", "metrics_dir", "source_dir"):
        setattr(args, key, str(project_path(getattr(args, key)).resolve()))
    args.datadir = str(project_path(args.datadir).resolve()) if args.datadir else str(source_directory(args.dataset, args.source_dir).resolve())
    args.imagenet_classes = str(project_path(args.imagenet_classes).resolve())
    if args.imagenet_val_manifest:
        args.imagenet_val_manifest = str(project_path(args.imagenet_val_manifest).resolve())
    if args.partition_file:
        args.partition_file = str(project_path(args.partition_file).resolve())
    if args.dataset.lower() == "imagenet100":
        args.dataset = "imagenet100"
        args.cpu_client_state = True
        if args.method not in ("fedcyon", "fixmatch", "fedconsist", "rscfed", "fedpdm", "sage"):
            parser.error("ImageNet-100 supports fedcyon, fixmatch, fedconsist, rscfed, fedpdm and sage")
        if args.num_classes != 100 or args.name_net not in ("resnet18", "resnet50"):
            parser.error("ImageNet-100 requires num_classes=100 and name_net=resnet18 or resnet50")
        expected_algorithm = "fixmatch" if args.method == "fedcyon" else args.method
        if args.name_algorithm != expected_algorithm:
            parser.error("ImageNet-100 --method {} requires name_algorithm={}; check the method/config pair".format(
                args.method, expected_algorithm))
    if args.method in ("fedpdm", "sage"):
        if args.dataset not in ("imagenet100", "cifar10", "svhn"):
            parser.error("FedPDM/SAGE support imagenet100, cifar10 and svhn only")
        if args.name_algorithm != args.method:
            parser.error("FedPDM/SAGE require matching name_algorithm and --method")
        if args.dataset != "imagenet100":
            expected_net = "cifar_resnet18" if args.dataset == "cifar10" else "vgg19_bn"
            if args.num_classes != 10 or args.name_net != expected_net or args.pretrained:
                parser.error("Small-image baselines require num_classes=10, name_net={} and pretrained=false".format(expected_net))
    if args.partition_file and not (args.method in ("fedpdm", "sage") and args.dataset in ("cifar10", "svhn")):
        parser.error("partition_file is supported only by FedPDM/SAGE CIFAR-10/SVHN adapters")
    if args.download_data and not (args.method in ("fedpdm", "sage") and args.dataset in ("cifar10", "svhn")):
        parser.error("download_data is supported only by FedPDM/SAGE CIFAR-10/SVHN adapters")
    if indexed_backend(args):
        args.cpu_client_state = True
        if args.train_sampler != "RandomSampler":
            parser.error("Indexed data paths require train_sampler=RandomSampler")
        if not 1 <= args.num_labeled_users < args.num_total_users:
            parser.error("num_labeled_users must leave at least one labeled and one unlabeled client")
        if not 2 <= args.num_subset_users <= args.num_total_users:
            parser.error("Indexed data paths require 2 <= num_subset_users <= num_total_users")
        if args.partition not in ("homo", "noniid-labeldir") or not math.isfinite(args.beta) or args.beta <= 0:
            parser.error("Indexed data paths require partition=homo or noniid-labeldir and finite beta > 0")
        if any(value < 1 for value in (args.num_global_iters, args.local_iterations, args.batch_size,
                                      args.uratio, args.eval_batch_size, args.eval_interval, args.min_client_samples)):
            parser.error("Iteration counts, batch sizes, eval_interval and min_client_samples must be positive")
        if args.num_workers < 0 or not 0 <= args.partition_seed < 2 ** 32:
            parser.error("num_workers must be nonnegative and partition_seed must fit uint32")
        if not 0 <= args.p_cutoff <= 1 or not 0 <= args.ta_threshold <= 1 or not 0 <= args.ema_m <= 1 or args.T <= 0:
            parser.error("Invalid indexed-data confidence/EMA thresholds or temperature")
        if args.name_optimizer not in ("SGD", "AdamW") or (args.name_optimizer == "SGD" and args.momentum <= 0):
            parser.error("Indexed data paths support SGD with positive momentum, or AdamW")
        if args.resume or args.load_path:
            parser.error("Checkpoint loading/resume is not implemented for indexed data paths")
    elif args.prepare_data:
        parser.error("--prepare-data supports ImageNet-100 and FedPDM/SAGE CIFAR-10/SVHN")
    if indexed_backend(args) and args.method != "fedcyon":
        if args.aggregate_method_server != "FedAvg" or args.local_epochs != 1:
            parser.error("New baselines use method-specific weighted averaging and a fixed local_iterations budget; set aggregate_method_server=FedAvg and local_epochs=1")
        if args.method in ("fedpdm", "sage") and args.hard_label != (args.method == "fedpdm"):
            parser.error("FedPDM requires hard_label=true; SAGE requires hard_label=false")
        if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.momentum,
                                                     args.T, args.ulb_loss_ratio, args.clip)):
            parser.error("Baseline optimizer/loss settings must be finite")
        if args.lr <= 0 or args.weight_decay < 0 or args.ulb_loss_ratio < 0 or args.clip < 0:
            parser.error("Baseline lr must be positive; weight_decay, ulb_loss_ratio and clip must be nonnegative")
        if args.method == "sage" and (not math.isfinite(args.sage_half_gap) or args.sage_half_gap <= 0):
            parser.error("sage_half_gap must be finite and positive")
        if args.method == "fedpdm":
            if not 0 <= args.pdm_prior_ema < 1 or not 0 <= args.ema_m < 1:
                parser.error("FedPDM prior/model EMA decays must be in [0, 1)")
            if not math.isfinite(args.pdm_sinkhorn_power) or args.pdm_sinkhorn_power <= 0 or args.pdm_sinkhorn_iters < 1:
                parser.error("FedPDM Sinkhorn power and iteration count must be positive and finite")
            if not math.isfinite(args.pdm_logit_strength) or args.pdm_logit_strength < 0 or args.pdm_rampup_rounds < 0:
                parser.error("FedPDM logit strength and rampup rounds must be nonnegative and finite")
        if args.method == "fedconsist" and (args.T != 1.0 or args.p_cutoff != 0.5 or args.ulb_loss_ratio != 1.0):
            parser.error("ImageNet FedConsist preserves the legacy fixed T=1, p_cutoff=0.5, ulb_loss_ratio=1 objective")
        if args.method == "rscfed":
            if args.S_M < 1 or not 1 <= args.S_K <= args.num_total_users or args.selected_mode not in (0, 1):
                parser.error("RSCFed requires S_M >= 1, 1 <= S_K <= num_total_users, selected_mode=0 or 1")
            if args.selected_mode == 1 and not (0 <= args.S_K_lb <= args.num_labeled_users and
                                                0 <= args.S_K - args.S_K_lb <= args.num_total_users - args.num_labeled_users):
                parser.error("RSCFed subgroup counts exceed available labeled/unlabeled clients")
            if (not math.isfinite(args.dist_scale_f) or args.re_input_ratio not in (0, 1) or not 0 <= args.model_m <= 1
                    or not 0 <= args.input_ratio_lb_percent <= 1 or args.dist_scale_f < 0):
                parser.error("Invalid RSCFed group weights, model momentum or distance scale")
            if args.ulb_loss_ratio != 1.0:
                parser.error("ImageNet RSCFed preserves the unweighted legacy MSE objective")
    if args.load_path:
        args.load_path = str(project_path(args.load_path).resolve())
    if args.c:
        args.c = config_path
    return args


def configuration_json(args):
    return json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True)
