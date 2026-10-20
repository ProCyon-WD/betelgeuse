"""FedCyon's ImageNet-100 path, with CPU client state and streaming evaluation."""

import copy
import json
import math
import numpy as np
import torch
import torch.nn.functional as functional
from torchvision import models

from Server.serverbase import Server
from User.userreweight_mycb import UserReweight_mycb
from datautils.data_utils.data_utils import get_data_loader
from datautils.data_utils.dataset import BasicDataset
from datautils.imagenet100 import ImageNet100Dataset, imagenet100_transforms, prepare_imagenet100
from datautils.partition_store import atomic_output
from experiments.paths import project_path


class ImageNet100FedCyon(Server):
    def __init__(self, args, logger, device, aggregate_method_server, name_algorithm, name_data_total,
                 name_net, name_optimizer, curtimes, dataset, datadir, partition, num_users, beta):
        super().__init__(args, logger, device, aggregate_method_server, name_algorithm, name_data_total,
                         name_net, name_optimizer, curtimes, dataset, datadir, partition, num_users, beta,
                         initialize_evaluation=False)
        self.balanced_k = 1.0
        weak, strong, basic = self._transforms()
        training_sets, evaluation, self.data_summary = self._prepare_data(basic)
        self.classescount = np.asarray(self.data_summary["labeled_class_counts"], dtype=np.int64)
        if self.data_summary["classes_without_labels"]:
            logger.warning("Selected labeled clients do not cover these classes: %s",
                           self.data_summary["classes_without_labels"])
        instance_model = self._build_model()
        self.model = copy.deepcopy(instance_model).to(device)
        self.loader_dict_eval["eval"] = torch.utils.data.DataLoader(
            evaluation, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        for client_id, training in enumerate(training_sets):
            labeled = BasicDataset("fixmatch", training.images, training.targets, args.num_classes, weak,
                                   basic_transform=basic)
            unlabeled = BasicDataset("fixmatch", training.images, None, args.num_classes, weak,
                                     is_ulb=True, strong_transform=strong)
            tag = 0 if client_id < args.num_labeled_users else 2
            loader = get_data_loader(labeled if tag == 0 else unlabeled,
                                     args.batch_size if tag == 0 else args.batch_size * args.uratio,
                                     num_iters=args.local_iterations, data_sampler=args.train_sampler,
                                     num_workers=args.num_workers, pin_memory=True)
            loaders = {"train_lb": loader if tag == 0 else None,
                       "train_ulb": loader if tag == 2 else None}
            user = self._create_user(client_id, tag, labeled, unlabeled, loaders, instance_model)
            self.users.append(user)
        self.evaluations = []
        self.training_history = []
        self.result_dir = project_path(args.metrics_dir) / args.dataset / args.save_name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        run_dir = project_path(args.save_dir) / args.save_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with atomic_output(run_dir / (args.dataset + "_data.json"), "w") as stream:
            json.dump(self.data_summary, stream, ensure_ascii=False, indent=2)
        logger.warning("%s: %s train, %s evaluation images; labeled class counts: %s",
                       args.dataset, self.data_summary["train_images"], self.data_summary["val_images"], self.classescount)

    def _transforms(self):
        return imagenet100_transforms()

    def _prepare_data(self, basic):
        manifest, clients, summary = prepare_imagenet100(self.args)
        training = [ImageNet100Dataset(self.datadir, manifest, "train", None, indices) for indices in clients]
        return training, ImageNet100Dataset(self.datadir, manifest, "val", basic), summary

    def _build_model(self):
        args = self.args
        if args.pretrained:
            self.logger.warning("ImageNet-1K supervised pretraining is extra supervision overlapping ImageNet-100")
        model_type, weights_type = ((models.resnet18, models.ResNet18_Weights) if args.name_net == "resnet18"
                                    else (models.resnet50, models.ResNet50_Weights))
        model = model_type(weights=weights_type.DEFAULT if args.pretrained else None)
        model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
        return model

    def _create_user(self, client_id, tag, labeled, unlabeled, loaders, model):
        args = self.args
        user = UserReweight_mycb(client_id, tag, labeled, unlabeled, loaders, model,
                                args, self.logger, self.device, args.aggregate_method_server,
                                args.name_algorithm, args.name_data_total, args.name_net,
                                args.name_optimizer, 100)
        user.set_global_distribution(self.classescount)
        return user

    def _select_clients(self):
        labeled_count = self.args.num_labeled_users
        unlabeled_count = self.num_total_users - labeled_count
        selected = self.num_subset_users
        selected_labeled = min(labeled_count, selected - 1,
                               max(1, selected - unlabeled_count,
                                   int(labeled_count * selected / self.num_total_users)))
        labeled = np.random.choice(labeled_count, selected_labeled, replace=False).tolist()
        unlabeled = np.random.choice(np.arange(labeled_count, self.num_total_users),
                                     selected - selected_labeled, replace=False).tolist()
        return labeled, unlabeled

    @staticmethod
    def _accumulate_state(accumulated, model):
        for name, value in model.state_dict().items():
            value = value.detach().to(device="cpu", dtype=torch.float32, copy=True)
            if name in accumulated:
                accumulated[name].add_(value)
            else:
                accumulated[name] = value

    def _aggregate(self, group_sums, group_counts, group_ta):
        scores = [math.exp(-self.balanced_k * float(np.mean(values))) for values in group_ta]
        labeled_ratio = scores[0] / sum(scores)
        combined = {}
        for name, template in self.model.state_dict().items():
            means = [totals[name] / count for totals, count in zip(group_sums, group_counts)]
            if not template.is_floating_point():
                means = [value.to(template.dtype) for value in means]
            combined[name] = means[0] * labeled_ratio + means[1] * (1.0 - labeled_ratio)
        self.model.load_state_dict(combined)
        return labeled_ratio

    def process(self):
        self.evaluate_global_test_dataset(completed_round=0)
        self.new_result_save()
        for round_index in range(self.num_global_iters):
            self.start_training_round(round_index + 1)
            record = self._train_round(round_index)
            measurements = self.finish_training_round()
            if measurements is not None:
                record.update(measurements)
            completed = round_index + 1
            self.training_history.append(record)
            self.logger.warning("%s round %s: %s", self.dataset, completed, record)
            if completed % self.args.eval_interval == 0 or completed == self.num_global_iters:
                self.evaluate_global_test_dataset(completed_round=completed)
            self.new_result_save()

    def _train_round(self, round_index):
        selections = self._select_clients()
        group_sums = [{}, {}]
        group_ta = [[], []]
        group_losses = [[], []]
        mask_ratios = []
        for group, clients in enumerate(selections):
            for client_id in clients:
                user = self.users[client_id]
                try:
                    user.train(self.model, self.args, round_index, logger=self.logger)
                    self._accumulate_state(group_sums[group], user.model)
                    group_ta[group].append(float(user.ta))
                    loss = user.supervisedloss if group == 0 else user.unsupervisedloss
                    group_losses[group].append(float(loss.detach()) / user.local_iterations)
                    if group == 1:
                        mask_ratios.append(user.mask_ratio)
                finally:
                    user.move_training_state("cpu")
        labeled_ratio = self._aggregate(group_sums, [len(clients) for clients in selections], group_ta)
        return {"round": round_index + 1, "labeled_clients": selections[0], "unlabeled_clients": selections[1],
                "supervised_loss": float(np.mean(group_losses[0])),
                "unsupervised_loss": float(np.mean(group_losses[1])),
                "labeled_ta": float(np.mean(group_ta[0])), "unlabeled_ta": float(np.mean(group_ta[1])),
                "unlabeled_mask_ratio": float(np.mean(mask_ratios)), "labeled_ratio": labeled_ratio}

    @torch.inference_mode()
    def evaluate_global_test_dataset(self, eval_loader=None, args=None, completed_round=0):
        loader = eval_loader if eval_loader is not None else self.loader_dict_eval["eval"]
        was_training = self.model.training
        total = 0
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        self.model.eval()
        try:
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                logits = self.model(images)
                total_loss += functional.cross_entropy(logits, targets, reduction="sum").item()
                predictions = logits.topk(5, dim=1).indices
                matches = predictions.eq(targets[:, None])
                correct_top1 += matches[:, 0].sum().item()
                correct_top5 += matches.any(dim=1).sum().item()
                total += targets.numel()
        finally:
            self.model.train(was_training)
        if total == 0:
            raise ValueError("{} evaluation dataset is empty".format(self.dataset))
        result = {"round": completed_round, "loss": total_loss / total,
                  "top1": correct_top1 / total, "top5": correct_top5 / total, "samples": total}
        self.evaluations.append(result)
        self.logger.warning("%s evaluation: %s", self.dataset, result)
        return result

    def new_result_save(self):
        filename = self.result_dir / "run_{}.json".format(self.curtimes)
        payload = {"dataset": self.dataset, "method": self.args.method, "backbone": self.args.name_net,
                   "pretrained": self.args.pretrained, "client_state": "cpu", "gradient_diagnostics": False,
                   "data": self.data_summary, "evaluations": self.evaluations, "training": self.training_history}
        if hasattr(self, "implementation"):
            payload["implementation"] = self.implementation
        if hasattr(self, "training_recorder"):
            payload["training_metrics_file"] = str(self.training_recorder.filename)
        with atomic_output(filename, "w") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        with atomic_output(filename.with_suffix(".npz")) as stream:
            np.savez_compressed(stream, eval_rounds=[result["round"] for result in self.evaluations],
                                testacc=[result["top1"] for result in self.evaluations],
                                top5=[result["top5"] for result in self.evaluations],
                                loss=[result["loss"] for result in self.evaluations])
