"""Shared ImageNet-100 infrastructure, independent baseline objectives/aggregation."""

import math

import torch

from Server.serverImageNet100 import ImageNet100FedCyon
from User.user_fssl_baselines import FedPDMClient, SAGEClient


class WeightedStateAccumulator:
    def __init__(self, average_buffers=False):
        self.average_buffers = average_buffers
        self.total_weight = 0.0
        self.largest_weight = -1.0
        self.sums = {}
        self.buffers = {}

    @torch.no_grad()
    def add(self, state, weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Aggregation weights must be finite and nonnegative")
        if weight == 0:
            return
        for name, value in state.items():
            if value.is_floating_point() or self.average_buffers:
                value = value.detach().to(device="cpu", dtype=torch.float32, copy=True)
                if name not in self.sums:
                    self.sums[name] = value.mul_(weight)
                else:
                    self.sums[name].add_(value, alpha=weight)
            elif weight > self.largest_weight:
                self.buffers[name] = value.detach().cpu().clone()
        self.total_weight += weight
        self.largest_weight = max(self.largest_weight, weight)

    def mean_state(self):
        if self.total_weight == 0:
            raise ValueError("Cannot average states with zero total weight")
        state = {name: value / self.total_weight for name, value in self.sums.items()}
        state.update(self.buffers)
        return state

    @torch.no_grad()
    def apply(self, model):
        if self.total_weight == 0:
            return False
        model.load_state_dict(self.mean_state())
        return True


class ImageNet100Baseline(ImageNet100FedCyon):
    def _create_user(self, client_id, tag, labeled, unlabeled, loaders, model):
        if tag == 2:
            unlabeled.targets = None
        if self.args.method == "fedpdm":
            client_type = FedPDMClient
            self.implementation = {
                "variant": "fedpdm_pure_reimplementation",
                "source": "AAAI 2025, Progressive Distribution Matching, Algorithm 1 and Eqs. 4-7",
                "aggregation": "EMA models weighted by labeled count or current-round unique selected unlabeled count",
                "prediction_memory": "per-client, CPU, indexed by local sample; retained across rounds",
                "prior": "labeled clients use their own labels; unlabeled clients use prediction EMA only",
                "model_ema": "reset from broadcast each participation; parameters EMA, buffers copied",
                "schedule": "fixed local_iterations, constant lr; engineering defaults, not paper result reproduction",
            }
        else:
            client_type = SAGEClient
            self.implementation = {
                "variant": "sage_pure_adapted",
                "source": "Jay-Codeman/SAGE@e90fa96e0766fc2d6e4ea64625b7eec8aa1fd506",
                "protocol_difference": "pure labeled/unlabeled clients instead of mixed-label clients",
                "objective": "labeled CE; unlabeled confidence-corrected KL with frozen broadcast teacher",
                "aggregation": "online models, sample-count-weighted FedAvg; no FedCyon TA weights",
                "schedule": "fixed local_iterations, constant lr; no cross-client optimizer sharing",
            }
        self.implementation["optimizer_state"] = "per-client, retained across rounds on CPU"
        self.implementation["integer_buffers"] = "taken from the largest-weight client (first wins ties)"
        self.implementation["protocol"] = "pure_clients"
        return client_type(client_id, tag, loaders, model, self.args, self.device)

    def _train_round(self, round_index):
        selections = self._select_clients()
        accumulator = WeightedStateAccumulator(average_buffers=self.args.method in ("fixmatch", "fedconsist"))
        statistics = []
        group_losses = [[], []]
        mask_ratios = []
        for group, clients in enumerate(selections):
            for client_id in clients:
                user = self.users[client_id]
                try:
                    result = user.train(self.model, self.args, round_index)
                    accumulator.add(user.upload_state(), result["aggregation_count"])
                    result["labeled"] = group == 0
                    statistics.append(result)
                    group_losses[group].append(result["loss"])
                    if group == 1:
                        mask_ratios.append(result["mask_ratio"])
                finally:
                    user.move_training_state("cpu")
        updated = accumulator.apply(self.model)
        for result in statistics:
            result["aggregation_weight"] = (result["aggregation_count"] / accumulator.total_weight
                                            if accumulator.total_weight else 0.0)
        return {"round": round_index + 1, "labeled_clients": selections[0], "unlabeled_clients": selections[1],
                "supervised_loss": sum(group_losses[0]) / len(group_losses[0]),
                "unsupervised_loss": sum(group_losses[1]) / len(group_losses[1]),
                "unlabeled_mask_ratio": sum(mask_ratios) / len(mask_ratios),
                "labeled_ratio": sum(result["aggregation_weight"] for result in statistics if result["labeled"]),
                "aggregation_total_count": accumulator.total_weight, "model_updated": updated,
                "clients": statistics}
