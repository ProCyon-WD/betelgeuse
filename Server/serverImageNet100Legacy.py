"""ImageNet-100 adapters: equal-client FixMatch/FedConsist and subgroup RSCFed."""

import copy
import math

import numpy as np
import torch

from Server.serverImageNet100Baselines import ImageNet100Baseline, WeightedStateAccumulator
from User.user_imagenet_adapters import FixMatchClient, FedConsistClient, RSCFedClient


def rscfed_weights(states, counts, labeled, reference, scale, rebalance, labeled_ratio):
    if (not states or len(states) != len(counts) or len(states) != len(labeled)
            or any(not math.isfinite(count) or count <= 0 for count in counts)
            or not math.isfinite(scale) or scale < 0 or not 0 <= labeled_ratio <= 1):
        raise ValueError("RSCFed requires matching nonempty states, positive counts and valid weighting parameters")
    distances = [sum(float(torch.linalg.vector_norm(value.float() - reference[name].float()))
                     for name, value in state.items()) for state in states]
    logits = torch.tensor([math.log(count) - scale * distance / count
                           for count, distance in zip(counts, distances)], dtype=torch.float64)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite RSCFed model distance")
    weights = logits.softmax(dim=0)
    if rebalance:
        labeled = torch.tensor(labeled, dtype=torch.bool)
        for mask, fraction in ((labeled, labeled_ratio), (~labeled, 1 - labeled_ratio)):
            if mask.any():
                mass = fraction if (~mask).any() else 1.0
                weights[mask] = mass * logits[mask].softmax(dim=0)
    return weights.tolist(), distances


class ImageNet100Legacy(ImageNet100Baseline):
    def _create_user(self, client_id, tag, labeled, unlabeled, loaders, model):
        client_type = {"fixmatch": FixMatchClient, "fedconsist": FedConsistClient,
                       "rscfed": RSCFedClient}[self.args.method]
        self.implementation = {
            "variant": self.args.method + "_imagenet100_pure_adapter",
            "source": "Repository legacy pure-client objective and active aggregation path",
            "aggregation": "equal-client mean" if self.args.method != "rscfed" else
                           "sample-count/distance weighted subgroups, optional labeled-group rebalance, equal subgroup mean",
            "integer_buffers": "weighted mean cast back to buffer dtype at each aggregation stage",
            "protocol": "pure_clients", "optimizer_state": "retained per client on CPU",
            "schedule": "fixed local_iterations; local_epochs=1; constant lr; no unused gradient diagnostics",
        }
        if self.args.method == "fedconsist":
            self.implementation["objective"] = "CE consistency; fixed temperature=1 and cutoff=0.5; unweighted unlabeled objective"
        if self.args.method == "rscfed":
            self.implementation.update({
                "objective": "EMA probability MSE, sharpening temperature 0.5; EMA decay min(1-1/(step+1), 0.999)",
                "teacher": "persistent per-client parameters and train-mode BN; no per-round reset or buffer copy",
                "adaptation": "use positive local sample counts for BOTH client groups, not the legacy uninitialized unlabeled zero count; stable log-space weights; single-group weights sum to one",
            })
        return client_type(client_id, tag, loaders, model, self.args, self.device)

    def _train_round(self, round_index):
        if self.args.method != "rscfed":
            return super()._train_round(round_index)
        args = self.args
        if not hasattr(self, "reference_model"):
            self.reference_model = copy.deepcopy(self.model).cpu()
        subgroups = WeightedStateAccumulator(average_buffers=True)
        records = []
        groups = []
        for subgroup in range(args.S_M):
            if args.selected_mode == 0:
                clients = np.random.choice(args.num_total_users, args.S_K, replace=False).tolist()
            else:
                clients = (np.random.choice(args.num_labeled_users, args.S_K_lb, replace=False).tolist()
                           + np.random.choice(np.arange(args.num_labeled_users, args.num_total_users),
                                              args.S_K - args.S_K_lb, replace=False).tolist())
            states = []
            counts = []
            labeled = []
            group_records = []
            for client_id in clients:
                user = self.users[client_id]
                try:
                    result = user.train(self.model, args, round_index)
                    states.append({name: value.detach().to("cpu", copy=True)
                                   for name, value in user.upload_state().items()})
                    counts.append(user.num_samples)
                    labeled.append(user.tag_client_lbulb == 0)
                    result.update(labeled=labeled[-1], subgroup=subgroup + 1)
                    group_records.append(result)
                finally:
                    user.move_training_state("cpu")
            average = WeightedStateAccumulator(average_buffers=True)
            for state, count in zip(states, counts):
                average.add(state, count)
            previous = {name: value.clone() for name, value in self.reference_model.state_dict().items()}
            self.reference_model.load_state_dict({
                name: args.model_m * previous[name] + (1 - args.model_m) * value
                for name, value in average.mean_state().items()})
            weights, distances = rscfed_weights(states, counts, labeled, self.reference_model.state_dict(),
                                                args.dist_scale_f, args.re_input_ratio == 1,
                                                args.input_ratio_lb_percent)
            group_average = WeightedStateAccumulator(average_buffers=True)
            for state, weight in zip(states, weights):
                group_average.add(state, weight)
            subgroups.add({name: value.to(states[0][name].dtype)
                           for name, value in group_average.mean_state().items()}, 1.0)
            for result, weight, distance in zip(group_records, weights, distances):
                result.update(subgroup_weight=weight, aggregation_weight=weight / args.S_M,
                              distance_to_reference=distance)
            records.extend(group_records)
            groups.append({"subgroup": subgroup + 1, "clients": clients, "weights": weights})
        subgroups.apply(self.model)
        return {"round": round_index + 1, "subgroups": groups, "clients": records,
                "labeled_clients": sorted({record["client_id"] for record in records if record["labeled"]}),
                "unlabeled_clients": sorted({record["client_id"] for record in records if not record["labeled"]}),
                "labeled_ratio": sum(record["aggregation_weight"] for record in records if record["labeled"]),
                "model_updated": True}
