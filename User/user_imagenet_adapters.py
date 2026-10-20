"""Device-independent pure-client adapters of the repository's three legacy methods."""

import copy

import torch
import torch.nn.functional as functional

from User.fixmatch_utils import consistency_loss
from User.user_fssl_baselines import BaselineClient
from utils.train_utils import get_optimizer


class FixMatchClient(BaselineClient):
    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.optimizer = get_optimizer(self.model, self.args.name_optimizer, self.args.lr,
                                       self.args.momentum, self.args.weight_decay)

    def _labeled_batch(self, batch):
        return batch[1], batch[3]

    def _unsupervised_loss(self, indices, weak, strong, round_index):
        weak_logits, strong_logits = self.model(torch.cat((weak, strong))).chunk(2)
        loss, _, selected, _ = consistency_loss(strong_logits.float(), weak_logits.float(), "ce",
                                                 self.args.T, self.args.p_cutoff, self.args.hard_label)
        return loss, selected.bool()

    def aggregation_count(self):
        return 1


class FedConsistClient(FixMatchClient):
    def _unsupervised_loss(self, indices, weak, strong, round_index):
        weak_logits, strong_logits = self.model(torch.cat((weak, strong))).chunk(2)
        loss, _, selected, _ = consistency_loss(strong_logits.float(), weak_logits.float(), "ce",
                                                 1.0, 0.5, self.args.hard_label)
        return loss, selected.bool()

    def _unlabeled_weight(self):
        return 1.0


class RSCFedClient(FixMatchClient):
    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.ema_teacher = copy.deepcopy(self.model).requires_grad_(False)
        self.ema_steps = 0

    def move_training_state(self, device):
        super().move_training_state(device)
        self.ema_teacher.to(device)

    def _unsupervised_loss(self, indices, weak, strong, round_index):
        self.ema_teacher.train()
        with torch.no_grad():
            probabilities = self.ema_teacher(weak).float().softmax(dim=1)
            sharpened = probabilities.square()
            sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)
        probabilities = self.model(strong).float().softmax(dim=1)
        loss = functional.mse_loss(probabilities, sharpened, reduction="sum") / len(weak)
        return loss, torch.ones(len(weak), dtype=torch.bool, device=weak.device)

    @torch.no_grad()
    def _after_step(self):
        decay = min(1 - 1 / (self.ema_steps + 1), 0.999)
        for teacher_parameter, parameter in zip(self.ema_teacher.parameters(), self.model.parameters()):
            teacher_parameter.mul_(decay).add_(parameter, alpha=1 - decay)
        self.ema_steps += 1

    def _unlabeled_weight(self):
        return 1.0

    def aggregation_count(self):
        return self.num_samples
