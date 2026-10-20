"""FedPDM reimplementation and SAGE adaptation for the pure-client protocol."""

import copy
import math

import torch
import torch.nn.functional as functional

from experiments.recording import record_training_loss


@torch.no_grad()
def match_class_distribution(probabilities, prior, power, iterations):
    """Log-domain Sinkhorn with unit row mass and column mass N * prior."""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("Sinkhorn requires a nonempty prediction matrix")
    if prior.shape != probabilities.shape[1:]:
        raise ValueError("Prediction matrix and prior have incompatible class dimensions")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(prior).all():
        raise ValueError("Sinkhorn inputs must be finite")
    if (probabilities < 0).any() or (prior < 0).any() or prior.sum() <= 0:
        raise ValueError("Sinkhorn inputs must be nonnegative with positive prior mass")
    if not math.isfinite(power) or power <= 0 or iterations < 1:
        raise ValueError("Sinkhorn power and iterations must be positive")
    log_kernel = probabilities.float().clamp_min(1e-12).log() * power
    prior = prior.to(log_kernel).clamp_min(1e-6)
    prior = prior / prior.sum()
    log_columns = (prior * len(probabilities)).log()
    column_scale = torch.zeros_like(prior)
    for _ in range(iterations):
        row_scale = -torch.logsumexp(log_kernel + column_scale[None, :], dim=1)
        column_scale = log_columns - torch.logsumexp(log_kernel + row_scale[:, None], dim=0)
    return torch.softmax(log_kernel + column_scale[None, :], dim=1)


def sage_consistency_loss(local_logits, global_logits, strong_logits, threshold, temperature, half_gap):
    """Confidence discrepancy correction and masked KL from the SAGE author code."""
    with torch.no_grad():
        local_probability = (local_logits.detach().float() / temperature).softmax(dim=1)
        global_probability = (global_logits.detach().float() / temperature).softmax(dim=1)
        local_confidence, local_class = local_probability.max(dim=1)
        global_confidence, global_class = global_probability.max(dim=1)
        local_mask = local_confidence >= threshold
        mask = local_mask | (global_confidence >= threshold)
        gap = (local_confidence - global_confidence).abs().add(1e-6).clamp(max=1.0)
        local_weight = torch.exp(-math.log(2.0) / half_gap * gap).clamp(min=1e-6, max=1.0)
        local_target = functional.one_hot(local_class, strong_logits.shape[1]).float()
        global_target = functional.one_hot(global_class, strong_logits.shape[1]).float()
        corrected = local_weight[:, None] * local_target + (1 - local_weight[:, None]) * global_target
        targets = torch.where(local_mask[:, None], corrected, global_target)
    per_sample = functional.kl_div(strong_logits.float().log_softmax(dim=1), targets,
                                   reduction="none").sum(dim=1)
    return (per_sample * mask).mean(), mask, targets


class BaselineClient:
    def __init__(self, client_id, tag, loaders, model, args, device):
        if tag not in (0, 2):
            raise ValueError("These baseline adapters support pure labeled/unlabeled clients only")
        self.id_user = client_id
        self.tag_client_lbulb = tag
        self.args = args
        self.device = torch.device(device)
        self.loader = loaders["train_lb" if tag == 0 else "train_ulb"]
        self.num_samples = len(self.loader.dataset)
        self.model = copy.deepcopy(model).cpu()
        self.teacher = None
        if args.name_optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
                                             momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr,
                                               weight_decay=args.weight_decay)

    def move_training_state(self, device):
        for parameter in self.model.parameters():
            parameter.data = parameter.data.to(device)
            if parameter.grad is not None:
                parameter.grad = parameter.grad.to(device)
        self.model.to(device)
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(device)
        if torch.device(device).type == "cpu":
            self.teacher = None

    def _start_round(self, global_model):
        self.move_training_state(self.device)
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()

    def _new_teacher(self):
        self.teacher = copy.deepcopy(self.model).eval().requires_grad_(False)

    def train(self, global_model, args, round_index, logger=None):
        self._start_round(global_model)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and self.device.type == "cuda")
        iterator = iter(self.loader)
        losses = []
        accepted_samples = 0
        seen_samples = 0
        for _ in range(args.local_iterations):
            batch = next(iterator)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=args.amp and self.device.type == "cuda"):
                if self.tag_client_lbulb == 0:
                    images, targets = self._labeled_batch(batch)
                    logits = self.model(images.to(self.device))
                    loss = self._supervised_loss(logits, targets.to(self.device), round_index)
                else:
                    indices, weak, strong = batch
                    loss, mask = self._unsupervised_loss(indices, weak.to(self.device),
                                                         strong.to(self.device), round_index)
                    accepted_samples += int(mask.sum())
                    seen_samples += len(mask)
                    loss = self._unlabeled_weight() * loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite {} loss on client {}".format(args.method, self.id_user))
            record_training_loss(self, loss)
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
            scaler.step(self.optimizer)
            scaler.update()
            self._after_step()
            losses.append(float(loss.detach()))
        self.optimizer.zero_grad(set_to_none=True)
        return {"client_id": self.id_user, "loss": sum(losses) / len(losses),
                "mask_ratio": accepted_samples / max(1, seen_samples),
                "sample_count": self.num_samples, "aggregation_count": self.aggregation_count()}

    def _supervised_loss(self, logits, targets, round_index):
        return functional.cross_entropy(logits.float(), targets)

    def _labeled_batch(self, batch):
        return batch[2], batch[3]

    def _unlabeled_weight(self):
        return self.args.ulb_loss_ratio

    def _after_step(self):
        pass

    def aggregation_count(self):
        return self.num_samples

    def upload_state(self):
        return self.model.state_dict()


class SAGEClient(BaselineClient):
    def _start_round(self, global_model):
        super()._start_round(global_model)
        if self.tag_client_lbulb == 2:
            self._new_teacher()

    def _unsupervised_loss(self, indices, weak, strong, round_index):
        local_logits, strong_logits = self.model(torch.cat((weak, strong), dim=0)).chunk(2)
        with torch.no_grad():
            global_logits = self.teacher(weak)
        loss, mask, _ = sage_consistency_loss(local_logits, global_logits, strong_logits,
                                             self.args.p_cutoff, self.args.T, self.args.sage_half_gap)
        return loss, mask


class FedPDMClient(BaselineClient):
    def __init__(self, client_id, tag, loaders, model, args, device):
        super().__init__(client_id, tag, loaders, model, args, device)
        self.prior = torch.full((args.num_classes,), 1.0 / args.num_classes)
        self.prediction_memory = None
        self.selected = None
        if tag == 0:
            labels = torch.as_tensor(self.loader.dataset.targets, dtype=torch.long)
            counts = torch.bincount(labels, minlength=args.num_classes).float()
            self.prior = counts / counts.sum()

    def _start_round(self, global_model):
        super()._start_round(global_model)
        self._new_teacher()
        if self.tag_client_lbulb == 2:
            if self.prediction_memory is None:
                self.prediction_memory = torch.full((self.num_samples, self.args.num_classes),
                                                    1.0 / self.args.num_classes)
            self.selected = torch.zeros(self.num_samples, dtype=torch.bool)

    def _adjust_logits(self, logits, round_index):
        rounds = self.args.pdm_rampup_rounds
        progress = min(1.0, (round_index + 1) / rounds) if rounds else 1.0
        adjustment = self.prior.to(logits.device).clamp_min(1e-6).log()
        return logits.float() + self.args.pdm_logit_strength * progress * adjustment

    def _supervised_loss(self, logits, targets, round_index):
        return functional.cross_entropy(self._adjust_logits(logits, round_index), targets)

    def _unsupervised_loss(self, indices, weak, strong, round_index):
        with torch.no_grad():
            probabilities = (self.teacher(weak).float() / self.args.T).softmax(dim=1).cpu()
            for position, sample_id in enumerate(indices.tolist()):
                self.prediction_memory[sample_id].copy_(probabilities[position])
            refined = match_class_distribution(self.prediction_memory, self.prior,
                                                self.args.pdm_sinkhorn_power,
                                                self.args.pdm_sinkhorn_iters)[indices]
            confidence, targets = refined.max(dim=1)
            mask = confidence >= self.args.p_cutoff
            for position, sample_id in enumerate(indices.tolist()):
                self.selected[sample_id] = mask[position]
        logits = self._adjust_logits(self.model(strong), round_index)
        loss = (functional.cross_entropy(logits, targets.to(self.device), reduction="none")
                * mask.to(self.device)).mean()
        with torch.no_grad():
            self.prior.mul_(self.args.pdm_prior_ema).add_(probabilities.mean(dim=0),
                                                        alpha=1 - self.args.pdm_prior_ema)
            self.prior.div_(self.prior.sum())
        return loss, mask

    @torch.no_grad()
    def _after_step(self):
        for teacher_parameter, parameter in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_parameter.mul_(self.args.ema_m).add_(parameter, alpha=1 - self.args.ema_m)
        for teacher_buffer, buffer in zip(self.teacher.buffers(), self.model.buffers()):
            teacher_buffer.copy_(buffer)

    def aggregation_count(self):
        return self.num_samples if self.tag_client_lbulb == 0 else int(self.selected.sum())

    def upload_state(self):
        return self.teacher.state_dict()
