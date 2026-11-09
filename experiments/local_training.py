"""Local budgets shared by the small-image epoch adapters."""

from contextlib import contextmanager

from torch.nn.modules.batchnorm import _BatchNorm


def uses_local_epochs(args):
    return getattr(args, "local_training_mode", "steps") == "epochs"


def local_loader_options(args, step_budget):
    if uses_local_epochs(args):
        return {"replacement": False, "num_epochs": 1, "drop_last": False}
    return {"num_iters": step_budget}


def local_batches(loader, args, iterator=None):
    if uses_local_epochs(args):
        if len(loader.dataset) == 0:
            raise ValueError("Epoch training requires a nonempty local dataset")
        for _ in range(args.local_epochs):
            yield from loader
    else:
        iterator = iter(loader) if iterator is None else iterator
        for _ in range(args.local_iterations):
            yield next(iterator)


@contextmanager
def singleton_batchnorm(model, enabled):
    """Use running statistics only for otherwise invalid singleton activations."""
    modules = [module for module in model.modules()
               if enabled and isinstance(module, _BatchNorm) and module.training]

    def use_running_statistics(module, inputs):
        if inputs[0].numel() // inputs[0].shape[1] == 1:
            module.eval()

    handles = [module.register_forward_pre_hook(use_running_statistics) for module in modules]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module in modules:
            module.train()
