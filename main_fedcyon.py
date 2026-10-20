"""Compatibility entrypoint; prefer train.py --method fedcyon."""

from experiments.config import str2bool
from experiments.runner import cli, run


def main(args):
    args.method = "fedcyon"
    run(args)


if __name__ == "__main__":
    cli(method="fedcyon")
