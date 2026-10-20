"""Compatibility entrypoint; prefer train.py --method cbafed."""

from experiments.config import str2bool
from experiments.runner import cli, run


def main(args):
    args.method = "cbafed"
    run(args)


if __name__ == "__main__":
    cli(method="cbafed")
