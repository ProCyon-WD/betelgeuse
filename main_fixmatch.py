"""Compatibility entrypoint; prefer train.py --method fixmatch."""

from experiments.config import str2bool
from experiments.runner import cli, run


def main(args):
    args.method = "fixmatch"
    run(args)


if __name__ == "__main__":
    cli(method="fixmatch")
