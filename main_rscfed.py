"""Compatibility entrypoint; prefer train.py --method rscfed."""

from experiments.config import str2bool
from experiments.runner import cli, run


def main(args):
    args.method = "rscfed"
    run(args)


if __name__ == "__main__":
    cli(method="rscfed")
