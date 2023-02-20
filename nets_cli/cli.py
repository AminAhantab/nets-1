import argparse

from nets_cli.args import (
    InitArgs,
    PruneArgs,
    SearchArgs,
    TrainArgs,
    add_arguments,
)
from nets_cli.runners import run_init, run_search, run_train, run_prune


def init_parser():
    # create parser
    parser = argparse.ArgumentParser(
        prog="nets",
        description="Neuroevolution Ticket Search (NeTS)",
    )

    # version
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + "v0.1.0",
    )

    return add_arguments(parser)


def main():
    parser = init_parser()
    args = parser.parse_args()

    subcommand = args.subcommand

    if subcommand == "init":
        init_args = InitArgs(**vars(args))
        run_init(init_args)
    elif subcommand == "search":
        search_args = SearchArgs(**vars(args))
        run_search(search_args)
    elif subcommand == "train":
        train_args = TrainArgs(**vars(args))
        run_train(train_args)
    elif subcommand == "prune":
        prune_args = PruneArgs(**vars(args))
        run_prune(prune_args)
    else:
        raise ValueError(f"invalid command: {subcommand}")
