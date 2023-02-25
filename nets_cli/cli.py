import argparse

from .args import (
    BaseArgs,
    InitArgs,
    IterativeMagnitudePruningArgs,
    PruneArgs,
    SearchArgs,
    TrainArgs,
    add_arguments,
)

from .config import configure_logger, configure_seed, configure_torch


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
    args: BaseArgs = parser.parse_args()

    subcommand = args.subcommand

    # Configure environment
    configure_logger(args)
    device = configure_torch()
    configure_seed(args.seed)

    from . import runners

    if subcommand == "init":
        init_args = InitArgs(**vars(args))
        runners.run_init(init_args)
    elif subcommand == "search":
        search_args = SearchArgs(**vars(args))
        runners.run_search(search_args, device)
    elif subcommand == "train":
        train_args = TrainArgs(**vars(args))
        runners.run_train(train_args, device)
    elif subcommand == "prune":
        prune_args = PruneArgs(**vars(args))
        runners.run_prune(prune_args)
    elif subcommand == "imp":
        imp_args = IterativeMagnitudePruningArgs(**vars(args))
        runners.run_imp(imp_args, device)
    else:
        raise ValueError(f"invalid command: {subcommand}")
