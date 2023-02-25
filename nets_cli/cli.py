import argparse

from nets_cli.args import (
    InitArgs,
    IterativeMagnitudePruningArgs,
    PruneArgs,
    SearchArgs,
    TrainArgs,
    add_arguments,
)
import nets_cli.runners as runners


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
        runners.run_init(init_args)
    elif subcommand == "search":
        search_args = SearchArgs(**vars(args))
        runners.run_search(search_args)
    elif subcommand == "train":
        train_args = TrainArgs(**vars(args))
        runners.run_train(train_args)
    elif subcommand == "prune":
        prune_args = PruneArgs(**vars(args))
        runners.run_prune(prune_args)
    elif subcommand == "imp":
        imp_args = IterativeMagnitudePruningArgs(**vars(args))
        runners.run_imp(imp_args)
    else:
        raise ValueError(f"invalid command: {subcommand}")
