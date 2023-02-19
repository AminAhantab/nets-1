import argparse

from nets import __version__
from nets_cli.args import (
    InitArgs,
    PruneArgs,
    SearchArgs,
    TrainArgs,
    add_arguments,
)


def init_parser():
    # create parser
    parser = argparse.ArgumentParser(
        prog="nets",
        description="Neuroevolution Ticket Search (NeTS)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )

    return add_arguments(parser)


def main():
    try:
        import torch

        print("torch version:", torch.__version__)
    except ImportError:
        raise ImportError(
            "Please install torch to use nets. "
            "You can install torch via `pip install torch`."
        )

    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("cuda is not available")
        torch.set_default_tensor_type("torch.FloatTensor")

    parser = init_parser()
    args = parser.parse_args()

    subcommand = args.subcommand

    if subcommand == "init":
        init_args = InitArgs(**vars(args))
        print(init_args)
    elif subcommand == "search":
        search_args = SearchArgs(**vars(args))
        print(search_args)
    elif subcommand == "train":
        train_args = TrainArgs(**vars(args))
        print(train_args)
    elif subcommand == "prune":
        prune_args = PruneArgs(**vars(args))
        print(prune_args)
    else:
        raise ValueError(f"invalid command: {subcommand}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred.")
        print(e)
