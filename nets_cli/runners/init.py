import logging
from typing import Callable, Tuple

import torch
from nets.nn.masked import MaskedNetwork
from nets_cli.args import InitArgs
from .common import configure_logger, configure_seed, write_model


def run_init(args: InitArgs) -> None:
    # Configure logger
    configure_logger(args)
    logger = logging.getLogger("nets_cli.init")

    # Set random seed
    seed = args.seed
    configure_seed(seed)

    # Get relevant arguments
    dataset = args.dataset
    architecture = args.architecture
    density = args.density

    # Initialise model
    den = f"{(density * 100):.0f}%"
    logger.info(f"Initialising {architecture} on {dataset} with density {den}.")

    init_fn = get_init_fn(architecture)
    model = init_fn(28 * 28, 10)
    logger.info(f"Initialised model: {model}")

    # Save model
    write_model(model, args.out_path, file_name="init.pt")


def get_init_fn(architecture: str) -> Callable[[int, int], MaskedNetwork]:
    if architecture == "lenet":
        from models.lenet import LeNetFeedForwardNetwork

        return LeNetFeedForwardNetwork
    elif architecture == "conv-2":
        raise NotImplementedError("conv-2 architecture not implemented yet.")
    elif architecture == "conv-4":
        raise NotImplementedError("conv-4 architecture not implemented yet.")
    elif architecture == "conv-6":
        raise NotImplementedError("conv-6 architecture not implemented yet.")
    elif architecture == "resnet-18":
        raise NotImplementedError("resnet-18 architecture not implemented yet.")
    elif architecture == "vgg-19":
        raise NotImplementedError("vgg-19 architecture not implemented yet.")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_dimensions(dataset: str) -> Tuple[int, int]:
    if dataset == "mnist":
        return 28 * 28, 10
    elif dataset == "cifar10":
        return 32 * 32 * 3, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
