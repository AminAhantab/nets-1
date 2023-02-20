import logging
from typing import Callable, Tuple

from nets.nn.masked import MaskedNetwork
from nets_cli.args import InitArgs
from .common import configure_logger, configure_seed, configure_torch, write_model

logger = logging.getLogger("nets_cli.init")


def run_init(args: InitArgs):
    # Configure environment
    configure_logger(args)
    configure_torch()
    configure_seed(args.seed)

    # Get relevant arguments
    dataset = args.dataset
    architecture = args.architecture
    density = args.density

    # Initialise and save model
    model = init_model(dataset, architecture, density)
    write_model(model, args.out_path, file_name="init.pt", overwrite=True)


def init_model(dataset: str, architecture: str, density: float) -> MaskedNetwork:
    # Initialise model
    logger.info(
        "Initialising a %s (density %.0f%%) for learning %s classifications.",
        architecture,
        density * 100,
        dataset,
    )

    init_fn = get_init_fn(architecture)
    channels, in_features, out_features = get_dimensions(dataset)

    if architecture == "lenet":
        model = init_fn(in_features, out_features)
    elif architecture.startswith("conv"):
        model = init_fn(channels, out_features)
    else:
        raise NotImplementedError

    logger.debug("Initialised model: %s", model)
    return model


def get_init_fn(architecture: str) -> Callable[..., MaskedNetwork]:
    if architecture == "lenet":
        from models.lenet import LeNetFeedForwardNetwork

        return LeNetFeedForwardNetwork
    elif architecture == "conv-2":
        from models.conv import ConvTwoNeuralNetwork

        return ConvTwoNeuralNetwork
    elif architecture == "conv-4":
        from models.conv import ConvFourNeuralNetwork

        return ConvFourNeuralNetwork
    elif architecture == "conv-6":
        from models.conv import ConvSixNeuralNetwork

        return ConvSixNeuralNetwork
    elif architecture == "resnet-18":
        raise NotImplementedError("resnet-18 architecture not implemented yet.")
    elif architecture == "vgg-19":
        raise NotImplementedError("vgg-19 architecture not implemented yet.")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_dimensions(dataset: str) -> Tuple[int, int]:
    if dataset == "mnist":
        return 1, 28 * 28, 10
    elif dataset == "cifar10":
        return 3, 35 * 35, 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
