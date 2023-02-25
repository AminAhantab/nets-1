from typing import Any, Callable, Tuple


def hydrate_new_model(
    architecture: str,
    dataset: str,
    density: float = 1.0,
    bias: bool = True,
) -> Any:
    from nets.utils import set_uniform_masks

    constructor = hydrate_architecture(architecture)
    dimensions = hydrate_data_dimensions(dataset)
    model = constructor(*dimensions, bias=bias)
    set_uniform_masks(model, density)
    return model


def hydrate_architecture(architecture: str) -> Callable[[int, int, int], Any]:
    """
    Hydrates an architecture name into a class.

    Args:
        architecture: The name of the architecture.

    Returns:
        The class corresponding to the architecture.
    """
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
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def hydrate_class_name(class_name: str) -> Callable[[int, int, int], Any]:
    """
    Hydrates a class name into a class.

    Args:
        class_name: The name of the class.

    Returns:
        The class corresponding to the class name.
    """
    if class_name == "LeNetFeedForwardNetwork":
        from models.lenet import LeNetFeedForwardNetwork

        return LeNetFeedForwardNetwork
    elif class_name == "ConvTwoNeuralNetwork":
        from models.conv import ConvTwoNeuralNetwork

        return ConvTwoNeuralNetwork
    elif class_name == "ConvFourNeuralNetwork":
        from models.conv import ConvFourNeuralNetwork

        return ConvFourNeuralNetwork
    elif class_name == "ConvSixNeuralNetwork":
        from models.conv import ConvSixNeuralNetwork

        return ConvSixNeuralNetwork
    else:
        raise ValueError(f"Unknown class name: {class_name}")


def hydrate_dataset(
    dataset: str,
    root_dir: str = "./data",
    download: bool = True,
    val_size: int = 5_000,
    generator=None,
) -> Tuple[Any, Any, Any]:
    """
    Hydrates a dataset name into the corresponding datasets.

    Args:
        dataset: The name of the dataset.

    Returns:
        The datasets corresponding to the dataset.
    """
    if dataset == "mnist":
        from data.mnist import load

        return load(root_dir, download=download, val_size=val_size, generator=generator)
    elif dataset == "cifar10":
        from data.cifar10 import load

        return load(root_dir, download=download, val_size=val_size, generator=generator)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def hydrate_data_dimensions(dataset: str) -> Tuple[int, Tuple[int, int], int]:
    """
    Hydrates a dataset name into the corresponding data dimensions.

    Args:
        dataset: The name of the dataset.

    Returns:
        The data dimensions corresponding to the dataset.
    """
    if dataset == "mnist":
        return 1, (28, 28), 10
    elif dataset == "cifar10":
        return 3, (32, 32), 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def hydrate_optimiser(optimiser: str) -> Callable[[Any], Any]:
    """
    Hydrates an optimiser name into the corresponding optimiser.

    Args:
        optimiser: The name of the optimiser.

    Returns:
        The optimiser corresponding to the optimiser name.
    """
    if optimiser == "adam":
        from torch.optim import Adam

        return Adam
    elif optimiser == "sgd":
        from torch.optim import SGD

        return SGD
    else:
        raise ValueError(f"Unknown optimiser: {optimiser}")


def hydrate_prune_method(
    criterion: str,
    threshold: float,
    count: int,
    fraction: float,
) -> Callable:
    if criterion == "magnitude":
        from nets.nn.prune import prune_magnitude

        # TODO: Implement other pruning methods
        if threshold is not None or count is not None or fraction is None:
            raise NotImplementedError()

        def prune_magnitude_fn(model):
            return prune_magnitude(model, fraction)

        return prune_magnitude_fn
    elif criterion == "random":
        from nets.nn.prune import prune_random

        def prune_random_fn(model):
            return prune_random(model, count=count, fraction=fraction)

        return prune_random_fn
    else:
        raise ValueError(f"Invalid criterion: {criterion}")
