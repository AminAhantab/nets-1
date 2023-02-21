import os
from typing import Any, Callable, Tuple


def is_dir(path: str) -> bool:
    """
    Checks whether a path is a directory.

    Args:
        path: The path to check.

    Returns:
        Whether the path is a directory.
    """
    if path.endswith(".pt"):
        return False

    return True


def create_path(path: str, file_name: str = "model.pt") -> str:
    """
    Creates a path to a file iff the parent directory exists (or can be
    created).

    Args:
        path: The path to the file.
        file_name: The name of the file.

    Returns:
        The full path to the file.
    """
    if is_dir(path):
        os.makedirs(path, exist_ok=True)
        if not path.endswith(os.path.sep):
            path = path + os.path.sep

        path = os.path.join(path, file_name)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


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
